"""
multi_trade_backtest.py
═══════════════════════════════════════════════════════════════════════════════
PURE HISTORICAL BACKTEST  —  zero lookahead bias
───────────────────────────────────────────────────────────────────────────────
Agents (2, parallel):
  • Technicals  — RSI, MACD, SMA-50/200, BB, ATR, momentum (point-in-time)
  • Quant/Macro — GLASSO long-short weights + macro ETF snapshot (point-in-time)

Then:
  • Risk    — synthesizes both into per-ticker conviction scores
  • Trader  — emits structured JSON trade book (LONG / SHORT / SKIP)

Intentionally EXCLUDED to avoid lookahead bias:
  ✗ Finnhub fundamentals  → returns TODAY's P/E and market cap
  ✗ News / sentiment      → yfinance .news returns TODAY's headlines

Graph:  START → [technicals ‖ quant] → risk → trader → END
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import json
import datetime
import numpy as np
import yfinance as yf
import pandas_ta as ta
from dateutil.relativedelta import relativedelta
from typing import TypedDict
from dotenv import load_dotenv
from sklearn.covariance import GraphicalLasso
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent as create_react_agent

load_dotenv()


# =============================================================================
# 1. STATE
# =============================================================================
class BacktestState(TypedDict):
    tickers:            list[str]
    target_date:        str
    technical_analysis: str
    quant_analysis:     str
    risk_analysis:      str
    final_decision:     str
    trade_book:         dict


# =============================================================================
# 2. TOOLS — 100% point-in-time via yfinance historical data
# =============================================================================

@tool
def get_technicals(ticker: str, target_date: str) -> str:
    """
    Point-in-time technicals ending strictly at target_date:
    RSI-14, MACD, SMA-50/200, Bollinger Bands, ATR-14, 1M/3M momentum.
    """
    end   = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(months=9)

    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
    )
    if df.empty:
        return f"No price data for {ticker}."

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50,  append=True)
    df.ta.sma(length=200, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    last  = df.iloc[-1]
    close = float(last["Close"])

    rsi    = float(last.get("RSI_14",       50))
    macd_v = float(last.get("MACD_12_26_9",  0))
    macd_s = float(last.get("MACDs_12_26_9", 0))
    sma50  = float(last.get("SMA_50",  close))
    sma200 = float(last.get("SMA_200", close))
    bbu    = float(last.get("BBU_20_2.0", close))
    bbl    = float(last.get("BBL_20_2.0", close))
    atr    = float(last.get("ATRr_14", 0))

    mom1m = ((close - float(df["Close"].iloc[-22])) / float(df["Close"].iloc[-22]) * 100) if len(df) > 22 else 0
    mom3m = ((close - float(df["Close"].iloc[-63])) / float(df["Close"].iloc[-63]) * 100) if len(df) > 63 else 0

    return (
        f"{ticker} @ {target_date} | Price: {close:.2f} | "
        f"RSI: {rsi:.1f} ({'Overbought' if rsi>70 else 'Oversold' if rsi<30 else 'Neutral'}) | "
        f"MACD: {macd_v:.3f} vs Signal: {macd_s:.3f} ({'Bull' if macd_v>macd_s else 'Bear'} cross) | "
        f"SMA50: {sma50:.2f}  SMA200: {sma200:.2f} ({'Above' if close>sma200 else 'Below'} 200d) | "
        f"BB: [{bbl:.2f}–{bbu:.2f}] | ATR: {atr:.3f} | "
        f"Mom 1M: {mom1m:+.1f}%  Mom 3M: {mom3m:+.1f}%"
    )


@tool
def calculate_markowitz_long_short(tickers: list[str], target_date: str) -> str:
    """
    GLASSO precision-matrix optimization with long-short weights.
    Uses 2 years of price history ending at target_date.
    Long book normalized to 100%, short book capped at -30%.
    """
    if len(tickers) < 2:
        return "Need at least 2 tickers."

    end   = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(years=2)

    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )["Close"].dropna(axis=1, how="all")

        valid = list(raw.columns)
        rets  = raw.pct_change().dropna()
        cent  = rets - rets.mean()

        glasso = GraphicalLasso(alpha=0.5, max_iter=200)
        glasso.fit(cent)
        prec = glasso.precision_

        ones = np.ones(len(valid))
        raw_w = prec @ ones / (ones @ prec @ ones)

        longs  = np.where(raw_w > 0, raw_w, 0.0)
        shorts = np.where(raw_w < 0, raw_w, 0.0)

        if longs.sum()  > 0: longs  = longs  / longs.sum()
        if shorts.sum() < 0: shorts = shorts / abs(shorts.sum()) * 0.30

        final_w = longs + shorts
        lines = [f"GLASSO Long-Short Weights as of {target_date} (2yr, alpha=0.5):"]
        for t, w in sorted(zip(valid, final_w), key=lambda x: -x[1]):
            label = "LONG " if w > 0.001 else "SHORT" if w < -0.001 else "SKIP "
            lines.append(f"  {label} {t:<6}: {w*100:>+7.2f}%")
        return "\n".join(lines)

    except Exception as e:
        return f"GLASSO failed: {e}"


@tool
def get_macro_context(target_date: str) -> str:
    """
    Point-in-time macro snapshot using ETF proxies:
    SPY, VIX, DXY, TLT, and sector ETFs — 1-month change ending at target_date.
    """
    end   = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(months=3)
    s, e  = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    proxies = {
        "SPY": "S&P 500", "^VIX": "VIX Fear", "DX-Y.NYB": "US Dollar",
        "TLT": "20Y Bonds", "XLK": "Tech", "XLF": "Financials",
        "XLE": "Energy", "XLV": "Healthcare", "XLU": "Utilities",
    }
    lines = [f"Macro snapshot as of {target_date}:"]
    for sym, label in proxies.items():
        try:
            px = yf.download(sym, start=s, end=e, progress=False)["Close"].dropna()
            if len(px) < 5:
                continue
            cur  = float(px.iloc[-1])
            prev = float(px.iloc[-22]) if len(px) >= 22 else float(px.iloc[0])
            chg  = (cur - prev) / prev * 100
            lines.append(f"  {label:<20} ({sym:<10}): {cur:>9.2f}  {'↑' if chg>0 else '↓'} {chg:>+5.1f}% (1M)")
        except Exception:
            pass
    return "\n".join(lines)


# =============================================================================
# 3. LLM + AGENTS
# =============================================================================
llm = ChatOllama(model="phi3", temperature=0.1)

tech_agent  = create_react_agent(llm, tools=[get_technicals],
    system_prompt=(
        "You are a senior technical analyst. "
        "For each ticker call get_technicals and give a clear LONG/SHORT/NEUTRAL verdict "
        "with momentum and trend summary. Be concise."
    ))

quant_agent = create_react_agent(llm, tools=[calculate_markowitz_long_short, get_macro_context],
    system_prompt=(
        "You are a quantitative analyst. "
        "Step 1: call get_macro_context. "
        "Step 2: call calculate_markowitz_long_short. "
        "Summarize macro regime and per-ticker LONG/SHORT/SKIP quant signal."
    ))


def _is_tool_error(exc: Exception) -> bool:
    return "does not support tools" in str(exc).lower()


# =============================================================================
# 4. NODES
# =============================================================================

def node_technicals(state: BacktestState):
    prompt = (
        f"Today is {state['target_date']}. "
        f"Analyze technicals for each ticker: {state['tickers']}. "
        f"Pass target_date='{state['target_date']}' to every tool call."
    )
    try:
        r = tech_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"technical_analysis": r["messages"][-1].content}
    except Exception as exc:
        if not _is_tool_error(exc): raise
        reports = [get_technicals.invoke({"ticker": t, "target_date": state["target_date"]}) for t in state["tickers"]]
        return {"technical_analysis": "\n".join(reports)}


def node_quant(state: BacktestState):
    prompt = (
        f"Today is {state['target_date']}. "
        f"Step 1: get_macro_context(target_date='{state['target_date']}'). "
        f"Step 2: calculate_markowitz_long_short(tickers={state['tickers']}, target_date='{state['target_date']}')."
    )
    try:
        r = quant_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"quant_analysis": r["messages"][-1].content}
    except Exception as exc:
        if not _is_tool_error(exc): raise
        macro = get_macro_context.invoke({"target_date": state["target_date"]})
        wts   = calculate_markowitz_long_short.invoke({"tickers": state["tickers"], "target_date": state["target_date"]})
        return {"quant_analysis": macro + "\n\n" + wts}


def node_risk(state: BacktestState):
    messages = [
        {"role": "system", "content": (
            "You are the Chief Risk Officer. Synthesize the two analyst reports. "
            "For EACH ticker give a conviction score -3 (strong short) to +3 (strong long) "
            "with one sentence reasoning. State overall portfolio risk: LOW / MEDIUM / HIGH. "
            "Be objective — do not default bullish or bearish."
        )},
        {"role": "user", "content": (
            f"Date: {state['target_date']}\nTickers: {state['tickers']}\n\n"
            f"=== TECHNICALS ===\n{state.get('technical_analysis','N/A')}\n\n"
            f"=== QUANT / MACRO ===\n{state.get('quant_analysis','N/A')}\n\n"
            "Provide per-ticker conviction scores and overall risk level."
        )},
    ]
    r = llm.invoke(messages)
    return {"risk_analysis": r.content}


def node_trader(state: BacktestState):
    messages = [
        {"role": "system", "content": (
            "You are the Head Trader. Full autonomy: LONG, SHORT, or SKIP any ticker. "
            "Output ONLY valid JSON — no markdown, no extra text.\n"
            'Format: {"rationale":"...","trades":{"TICKER":{"action":"LONG|SHORT|SKIP","weight":0.00,"reason":"..."}}}\n'
            "Rules: weight is decimal (0.15=15%). Negative for shorts (e.g. -0.10). "
            "LONG weights sum 0.80–1.00. SHORT weights sum -0.30–0.00. SKIP weight=0. "
            "HIGH risk: reduce longs, increase shorts. Use GLASSO as starting point."
        )},
        {"role": "user", "content": (
            f"Date: {state['target_date']}\nTickers: {', '.join(state['tickers'])}\n\n"
            f"=== TECHNICALS ===\n{state.get('technical_analysis','N/A')}\n\n"
            f"=== QUANT / MACRO ===\n{state.get('quant_analysis','N/A')}\n\n"
            f"=== RISK SYNTHESIS ===\n{state.get('risk_analysis','N/A')}\n\n"
            "Output ONLY the JSON trade book."
        )},
    ]
    r     = llm.invoke(messages)
    clean = re.sub(r"```(?:json)?|```", "", r.content.strip()).strip()
    trade_book: dict = {}
    rationale: str   = r.content

    try:
        parsed    = json.loads(clean)
        trades    = parsed.get("trades", {})
        rationale = parsed.get("rationale", r.content)

        l_total = sum(v["weight"] for v in trades.values() if v.get("weight", 0) > 0)
        s_total = sum(v["weight"] for v in trades.values() if v.get("weight", 0) < 0)
        if l_total > 1.05:
            for k in trades:
                if trades[k]["weight"] > 0: trades[k]["weight"] = round(trades[k]["weight"] / l_total, 4)
        if s_total < -0.35:
            for k in trades:
                if trades[k]["weight"] < 0: trades[k]["weight"] = round(trades[k]["weight"] / abs(s_total) * 0.30, 4)
        trade_book = trades

    except (json.JSONDecodeError, KeyError, TypeError):
        eq_w = round(1.0 / len(state["tickers"]), 4)
        trade_book = {t: {"action": "LONG", "weight": eq_w, "reason": "JSON parse fallback"} for t in state["tickers"]}

    return {"final_decision": rationale, "trade_book": trade_book}


# =============================================================================
# 5. GRAPH
# =============================================================================
graph = StateGraph(BacktestState)
graph.add_node("technicals", node_technicals)
graph.add_node("quant",      node_quant)
graph.add_node("risk",       node_risk)
graph.add_node("trader",     node_trader)

graph.add_edge(START, "technicals")
graph.add_edge(START, "quant")
graph.add_edge(["technicals", "quant"], "risk")
graph.add_edge("risk",   "trader")
graph.add_edge("trader", END)

app = graph.compile()


# =============================================================================
# 6. BACKTEST RUNNER
# =============================================================================
if __name__ == "__main__":
    print("🚀  Multi-Agent Backtest  —  Zero Lookahead Bias")
    print("    Agents: Technicals ‖ Quant/Macro → Risk → Trader\n")

    tickers_to_test = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",   # Mega-cap tech
        "TSLA", "AMD",  "SHOP", "PLTR",                      # High-beta growth
        "JPM",  "GS",   "V",                                  # Financials
        "XOM",  "CVX",                                        # Energy
        "JNJ",  "PG",   "UNH",                               # Defensives
        "GLD",  "TLT",  "SHV",  "QQQ",  "SPY",              # Macro ETFs
    ]

    starting_capital = 10_000.0
    current_capital  = starting_capital
    backtest_dates   = [
        "2023-03-01",  # Post-SVB banking stress
        "2023-07-01",  # Summer rally
        "2023-10-01",  # Q4 rate fears
        "2024-01-01",  # New-year momentum
        "2024-04-01",  # Rate-cut pushback
        "2024-07-01",  # AI mania / rotation
    ]
    trade_log = []

    for date in backtest_dates:
        print(f"\n{'='*75}\n📅  {date}\n{'='*75}")

        final_state    = app.invoke({"tickers": tickers_to_test, "target_date": date})
        trade_book     = final_state.get("trade_book", {})
        final_decision = final_state.get("final_decision", "")

        print(f"\n🤖  Trader Rationale:\n{'-'*65}\n{final_decision}\n{'-'*65}")

        next_month = (datetime.datetime.strptime(date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")

        print(f"\n{'Ticker':<7} {'Action':<6} {'Weight':>7}  {'Buy @':>8} {'Sell @':>8}  {'Ret%':>7}    {'PnL $':>9}  Reason")
        print("-" * 95)

        month_rows = []
        for ticker in sorted(set(list(trade_book.keys()) + tickers_to_test)):
            entry  = trade_book.get(ticker, {"action": "SKIP", "weight": 0.0, "reason": ""})
            action = str(entry.get("action", "SKIP")).upper()
            weight = float(entry.get("weight", 0.0))
            reason = str(entry.get("reason", ""))[:40]

            if action == "SKIP" or abs(weight) < 0.001:
                print(f"{ticker:<7} {'SKIP':<6} {'':>7}  {'':>8} {'':>8}  {'':>7}    {'':>9}  {reason}")
                continue

            try:
                px = yf.download(ticker, start=date, end=next_month, progress=False)["Close"]
                if px.empty or len(px) < 2:
                    print(f"{ticker:<7} ⚠️  no data"); continue

                buy_px  = float(px.iloc[0].squeeze())
                sell_px = float(px.iloc[-1].squeeze())
                raw_ret = (sell_px - buy_px) / buy_px
                actual_ret = max(-raw_ret if action == "SHORT" else raw_ret, -0.08)

                allocated = current_capital * abs(weight)
                pnl       = allocated * actual_ret
                arrow     = "📈" if actual_ret > 0 else "📉"

                print(f"{ticker:<7} {action:<6} {weight*100:>+6.1f}%  {buy_px:>8.2f} {sell_px:>8.2f}  {actual_ret*100:>+6.2f}% {arrow}   {pnl:>+8.2f}  {reason}")
                month_rows.append({"date": date, "ticker": ticker, "action": action, "weight": weight,
                                   "buy_price": buy_px, "sell_price": sell_px, "return_pct": actual_ret * 100, "pnl": pnl})
            except Exception as e:
                print(f"{ticker:<7} ⚠️  {e}")

        if month_rows:
            total_pnl    = sum(r["pnl"] for r in month_rows)
            prev_capital = current_capital
            current_capital += total_pnl
            month_ret    = total_pnl / prev_capital * 100
            longs_pnl    = sum(r["pnl"] for r in month_rows if r["action"] == "LONG")
            shorts_pnl   = sum(r["pnl"] for r in month_rows if r["action"] == "SHORT")

            print("-" * 95)
            print(f"{'MONTH':<14} {month_ret:>+6.2f}%    {total_pnl:>+8.2f}  (longs ${longs_pnl:>+,.2f} | shorts ${shorts_pnl:>+,.2f})")
            print(f"\n💰  Portfolio: ${current_capital:,.2f}  ({'▲' if total_pnl>=0 else '▼'} ${abs(total_pnl):,.2f})")
            trade_log.extend(month_rows)

    total_return = (current_capital - starting_capital) / starting_capital * 100
    total_pnl    = current_capital - starting_capital

    print(f"\n{'='*75}\n🏁  BACKTEST COMPLETE")
    print(f"    Starting : ${starting_capital:>10,.2f}")
    print(f"    Ending   : ${current_capital:>10,.2f}")
    print(f"    P&L      : ${total_pnl:>+10,.2f}")
    print(f"    Return   : {total_return:>+.2f}%")
    print(f"{'='*75}")

    if trade_log:
        best  = max(trade_log, key=lambda x: x["pnl"])
        worst = min(trade_log, key=lambda x: x["pnl"])
        print(f"\n🥇  Best  : {best['ticker']}  {best['date']}  ({best['action']}) → {best['return_pct']:>+.2f}%   ${best['pnl']:>+,.2f}")
        print(f"💀  Worst : {worst['ticker']}  {worst['date']}  ({worst['action']}) → {worst['return_pct']:>+.2f}%   ${worst['pnl']:>+,.2f}")

        def _stats(rows, label):
            if not rows: return
            t   = sum(r["pnl"] for r in rows)
            avg = sum(r["return_pct"] for r in rows) / len(rows)
            w   = sum(1 for r in rows if r["pnl"] > 0)
            print(f"   {label:<30}: {len(rows):>3} trades | win {w/len(rows)*100:.0f}% | avg {avg:>+.2f}% | PnL ${t:>+,.2f}")

        print("\n📋  Breakdown:")
        _stats([r for r in trade_log if r["action"] == "LONG"],  "LONG positions")
        _stats([r for r in trade_log if r["action"] == "SHORT"], "SHORT positions")