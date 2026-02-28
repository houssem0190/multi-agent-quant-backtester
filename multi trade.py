import os
import datetime
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
from dateutil.relativedelta import relativedelta
from typing import TypedDict
from dotenv import load_dotenv
from sklearn.covariance import GraphicalLasso
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent as create_react_agent
# 1. Load Environment Variables
load_dotenv()

# 2. Define the State (Added target_date)
class InitialState(TypedDict):
    tickers: list[str]
    target_date: str
    fundamental_analysis: str
    technical_analysis: str
    sentiment_analysis: str
    optimal_weights: str
    market_direction: str
    final_decision: str

# ---------------------------------------------------------
# 3. DEFINE TOOLS (Updated for Point-in-Time Data)
# ---------------------------------------------------------
@tool
def get_fundamentals(ticker: str) -> str:
    """Gets company fundamentals from Finnhub."""
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_api_key:
        return "FINNHUB_API_KEY is missing."

    symbol = ticker.strip().upper()
    base_url = "https://finnhub.io/api/v1"
    params = {"symbol": symbol, "token": finnhub_api_key}

    try:
        profile_resp = requests.get(f"{base_url}/stock/profile2", params=params, timeout=15)
        metrics_resp = requests.get(f"{base_url}/stock/metric", params={**params, "metric": "all"}, timeout=15)
    except requests.RequestException as exc:
        return f"Finnhub request failed for {symbol}: {exc}"

    profile = profile_resp.json() or {}
    metrics_data = metrics_resp.json() or {}
    metric = metrics_data.get("metric", {}) if isinstance(metrics_data, dict) else {}

    company_name = profile.get("name") or symbol
    industry = profile.get("finnhubIndustry") or "N/A"
    market_cap = profile.get("marketCapitalization", "N/A")
    pe_ttm = metric.get("peTTM", "N/A")
    
    return f"{company_name} ({symbol}) fundamentals | Industry: {industry}, Market Cap: {market_cap}M, P/E TTM: {pe_ttm}."

@tool
def get_technicals(ticker: str, target_date: str) -> str:
    """Calculates RSI, MACD, and SMAs stopping exactly at the target_date."""
    end_date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start_date_obj = end_date_obj - relativedelta(months=6)
    
    df = yf.Ticker(ticker).history(start=start_date_obj.strftime("%Y-%m-%d"), end=end_date_obj.strftime("%Y-%m-%d"))
    if df.empty:
        return f"No historical data found for {ticker}."

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50, append=True)
    
    latest = df.iloc[-1]
    rsi = latest.get('RSI_14', 50)
    macd = latest.get('MACD_12_26_9', 0)
    signal = latest.get('MACDs_12_26_9', 0)
    
    status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    trend = "Bullish Cross" if macd > signal else "Bearish Cross"
    
    return f"{ticker} Technicals on {target_date}: RSI is {rsi:.2f} ({status}). MACD is {macd:.2f} with a {trend}."

@tool
def get_sentiment(ticker: str) -> str:
    """Fetches the most recent news headlines for a given stock ticker."""
    try:
        news_items = yf.Ticker(ticker).news
        if not news_items:
            return f"No recent news found for {ticker}."
        report = f"Latest Headlines for {ticker}:\n"
        for item in news_items[:5]:
            report += f"- {item.get('title', 'No Title')}\n"
        return report
    except Exception as e:
        return f"Failed to fetch news for {ticker}: {str(e)}"

@tool
def calculate_markowitz(tickers: list[str], target_date: str) -> str:
    """Calculates GLASSO optimal weights using 1 year of data ending on the target_date."""
    try:
        if len(tickers) < 2:
            return "Error: Markowitz optimization requires at least 2 tickers."

        end_date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - relativedelta(years=1)
        
        data = yf.download(tickers, start=start_date_obj.strftime("%Y-%m-%d"), end=end_date_obj.strftime("%Y-%m-%d"))['Close']
        returns = data.pct_change().dropna()
        returns_centered = returns - returns.mean()

        glasso = GraphicalLasso(alpha=0.6, max_iter=100)
        glasso.fit(returns_centered)
        precision_matrix = glasso.precision_

        ones = np.ones(len(tickers))
        raw_weights = precision_matrix.dot(ones) / ones.dot(precision_matrix).dot(ones)

        weight_dict = dict(zip(tickers, raw_weights))
        report = f"GLASSO-Optimized Weights as of {target_date} (alpha=0.6):\n"
        for ticker, weight in weight_dict.items():
            report += f"- {ticker}: {weight * 100:.2f}%\n"
            
        return report
    except Exception as e:
        return f"GLASSO Optimization failed: {str(e)}"

# ---------------------------------------------------------
# 4. INITIALIZE LLM AND AGENTS
# ---------------------------------------------------------
llm = ChatOllama(model="phi3", temperature=0.1)

fund_agent = create_react_agent(llm, tools=[get_fundamentals], system_prompt="You are a fundamental analyst.")
tech_agent = create_react_agent(llm, tools=[get_technicals], system_prompt="You are a technical analyst.")
sent_agent = create_react_agent(llm, tools=[get_sentiment], system_prompt="You are a sentiment analyst.")
quant_agent = create_react_agent(llm, tools=[calculate_markowitz], system_prompt="You are a quantitative analyst.")


def _is_tool_support_error(exc: Exception) -> bool:
    """Detect Ollama tool-calling compatibility errors."""
    return "does not support tools" in str(exc).lower()

# ---------------------------------------------------------
# 5. WRAPPER NODES (Injecting the target_date)
# ---------------------------------------------------------
def fundamentals(state: InitialState):
    prompt = f"Pretend today is {state['target_date']}. Analyze the fundamentals for {state['tickers']}."
    try:
        response = fund_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"fundamental_analysis": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        reports = [get_fundamentals.invoke({"ticker": t}) for t in state["tickers"]]
        return {"fundamental_analysis": "\n".join(reports)}

def technicals(state: InitialState):
    prompt = f"Pretend today is {state['target_date']}. Analyze the technicals for {state['tickers']} passing the target_date '{state['target_date']}' to your tool."
    try:
        response = tech_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"technical_analysis": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        reports = [
            get_technicals.invoke({"ticker": t, "target_date": state["target_date"]})
            for t in state["tickers"]
        ]
        return {"technical_analysis": "\n".join(reports)}

def sentiment(state: InitialState): 
    prompt = f"Pretend today is {state['target_date']}. Analyze the sentiment for {state['tickers']}."
    try:
        response = sent_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"sentiment_analysis": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        reports = [get_sentiment.invoke({"ticker": t}) for t in state["tickers"]]
        return {"sentiment_analysis": "\n".join(reports)}

def risk(state: InitialState):
    message = f"""You are a risk analyst on {state['target_date']}. Review these reports for {state['tickers']}:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    If the overall consensus is clearly negative/bearish, output EXACTLY the word "BEARISH".
    Otherwise, if it is neutral or positive, output EXACTLY the word "BULLISH"."""
    
    response = llm.invoke(message)
    raw_direction = response.content.strip().upper()
    direction = "BEARISH" if "BEARISH" in raw_direction else "BULLISH"
    return {"market_direction": direction}  

def route_market_direction(state: InitialState) -> str:
    if state.get("market_direction") == "BEARISH":
        return "short"
    return "long" 

def markowitz(state: InitialState):
    prompt = f"Calculate the optimal portfolio weights for {state['tickers']} passing the target_date '{state['target_date']}' to your tool."
    try:
        response = quant_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"optimal_weights": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        return {
            "optimal_weights": calculate_markowitz.invoke(
                {"tickers": state["tickers"], "target_date": state["target_date"]}
            )
        }

def trader_node(state: InitialState):
    prompt = f"""You are a head trader on {state['target_date']}. Review these reports and make a final LONG decision:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    Quant: {state.get('optimal_weights')}"""
    response = llm.invoke(prompt)
    return {"final_decision": response.content}

def short_trader_node(state: InitialState):
    prompt = f"""You are a head trader on {state['target_date']}. Review these reports and make a final SHORT decision:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}"""
    response = llm.invoke(prompt)
    return {"final_decision": f"[SHORT STRATEGY TRIGGERED]\n{response.content}"}

# ---------------------------------------------------------
# 6. BUILD AND COMPILE THE GRAPH
# ---------------------------------------------------------
graph = StateGraph(InitialState)

graph.add_node("fundamental_analysis", fundamentals)
graph.add_node("technical_analysis", technicals)
graph.add_node("sentiment_analysis", sentiment)
graph.add_node("risk_analysis", risk)
graph.add_node("markowitz", markowitz)
graph.add_node("trader", trader_node)
graph.add_node("short_trader", short_trader_node)

graph.add_edge(START, "fundamental_analysis")
graph.add_edge(START, "technical_analysis")    
graph.add_edge(START, "sentiment_analysis")
graph.add_edge(['fundamental_analysis', 'technical_analysis', 'sentiment_analysis'], 'risk_analysis')
graph.add_conditional_edges('risk_analysis', route_market_direction, {'short': 'short_trader', 'long': 'markowitz'})
graph.add_edge('markowitz', 'trader')
graph.add_edge("trader", END)
graph.add_edge("short_trader", END)

app = graph.compile()

# ---------------------------------------------------------
# 7. RUN / TEST THE CODE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Local LLM Historical Backtest...")
    
    tickers_to_test = ["AAPL", "MSFT", "NVDA"]
    starting_capital = 10000.0
    current_capital = starting_capital
    
    # Test a sequence of months 
    backtest_dates = ["2023-08-01", "2023-09-01", "2023-10-01"]
    
    for date in backtest_dates:
        print(f"\n{'='*60}\n📅 BACKTEST DATE: {date}\n{'='*60}")
        
        inputs = {
            "tickers": tickers_to_test,
            "target_date": date
        }
        
        final_state = app.invoke(inputs)
        
        regime = final_state.get('market_direction')
        print(f"\n🏆 Final Decision ({date})")
        print(f"Detected Regime: {regime}")
        print("-" * 40)
        
        # PnL Calculation
        try:
            next_month_obj = datetime.datetime.strptime(date, "%Y-%m-%d") + relativedelta(months=1)
            next_month_str = next_month_obj.strftime("%Y-%m-%d")
            
            proxy_data = yf.download("QQQ", start=date, end=next_month_str)['Close']
            
            if not proxy_data.empty:
                start_price = float(proxy_data.iloc[0].squeeze())
                end_price = float(proxy_data.iloc[-1].squeeze())
                
                month_return = (end_price - start_price) / start_price
                
                if regime == "BEARISH":
                    month_return = -month_return 
                    
                monthly_pnl = current_capital * month_return
                current_capital += monthly_pnl
                
                print(f"📈 Simulated Monthly Return: {month_return*100:.2f}%")
                print(f"💰 Portfolio Value: ${current_capital:.2f}")
            else:
                print("⚠️ Could not fetch market proxy data for this month.")
        except Exception as e:
            print(f"Could not calculate PnL for {date}: {e}")

    total_return = ((current_capital - starting_capital) / starting_capital) * 100
    print(f"\n🏁 BACKTEST COMPLETE. Total Return: {total_return:.2f}%")
