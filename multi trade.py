import os
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
from typing import TypedDict
from dotenv import load_dotenv
from sklearn.covariance import GraphicalLasso
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# 1. Load Environment Variables
load_dotenv()

# 2. Define the State (The Shared Clipboard)
class InitialState(TypedDict):
    tickers: list[str]
    fundamental_analysis: str
    technical_analysis: str
    sentiment_analysis: str
    optimal_weights: str
    market_direction: str
    final_decision: str

# ---------------------------------------------------------
# 3. DEFINE TOOLS (The "Hands")
# ---------------------------------------------------------
@tool
def get_fundamentals(ticker: str) -> str:
    """Gets company fundamentals from Finnhub."""
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_api_key:
        return "FINNHUB_API_KEY is missing. Add it to .env to enable live fundamental analysis."

    symbol = ticker.strip().upper()
    base_url = "https://finnhub.io/api/v1"
    params = {"symbol": symbol, "token": finnhub_api_key}

    try:
        profile_resp = requests.get(f"{base_url}/stock/profile2", params=params, timeout=15)
        metrics_resp = requests.get(f"{base_url}/stock/metric", params={**params, "metric": "all"}, timeout=15)
    except requests.RequestException as exc:
        return f"Finnhub request failed for {symbol}: {exc}"

    if profile_resp.status_code != 200 or metrics_resp.status_code != 200:
        return f"Finnhub returned an error for {symbol}."

    profile = profile_resp.json() or {}
    metrics_data = metrics_resp.json() or {}
    metric = metrics_data.get("metric", {}) if isinstance(metrics_data, dict) else {}

    company_name = profile.get("name") or symbol
    industry = profile.get("finnhubIndustry") or "N/A"
    market_cap = profile.get("marketCapitalization")
    pe_ttm = metric.get("peTTM")
    
    return (
        f"{company_name} ({symbol}) fundamentals | "
        f"Industry: {industry}, Market Cap: {market_cap}M, P/E TTM: {pe_ttm}."
    )

@tool
def get_technicals(ticker: str) -> str:
    """Calculates RSI, MACD, and 50/200-day Moving Averages using yfinance."""
    df = yf.Ticker(ticker).history(period="6mo")
    if df.empty:
        return f"No data found for {ticker}."

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50, append=True)
    
    latest = df.iloc[-1]
    rsi = latest.get('RSI_14', 50)
    macd = latest.get('MACD_12_26_9', 0)
    signal = latest.get('MACDs_12_26_9', 0)
    
    status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    trend = "Bullish Cross" if macd > signal else "Bearish Cross"
    
    return f"{ticker} Technicals: RSI is {rsi:.2f} ({status}). MACD is {macd:.2f} with a {trend}."

@tool
def get_sentiment(ticker: str) -> str:
    """Fetches the most recent news headlines for a given stock ticker."""
    try:
        news_items = yf.Ticker(ticker).news
        if not news_items:
            return f"No recent news found for {ticker}."
        
        report = f"Latest Headlines for {ticker}:\n"
        for item in news_items[:5]:
            title = item.get('title', 'No Title')
            report += f"- {title}\n"
            
        return report
    except Exception as e:
        return f"Failed to fetch news for {ticker}: {str(e)}"

@tool
def calculate_markowitz(tickers: list[str]) -> str:
    """Calculates optimal Global Minimum Variance weights using Graphical Lasso."""
    try:
        if len(tickers) < 2:
            return "Error: Markowitz optimization requires at least 2 tickers."

        data = yf.download(tickers, period="1y")['Close']
        returns = data.pct_change().dropna()
        returns_centered = returns - returns.mean()

        glasso = GraphicalLasso(alpha=0.6, max_iter=100)
        glasso.fit(returns_centered)
        precision_matrix = glasso.precision_

        ones = np.ones(len(tickers))
        raw_weights = precision_matrix.dot(ones) / ones.dot(precision_matrix).dot(ones)

        weight_dict = dict(zip(tickers, raw_weights))
        report = "GLASSO-Optimized Portfolio Weights (alpha=0.6):\n"
        for ticker, weight in weight_dict.items():
            report += f"- {ticker}: {weight * 100:.2f}%\n"
            
        return report
    except Exception as e:
        return f"GLASSO Optimization failed: {str(e)}"

# ---------------------------------------------------------
# 4. INITIALIZE LLM AND AGENTS (The "Brains")
# ---------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Put it in a local .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=api_key)

fund_agent = create_react_agent(llm, tools=[get_fundamentals], prompt="You are a fundamental analyst. Summarize the fundamentals.")
tech_agent = create_react_agent(llm, tools=[get_technicals], prompt="You are a technical analyst. Summarize the chart indicators.")
sent_agent = create_react_agent(llm, tools=[get_sentiment], prompt="You are a sentiment analyst. Summarize the news.")
quant_agent = create_react_agent(llm, tools=[calculate_markowitz], prompt="You are a quantitative analyst. Output the weights.")

# ---------------------------------------------------------
# 5. WRAPPER NODES (Translating State to Agents)
# ---------------------------------------------------------
def fundamentals(state: InitialState):
    response = fund_agent.invoke({"messages": [("user", f"Analyze fundamentals for {state['tickers']}")]})
    return {"fundamental_analysis": response['messages'][-1].content} # Fixed Key!

def technicals(state: InitialState):
    response = tech_agent.invoke({"messages": [("user", f"Analyze technicals for {state['tickers']}")]})
    return {"technical_analysis": response['messages'][-1].content} # Fixed Key!

def sentiment(state: InitialState): 
    response = sent_agent.invoke({"messages": [("user", f"Analyze sentiment for {state['tickers']}")]})
    return {"sentiment_analysis": response['messages'][-1].content} # Fixed Key!

def risk(state: InitialState):
    message = f"""You are a risk analyst. Review the reports of {state['tickers']}:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    If the overall consensus is clearly negative/bearish, output EXACTLY the word "BEARISH".
    Otherwise, if it is neutral or positive, output EXACTLY the word "BULLISH"."""
    
    response = llm.invoke(message)
    direction = response.content.strip().upper()
    if "BEARISH" not in direction:
        direction = "BULLISH"
    return {"market_direction": direction}  

def route_market_direction(state: InitialState) -> str:
    """Checks the market direction and routes the graph."""
    if state.get("market_direction") == "BEARISH":
        return "short"
    else:
        return "long" 

def markowitz(state: InitialState):
    response = quant_agent.invoke({"messages": [("user", f"Calculate optimal weights for {state['tickers']}")]})
    return {"optimal_weights": response["messages"][-1].content}

def trader_node(state: InitialState):
    prompt = f"""You are a head trader. Review these reports and make a final LONG decision:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    Quant: {state.get('optimal_weights')}"""
    
    response = llm.invoke(prompt)
    return {"final_decision": response.content}

def short_trader_node(state: InitialState):
    # Added this node so the graph has somewhere to go if Risk outputs BEARISH
    prompt = f"""You are a head trader. Review these reports and make a final SHORT decision:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}"""
    
    response = llm.invoke(prompt)
    return {"final_decision": f"[SHORT STRATEGY TRIGGERED]\n{response.content}"}

# ---------------------------------------------------------
# 6. BUILD AND COMPILE THE GRAPH
# ---------------------------------------------------------
graph = StateGraph(InitialState)

# Add Nodes
graph.add_node("fundamental_analysis", fundamentals)
graph.add_node("technical_analysis", technicals)
graph.add_node("sentiment_analysis", sentiment)
graph.add_node("risk_analysis", risk)
graph.add_node("markowitz", markowitz)
graph.add_node("trader", trader_node)
graph.add_node("short_trader", short_trader_node)

# Add Edges (Fan-Out)
graph.add_edge(START, "fundamental_analysis")
graph.add_edge(START, "technical_analysis")    
graph.add_edge(START, "sentiment_analysis")

# Add Edges (Fan-In to Risk)
graph.add_edge(['fundamental_analysis', 'technical_analysis', 'sentiment_analysis'], 'risk_analysis')

# Add Conditional Routing
graph.add_conditional_edges(
    'risk_analysis', 
    route_market_direction, 
    {'short': 'short_trader', 'long': 'markowitz'}
)

# Execution Paths
graph.add_edge('markowitz', 'trader')
graph.add_edge("trader", END)
graph.add_edge("short_trader", END)

# Compile
app = graph.compile()

# ---------------------------------------------------------
# 7. RUN / TEST THE CODE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting the Multi-Agent Hedge Fund...")
    
    # Put the tickers you want to analyze here!
    inputs = {"tickers": ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META"]}    
    # Stream the graph so we can watch the agents work
    for event in app.stream(inputs, stream_mode="updates"):
        for node_name, node_state in event.items():
            print(f"\n✅ --- {node_name.upper()} FINISHED ---")
            
    # Once finished, grab the final state to print the ultimate decision
    final_state = app.invoke(inputs)
    
    print("\n" + "="*60)
    print("🏆 FINAL PORTFOLIO MANAGER DECISION 🏆")
    print("="*60)
    print(f"Detected Regime: {final_state.get('market_direction')}")
    print("-"*60)
    print(final_state.get("final_decision", "No decision recorded."))