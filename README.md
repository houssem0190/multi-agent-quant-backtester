# Multi-Agent Trading System

A multi-agent trading workflow using LangGraph + Gemini + market data APIs.

## Features
- Fundamental analysis (Finnhub)
- Technical analysis (yfinance + pandas-ta)
- Sentiment analysis (news headlines)
- Portfolio optimization (Graphical Lasso / Markowitz-style)
- Final trader decision node

## Tech Stack
- Python
- LangGraph
- LangChain
- Google Gemini (`gemini-1.5-flash`)
- Finnhub API
- yfinance
- scikit-learn

## Setup

### 1) Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Create `.env` in project root:
```env
GOOGLE_API_KEY=your_google_key
FINNHUB_API_KEY=your_finnhub_key
```

## Run
```bash
python "multi trade.py"
```

## Notes
- `.env` is ignored by git and should never be committed.
- This project is educational and not financial advice.

