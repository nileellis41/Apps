"""
MarketBridge — Robinhood-Style Trading App (Streamlit MVP)
Author: CFA-educated Quantitative Software Engineer (assistant)

⚠️ Notes
- This MVP supports: market data, watchlists, charts, paper trading (via Alpaca), simulated trading fallback,
  portfolio analytics, and news.
- Loads Alpaca keys from an environment file using python-dotenv (as requested).
- If keys are missing or invalid, it automatically falls back to a built-in simulator.
- Single-file app; you can later split into multipage modules.
- Dark charts to match your preference.

Quick start
-----------
1) pip install streamlit yfinance plotly numpy pandas requests python-dateutil alpaca-trade-api python-dotenv
2) Create an env file (path is configurable below, default: "TBD/keys.txt") with:
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
3) streamlit run app.py

"""

from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import requests

# NEW: dotenv + Alpaca official SDK
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient

# ------------------------------
# Config & Theming
# ------------------------------
st.set_page_config(
    page_title="MarketBridge — Robinhood-Style MVP",
    page_icon="📈",
    layout="wide",
)

PRIMARY_BG = "#0E1117"  # Streamlit dark bg
CARD_BG = "#111827"
ACCENT = "#22c55e"

# Simple CSS polish
st.markdown(
    f"""
    <style>
    .metric-card {{ background:{CARD_BG}; padding: 1rem; border-radius:16px; }}
    .accent {{ color:{ACCENT}; }}
    .small {{ opacity:0.8; font-size:0.85rem; }}
    .badge {{ background:#1f2937; padding:2px 8px; border-radius:999px; margin-left:6px; }}
    .danger {{ color:#ef4444; }}
    .ok {{ color:#22c55e; }}
    .warn {{ color:#f59e0b; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers & Data
# ------------------------------
@st.cache_data(show_spinner=False)
def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and not data.empty:
        data.index = pd.to_datetime(data.index)
    return data

@st.cache_data(show_spinner=False)
def get_quote(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    info = t.fast_info if hasattr(t, "fast_info") else {}
    price = None
    try:
        price = float(info.get("last_price")) if info else None
    except Exception:
        price = None
    return {
        "symbol": ticker.upper(),
        "price": price,
        "currency": info.get("currency", "USD") if info else "USD",
        "market_cap": info.get("market_cap", None) if info else None,
    }

@st.cache_data(show_spinner=False)
def get_news(ticker: str) -> List[Dict]:
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []

# ------------------------------
# Simulated broker in-memory (fallback)
# ------------------------------
@dataclass
class SimOrder:
    id: str
    ts: float
    symbol: str
    qty: int
    side: str
    price: float

class SimBroker:
    def __init__(self, starting_cash: float = 10000.0):
        self.cash = starting_cash
        self.positions: Dict[str, int] = {}
        self.trades: List[SimOrder] = []

    def _mark(self, symbol: str) -> float:
        q = get_quote(symbol)
        return float(q["price"]) if q["price"] else 0.0

    def place(self, symbol: str, qty: int, side: str) -> Dict:
        px = self._mark(symbol)
        if px <= 0: return {"error": "No live price available"}
        notional = qty * px
        if side == "buy":
            if self.cash < notional:
                return {"error": "Insufficient cash"}
            self.cash -= notional
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
        else:
            if self.positions.get(symbol, 0) < qty:
                return {"error": "Insufficient shares"}
            self.positions[symbol] -= qty
            self.cash += notional
        oid = f"sim-{len(self.trades)+1}"
        self.trades.append(SimOrder(oid, time.time(), symbol, qty, side, px))
        return {"id": oid, "filled_avg_price": px}

    def snapshot(self) -> Dict:
        equity = self.cash
        pos_list = []
        for s, q in self.positions.items():
            if q == 0: continue
            px = self._mark(s)
            equity += q * px
            pos_list.append({"symbol": s, "qty": q, "market_price": px, "market_value": q*px})
        return {"cash": self.cash, "equity": equity, "positions": pos_list, "trades": [asdict(t) for t in self.trades]}

# ------------------------------
# Auth & Clients — dotenv first, then fallback
# ------------------------------
st.sidebar.title("⚙️ Settings")
ENV_PATH = st.sidebar.text_input("Env file path", value="TBD/keys.txt")

# Load API keys from environment file
try:
    load_dotenv(ENV_PATH)
except Exception:
    pass
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if "sim" not in st.session_state:
    st.session_state.sim = SimBroker(25_000.0)

trading_client: Optional[TradingClient] = None
stock_data_client: Optional[StockHistoricalDataClient] = None
option_data_client: Optional[OptionHistoricalDataClient] = None
use_alpaca = False

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    st.sidebar.error("❌ API keys not loaded. Using simulator.")
else:
    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        # Will raise if wrong creds upon first call
        acct = trading_client.get_account()
        stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        option_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        use_alpaca = True
        st.sidebar.success(f"Alpaca connected (paper). Buying power: ${float(acct.buying_power):,.2f}")
    except Exception as e:
        st.sidebar.error(f"Alpaca connection failed — using simulator. {e}")
        use_alpaca = False

# Watchlist persistence in session (you can later persist to a CSV/DB)
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

# ------------------------------
# Header
# ------------------------------
st.markdown("# 🟢 MarketBridge — Robinhood-Style MVP")
st.caption("Live quotes · Dark charts · Watchlists · Paper trading · Portfolio analytics · News")

# ------------------------------
# Tabs
# ------------------------------
markets_tab, watchlist_tab, trade_tab, portfolio_tab, news_tab, settings_tab = st.tabs([
    "Markets", "Watchlist", "Trade", "Portfolio", "News", "Settings"
])

# ------------------------------
# Markets
# ------------------------------
with markets_tab:
    cols = st.columns([1.5, 1])
    with cols[0]:
        symbol = st.text_input("Symbol", value="AAPL").upper().strip()
        period = st.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], index=4)
        interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"], index=8)
        data = get_price_history(symbol, period=period, interval=interval)
        if data.empty:
            st.warning("No price data.")
        else:
            fig = go.Figure()
            fig.add_candlestick(
                x=data.index,
                open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']
            )
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Indicators
            with st.expander("Add Indicators", expanded=False):
                ma_choices = st.multiselect("Moving Averages", ["MA20","MA50","MA200"], default=["MA20","MA50"])
                if ma_choices:
                    for m in ma_choices:
                        w = int(m.replace("MA",""))
                        data[m] = data['Close'].rolling(w).mean()
                        fig.add_trace(go.Scatter(x=data.index, y=data[m], name=m))
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        q = get_quote(symbol)
        price = q.get("price")
        delta = None
        try:
            if len(data) >= 2:
                prev = float(data['Close'].iloc[-2])
                delta = (float(price) - prev) / prev * 100 if price else None
        except Exception:
            pass

        st.markdown(f"<div class='metric-card'><h3>{symbol}</h3>", unsafe_allow_html=True)
        st.metric("Last Price", f"{price:.2f}" if price else "—", f"{delta:.2f}%" if delta is not None else None)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("➕ Add to Watchlist"):
            if symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(symbol)
                st.success(f"Added {symbol}")
            else:
                st.info("Already in watchlist")

# ------------------------------
# Watchlist
# ------------------------------
with watchlist_tab:
    st.subheader("Your Watchlist")
    wl = st.session_state.watchlist
    if not wl:
        st.info("Empty. Add symbols from Markets tab.")
    else:
        rows = []
        for s in wl:
            qt = get_quote(s)
            rows.append({"Symbol": s, "Price": qt.get("price"), "Currency": qt.get("currency")})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=280)

    edit_col, del_col = st.columns(2)
    with edit_col:
        new_symbol = st.text_input("Add symbol", key="add_wl").upper().strip()
        if st.button("Add", key="add_btn") and new_symbol:
            if new_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol)
            else:
                st.warning("Already exists.")
    with del_col:
        rm_symbol = st.text_input("Remove symbol", key="rm_wl").upper().strip()
        if st.button("Remove", key="rm_btn") and rm_symbol:
            if rm_symbol in st.session_state.watchlist:
                st.session_state.watchlist.remove(rm_symbol)
            else:
                st.warning("Not in watchlist.")

# ------------------------------
# Trade
# ------------------------------
with trade_tab:
    st.subheader("Place Order")
    tcol1, tcol2, tcol3 = st.columns([1,1,1])
    with tcol1:
        t_symbol = st.text_input("Symbol", value="AAPL", key="trade_symbol").upper().strip()
        side = st.radio("Side", ["buy", "sell"], horizontal=True)
    with tcol2:
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        order_type = st.selectbox("Order Type", ["market"], index=0)  # extend later
    with tcol3:
        tif = st.selectbox("Time in Force", ["day", "gtc"], index=0)
        price_preview = get_quote(t_symbol).get("price")
        st.caption(f"Est. price: {price_preview if price_preview else '—'}")

    if st.button("Submit Order", type="primary"):
        if use_alpaca and trading_client:
            try:
                req = MarketOrderRequest(
                    symbol=t_symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY if tif == "day" else TimeInForce.GTC,
                )
                order = trading_client.submit_order(req)
                avg = getattr(order, "filled_avg_price", None) or "—"
                st.success(f"Order submitted (ID: {order.id}). Avg price: {avg}")
            except Exception as e:
                st.error(f"Alpaca order failed: {e}")
        else:
            resp = st.session_state.sim.place(t_symbol, qty, side)
            if "error" in resp:
                st.error(resp["error"])
            else:
                st.success(f"Sim order OK. Avg price: {resp.get('filled_avg_price', '—')}")

# ------------------------------
# Portfolio
# ------------------------------
with portfolio_tab:
    st.subheader("Portfolio Snapshot")
    if use_alpaca and trading_client:
        try:
            acct = trading_client.get_account()
            cash = float(acct.cash)
            equity = float(acct.equity)
            st.markdown(
                f"<div class='metric-card'><h4>Paper Account</h4>", unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Cash", f"${cash:,.2f}")
            c2.metric("Equity", f"${equity:,.2f}")
            c3.metric("Buying Power", f"${float(acct.buying_power):,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

            positions = trading_client.get_all_positions()
            if positions:
                p_rows = []
                for p in positions:
                    # Alpaca returns strings for numeric fields; cast safely
                    def f(x):
                        try:
                            return float(x)
                        except Exception:
                            return None
                    p_rows.append({
                        "Symbol": p.symbol,
                        "Qty": int(f(p.qty) or 0),
                        "Avg Cost": f(p.avg_entry_price),
                        "Mkt Price": f(p.market_price),
                        "Mkt Value": f(p.market_value),
                        "Unrealized P/L": f(p.unrealized_pl),
                        "Unrealized P/L %": (f(p.unrealized_plpc) or 0)*100 if p.unrealized_plpc is not None else None,
                    })
                st.dataframe(pd.DataFrame(p_rows), use_container_width=True, height=320)
            else:
                st.info("No open positions.")
        except Exception as e:
            st.error(f"Failed to load Alpaca portfolio: {e}")
    else:
        snap = st.session_state.sim.snapshot()
        c1, c2 = st.columns(2)
        c1.metric("Cash", f"${snap['cash']:,.2f}")
        c2.metric("Equity", f"${snap['equity']:,.2f}")
        if snap["positions"]:
            st.dataframe(pd.DataFrame(snap["positions"]), use_container_width=True, height=320)
        else:
            st.info("No positions in simulator.")

    st.divider()
    st.subheader("Risk & Performance (close-to-close)")
    bench = st.text_input("Benchmark (optional)", value="SPY")
    symbols_for_perf = st.multiselect("Symbols to analyze", st.session_state.watchlist, default=st.session_state.watchlist[:5])
    lookback = st.selectbox("Lookback", ["3mo","6mo","1y","2y"], index=1)

    perf_rows = []
    for s in symbols_for_perf:
        hist = get_price_history(s, period=lookback, interval="1d")
        if hist.empty or len(hist) < 30:
            continue
        close = hist["Close"].dropna()
        rets = close.pct_change().dropna()
        cagr = (close.iloc[-1]/close.iloc[0])**(252/len(close)) - 1
        vol = rets.std()*np.sqrt(252)
        sharpe = (rets.mean()*252)/(vol+1e-9)
        perf_rows.append({"Symbol": s, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe})
    if perf_rows:
        pdf = pd.DataFrame(perf_rows).set_index("Symbol")
        st.dataframe((pdf*100).round(2), use_container_width=True)
    else:
        st.info("Add more symbols with sufficient history.")

# ------------------------------
# News
# ------------------------------
with news_tab:
    st.subheader("News Feed")
    n_symbol = st.text_input("Symbol", value="AAPL", key="news_symbol").upper().strip()
    items = get_news(n_symbol)
    if not items:
        st.info("No news items.")
    else:
        for it in items[:12]:
            title = it.get("title")
            link = it.get("link")
            publisher = it.get("publisher")
            ts = it.get("providerPublishTime")
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            st.markdown(f"**{title}**  ")
            st.caption(f"{publisher} · {dt}")
            if link:
                st.markdown(f"[Open article]({link})")
            st.divider()

# ------------------------------
# Settings / Export
# ------------------------------
with settings_tab:
    st.subheader("API & Export")
    st.caption("Reload env to pick up changes if you edit keys.txt")
    if st.button("Reload .env"):
        load_dotenv(ENV_PATH, override=True)
        st.experimental_rerun()

    if st.button("Export Watchlist to CSV"):
        df = pd.DataFrame({"symbol": st.session_state.watchlist})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download watchlist.csv", csv, file_name="watchlist.csv", mime="text/csv")

    st.markdown("### Disclaimers")
    st.markdown(
        """
        - For education only. Not investment advice. MarketBridge is not a broker-dealer.
        - Live trading requires proper licensing, KYC/AML, disclosures, and a compliant broker.
        - Paper trading via Alpaca's paper API is supported here; review Alpaca's terms.
        """
    )

    st.markdown("### Roadmap")
    st.markdown(
        """
        - Options chain & Greeks (via compatible data source)
        - Level II quotes / order book (if provider available)
        - Real-time websockets & alerts
        - Fractional shares & recurring buys (if broker supports)
        - Multi-asset (crypto via exchange API, futures/FX via MT5)
        - User auth + database persistence (Supabase/Firebase/Postgres)
        - Backtests and model-driven signals (RSI/MA, ML forecasts)
        """
    )
