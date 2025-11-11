# accessalpha_app.py
# Streamlit scaffold for the AccessAlpha forex analytics dashboard
# ---------------------------------------------------------------
# Notes:
# - Dark charts (black background) for user preference
# - Modular layout with cached data functions & session state
# - API placeholders for FRED, MT5, Yahoo Finance, and News
# - Safe to run as-is; all external calls stubbed
# ---------------------------------------------------------------

import os
import time
import json
import math
import random
import datetime as dt
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ---------------------------
# Global Config / Theme Setup
# ---------------------------
st.set_page_config(
    page_title="AccessAlpha — FX Command Center",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Matplotlib dark theme (no seaborn, no external styles)
plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["grid.color"] = "gray"

# --------------------------------
# Session State & App-Wide Helpers
# --------------------------------
def init_session_state():
    defaults = dict(
        connected_fred=False,
        connected_mt5=False,
        connected_news=False,
        connected_yf=False,
        live_trading=False,
        strategies_enabled=dict(carry=True, breakout=True, correlation=False),
        auto_trade=False,
        selected_pairs=["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD", "USDCHF"],
        usd_cad_window=180,  # days
        risk_leverage=2.0,
        last_refresh=str(dt.datetime.now()),
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

def badge(value: bool) -> str:
    return "🟢 Connected" if value else "🟥 Disconnected"

def section_title(title: str, help_text: str = ""):
    st.subheader(title)
    if help_text:
        st.caption(help_text)

# --------------
# Data Utilities
# --------------
@st.cache_data(show_spinner=False)
def get_dummy_price_series(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generates dummy OHLC for rapid prototyping."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=dt.date.today(), periods=days, freq="D")
    price = np.cumsum(rng.normal(0, 0.3, size=days)) + 100
    high = price + rng.normal(0.2, 0.1, size=days)
    low = price - rng.normal(0.2, 0.1, size=days)
    open_ = price + rng.normal(0, 0.15, size=days)
    close = price
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=dates
    )

@st.cache_data(show_spinner=False)
def dummy_macro() -> pd.DataFrame:
    """Placeholder macro indicators."""
    idx = pd.date_range(end=dt.date.today(), periods=24, freq="M")
    return pd.DataFrame(
        {
            "CPI YoY %": np.random.normal(3.0, 0.4, len(idx)).round(2),
            "GDP QoQ %": np.random.normal(0.6, 0.3, len(idx)).round(2),
            "PMI": np.random.normal(51.5, 1.2, len(idx)).round(1),
            "Policy Rate % (US)": np.clip(np.linspace(5.25, 4.0, len(idx)) + np.random.normal(0, 0.05, len(idx)), 0, 10).round(2),
            "Policy Rate % (CA)": np.clip(np.linspace(5.0, 3.5, len(idx)) + np.random.normal(0, 0.05, len(idx)), 0, 10).round(2),
        },
        index=idx,
    )

# -------------------------
# API Placeholders / Stubs
# -------------------------
def connect_fred(api_key: str) -> bool:
    """
    TODO: Replace with real FRED client (e.g., fredapi).
    """
    ok = bool(api_key and len(api_key) >= 10)
    st.session_state.connected_fred = ok
    return ok

def connect_mt5(host: str, login: str, password: str) -> bool:
    """
    TODO: Replace with real MetaTrader5.initialize(...) call.
    """
    ok = all([host, login, password])
    st.session_state.connected_mt5 = ok
    return ok

def connect_news(api_key: str) -> bool:
    """
    TODO: Replace with real News API client.
    """
    ok = bool(api_key and len(api_key) >= 10)
    st.session_state.connected_news = ok
    return ok

def connect_yfinance() -> bool:
    """
    Placeholder for yfinance readiness.
    """
    st.session_state.connected_yf = True
    return True

# --------------------
# Plotting Convenience
# --------------------
def line_chart_dark(series: pd.Series, title: str = "", xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(series.index, series.values)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def multi_line_chart_dark(df: pd.DataFrame, title: str = "", xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(8, 3))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False)
    st.pyplot(fig)

# ----------------------------
# Indicators (Technical Stubs)
# ----------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.clip(delta, 0, None)
    down = -np.clip(delta, None, 0)
    rs = up.rolling(window).mean() / (down.rolling(window).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# ---------------------------
# Strategy Logic (Placeholders)
# ---------------------------
def signal_carry() -> pd.DataFrame:
    data = []
    for pair in st.session_state.selected_pairs:
        score = np.random.normal(0, 1)
        data.append({"Pair": pair, "Carry Score": score})
    return pd.DataFrame(data).set_index("Pair")

def signal_breakout(price: pd.Series, lookback: int = 20) -> float:
    recent = price.tail(lookback)
    if len(recent) < lookback:
        return 0.0
    rng = (recent.max() - recent.min())
    return float(rng / (recent.mean() + 1e-9))

# ----------------------
# UI: Sidebar (Controls)
# ----------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("**Connections**")
    fred_key = st.text_input("FRED API Key", type="password")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Connect FRED"):
            ok = connect_fred(fred_key)
            st.toast(f"FRED: {badge(ok)}")
    with colB:
        if st.button("Connect yfinance"):
            ok = connect_yfinance()
            st.toast(f"yfinance: {badge(ok)}")

    mt5_host = st.text_input("MT5 Host (e.g., 127.0.0.1)")
    mt5_login = st.text_input("MT5 Login")
    mt5_pass = st.text_input("MT5 Password", type="password")
    if st.button("Connect MT5"):
        ok = connect_mt5(mt5_host, mt5_login, mt5_pass)
        st.toast(f"MT5: {badge(ok)}")

    news_key = st.text_input("News API Key", type="password")
    if st.button("Connect News"):
        ok = connect_news(news_key)
        st.toast(f"News: {badge(ok)}")

    st.markdown("---")
    st.write("**Pairs**")
    pairs = st.multiselect(
        "Tracked FX Pairs",
        options=["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD", "USDCHF", "NZDUSD", "EURJPY"],
        default=st.session_state.selected_pairs,
    )
    st.session_state.selected_pairs = pairs

    st.markdown("---")
    st.write("**Strategies**")
    st.session_state.strategies_enabled["carry"] = st.checkbox("Carry", value=st.session_state.strategies_enabled["carry"])
    st.session_state.strategies_enabled["breakout"] = st.checkbox("Breakout", value=st.session_state.strategies_enabled["breakout"])
    st.session_state.strategies_enabled["correlation"] = st.checkbox("Correlation", value=st.session_state.strategies_enabled["correlation"])

    st.markdown("---")
    st.write("**Trading**")
    st.session_state.auto_trade = st.toggle("Auto-Trade", value=st.session_state.auto_trade)
    st.session_state.risk_leverage = st.slider("Leverage", 1.0, 10.0, value=float(st.session_state.risk_leverage), step=0.5)
    st.caption(f"Last refresh: {st.session_state.last_refresh}")

# ----------------------
# Tab Layout Definitions
# ----------------------
(
    tab_overview,
    tab_macro,
    tab_technical,
    tab_sentiment,
    tab_derivs,
    tab_ml,
    tab_strategy,
    tab_scenarios,
    tab_backtest,
    tab_execution,
    tab_portfolio,
    tab_usdcad,
    tab_geo,
    tab_events,
    tab_reports,
    tab_settings,
) = st.tabs([
    "🏠 Overview",
    "🌍 Macro & Policy Dashboard",
    "📈 Technical Signals",
    "💬 Sentiment & Order Flow",
    "⚙️ Derivatives Insights",
    "🤖 Machine Learning Forecasts",
    "🛠️ Strategy Control Panel",
    "🔍 Scenario Analysis",
    "📊 Backtesting & Performance",
    "🚀 Live Trade Execution",
    "💼 Portfolio Management",
    "🇺🇸 USD/CAD Focus Zone",
    "🌐 Geopolitical Risk Map",
    "🗓️ Economic Calendar + Events",
    "📤 Reports & Exports",
    "🔧 System Settings & Logs",
])

# ----------------
# Tab: Overview
# ----------------
with tab_overview:
    st.title("🏠 Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracked Pairs", len(st.session_state.selected_pairs))
    col2.metric("Auto-Trade", "On" if st.session_state.auto_trade else "Off")
    col3.metric("Leverage", f"{st.session_state.risk_leverage}x")
    col4.metric("System Health", "OK ✅")

    st.markdown("### Market Snapshot")
    snap_cols = st.columns(3)
    for i, c in enumerate(snap_cols):
        with c:
            series = get_dummy_price_series(120, seed=40+i)["close"]
            line_chart_dark(series, title=f"{st.session_state.selected_pairs[i % len(st.session_state.selected_pairs)]} — Close")

    with st.expander("Top Trade Signals (Stub)"):
        df_carry = signal_carry()
        st.dataframe(df_carry.style.background_gradient(cmap="Greys"))

# ----------------------------
# Tab: Macro & Policy Dashboard
# ----------------------------
with tab_macro:
    st.title("🌍 Macro & Policy Dashboard")
    section_title("Key Indicators", "CPI, GDP, PMI, and policy rates (placeholder).")
    macro = dummy_macro()
    st.dataframe(macro.tail(12))

    colA, colB = st.columns(2)
    with colA:
        line_chart_dark(macro["CPI YoY %"], "Inflation (YoY)")
        line_chart_dark(macro["PMI"], "PMI")
    with colB:
        line_chart_dark(macro["GDP QoQ %"], "GDP (QoQ)")
        multi_line_chart_dark(macro[["Policy Rate % (US)", "Policy Rate % (CA)"]], "Policy Rates: US vs CA")

    with st.expander("Valuations: REER / PPP / BEER (Placeholder)"):
        st.info("TODO: compute REER/PPP/BEER using trade weights & price levels.")

# ------------------------
# Tab: Technical Signals
# ------------------------
with tab_technical:
    st.title("📈 Technical Signals")
    prices = get_dummy_price_series(240)["close"]
    st.write("Select windows")
    c1, c2, c3 = st.columns(3)
    w_fast = c1.number_input("SMA Fast", 5, 100, 20)
    w_slow = c2.number_input("SMA Slow", 10, 200, 50)
    w_rsi  = c3.number_input("RSI Window", 5, 30, 14)

    sma_fast = sma(prices, w_fast)
    sma_slow = sma(prices, w_slow)
    rsi_vals = rsi(prices, w_rsi)
    macd_line, signal_line = macd(prices)

    multi_line_chart_dark(
        pd.DataFrame({"Close": prices, f"SMA {w_fast}": sma_fast, f"SMA {w_slow}": sma_slow}),
        title="Price & Moving Averages"
    )
    multi_line_chart_dark(
        pd.DataFrame({"RSI": rsi_vals}),
        title="RSI"
    )
    multi_line_chart_dark(
        pd.DataFrame({"MACD": macd_line, "Signal": signal_line}),
        title="MACD"
    )

    with st.expander("Breakout & Pattern Recognition (Stub)"):
        score = signal_breakout(prices, lookback=20)
        st.write(f"Breakout Strength (0-1+): **{score:.2f}**")

# -----------------------------
# Tab: Sentiment & Order Flow
# -----------------------------
with tab_sentiment:
    st.title("💬 Sentiment & Order Flow")
    st.info("Placeholders for CFTC CoT, broker long/short, Twitter/Reddit news sentiment, options skew.")
    c1, c2 = st.columns(2)
    with c1:
        st.write("CFTC CoT (Stub)")
        st.dataframe(pd.DataFrame({
            "Asset": ["USD", "EUR", "JPY", "CAD"],
            "Net Specs (K)": np.random.randint(-50, 50, 4)
        }))
    with c2:
        st.write("Broker Long/Short (Stub)")
        st.dataframe(pd.DataFrame({
            "Pair": ["EURUSD", "USDJPY", "USDCAD", "GBPUSD"],
            "Long %": np.random.randint(30, 70, 4),
            "Short %": lambda df: 100 - df["Long %"],
        }).assign(**{"Short %": lambda d: 100 - d["Long %"]}))

# --------------------------
# Tab: Derivatives Insights
# --------------------------
with tab_derivs:
    st.title("⚙️ Derivatives Insights")
    st.info("FX options IV, risk reversals, smiles, forwards & swaps (placeholders).")
    ttm = np.array([7, 30, 90, 180, 365])
    iv = 0.12 + 0.02*np.sin(np.linspace(0, 2*np.pi, len(ttm)))
    df_iv = pd.DataFrame({"TTM (days)": ttm, "Implied Vol": iv})
    st.dataframe(df_iv)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ttm, iv, marker="o")
    ax.set_title("Term Structure: Implied Volatility")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# -------------------------------
# Tab: Machine Learning Forecasts
# -------------------------------
with tab_ml:
    st.title("🤖 Machine Learning Forecasts")
    st.info("Stubs for Random Forest / XGBoost / LSTM. Add feature importances & prediction intervals.")
    horizon = st.slider("Forecast Horizon (days)", 5, 60, 14)
    series = get_dummy_price_series(300)["close"]
    # Dummy forecast: last value + noise
    forecast_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.Series(series.iloc[-1] + np.cumsum(np.random.normal(0, 0.2, horizon)), index=forecast_idx)

    multi_line_chart_dark(pd.DataFrame({"History": series.tail(120)}), "Recent History")
    line_chart_dark(forecast, "Dummy Forecast")

    with st.expander("Feature Importances (Placeholder)"):
        feats = pd.DataFrame({"Feature": ["carry", "momentum", "term_spread", "oil", "risk_aversion"],
                              "Importance": np.random.dirichlet(np.ones(5))})
        st.bar_chart(feats.set_index("Feature"))

# ---------------------------------
# Tab: Strategy Control Panel
# ---------------------------------
with tab_strategy:
    st.title("🛠️ Strategy Control Panel")
    st.write("Blend macro + technical + sentiment signals.")
    w1, w2, w3 = st.slider("Weights (Macro, Technical, Sentiment)", 0.0, 1.0, (0.33, 0.33, 0.34))
    st.write(f"Selected: Macro={w1:.2f}, Technical={w2:.2f}, Sentiment={w3:.2f}")

    st.write("Active Strategies:", ", ".join([k for k, v in st.session_state.strategies_enabled.items() if v]))

    with st.expander("Signal Preview (Stub)"):
        preview = pd.DataFrame({
            "Pair": st.session_state.selected_pairs,
            "Composite Score": np.random.normal(0, 1, len(st.session_state.selected_pairs)).round(2),
            "Direction": np.random.choice(["Long", "Short", "Neutral"], len(st.session_state.selected_pairs))
        }).set_index("Pair")
        st.dataframe(preview)

# ---------------------------
# Tab: Scenario Analysis
# ---------------------------
with tab_scenarios:
    st.title("🔍 Scenario Analysis")
    st.info("Run what-if simulations: policy shifts, forward curve shocks, geopolitics.")
    col1, col2 = st.columns(2)
    with col1:
        d_policy = st.slider("Policy Rate Shock (bps)", -200, 200, 25, step=25)
        oil = st.slider("Oil Shock ($/bbl)", -20, 20, 5)
    with col2:
        risk = st.slider("Risk Aversion Shock (σ)", -2.0, 2.0, 0.5, step=0.1)
        trade = st.slider("Trade Shock (% tariffs)", -5, 10, 2)

    st.write("**USD Impact (Dummy):**")
    imp = 0.002 * d_policy + 0.001 * oil + 0.05 * risk - 0.003 * trade
    st.metric("Estimated DXY Move", f"{imp:.2%}")

# -------------------------------------
# Tab: Backtesting & Performance
# -------------------------------------
with tab_backtest:
    st.title("📊 Backtesting & Performance")
    st.info("Walk-forward, Sharpe, drawdowns, attribution (placeholders).")
    lookback = st.slider("Backtest Window (days)", 90, 720, 360, step=30)
    pnl = np.cumsum(np.random.normal(0.0, 0.5, lookback))
    dd = pnl - np.maximum.accumulate(pnl)
    df_bt = pd.DataFrame({"PnL": pnl, "Drawdown": dd}, index=pd.date_range(end=dt.date.today(), periods=lookback, freq="D"))
    multi_line_chart_dark(df_bt[["PnL"]], "Strategy PnL")
    multi_line_chart_dark(df_bt[["Drawdown"]], "Drawdown")

    with st.expander("Attribution (Stub)"):
        attrib = pd.DataFrame({
            "Module": ["Macro", "Technical", "Sentiment"],
            "Return (bps)": np.random.randint(-50, 120, 3)
        }).set_index("Module")
        st.dataframe(attrib)

# -----------------------------
# Tab: Live Trade Execution
# -----------------------------
with tab_execution:
    st.title("🚀 Live Trade Execution")
    st.info("MT5 order panel (placeholder).")
    c1, c2, c3 = st.columns(3)
    with c1:
        pair = st.selectbox("Pair", st.session_state.selected_pairs)
    with c2:
        side = st.selectbox("Side", ["Buy", "Sell"])
    with c3:
        qty = st.number_input("Quantity (lots)", min_value=0.01, value=0.10, step=0.01)

    if st.button("Submit Order (Stub)"):
        st.success(f"Order submitted: {side} {qty} {pair} (demo)")

    with st.expander("Open Positions (Stub)"):
        st.dataframe(pd.DataFrame({
            "Pair": ["EURUSD", "USDJPY"],
            "Side": ["Buy", "Sell"],
            "Qty": [0.2, 0.15],
            "Entry": [1.0835, 160.25],
            "PnL ($)": [125.4, -42.8]
        }))

# -----------------------------
# Tab: Portfolio Management
# -----------------------------
with tab_portfolio:
    st.title("💼 Portfolio Management")
    st.info("Exposure, correlation, sizing, leverage (placeholders).")
    expo = pd.DataFrame({
        "Pair": st.session_state.selected_pairs,
        "Exposure ($k)": np.random.randint(-200, 200, len(st.session_state.selected_pairs))
    }).set_index("Pair")
    st.dataframe(expo)

    st.write("Correlation Matrix (Dummy)")
    n = len(st.session_state.selected_pairs)
    corr = pd.DataFrame(np.corrcoef(np.random.normal(size=(n, 200))), columns=st.session_state.selected_pairs, index=st.session_state.selected_pairs)
    st.dataframe(corr.style.background_gradient(cmap="Greys"))

# -----------------------------
# Tab: USD/CAD Focus Zone
# -----------------------------
with tab_usdcad:
    st.title("🇺🇸 USD/CAD Focus Zone")
    window = st.slider("Window (days)", 60, 365, int(st.session_state.usd_cad_window), step=15)
    usdcad = get_dummy_price_series(window, seed=123)["close"]
    line_chart_dark(usdcad, "USD/CAD Price")

    with st.expander("Oil Correlation (Stub)"):
        oil_prices = pd.Series(np.cumsum(np.random.normal(0, 0.4, window)) + 70, index=usdcad.index)
        df = pd.DataFrame({"USDCAD": usdcad, "WTI": oil_prices})
        multi_line_chart_dark(df, "USDCAD vs WTI (proxy)")
        st.write("Rolling Corr (30d):", df["USDCAD"].rolling(30).corr(df["WTI"]).dropna().tail(1).iloc[0].round(2))

# -----------------------------
# Tab: Geopolitical Risk Map
# -----------------------------
with tab_geo:
    st.title("🌐 Geopolitical Risk Map")
    st.info("Interactive map placeholder with risk overlays.")
    st.map(pd.DataFrame({
        "lat": [50.45, 25.03, 31.95, 35.69],
        "lon": [30.52, 121.56, 35.93, 139.69],
        "label": ["Ukraine", "Shanghai/Taiwan Strait", "Middle East", "Japan"]
    }))

# -----------------------------------
# Tab: Economic Calendar + Events
# -----------------------------------
with tab_events:
    st.title("🗓️ Economic Calendar + Events")
    st.info("Placeholder: integrate Yahoo Finance / Investing.com / Econ APIs.")
    upcoming = pd.DataFrame({
        "Time (ET)": ["08:30", "10:00", "14:00"],
        "Event": ["CPI (MoM, YoY)", "ISM Services PMI", "FOMC Decision"],
        "Priority": ["High", "Medium", "High"],
        "Previous": ["0.2% / 3.0%", "52.6", "5.25%"],
        "Consensus": ["0.2% / 3.1%", "52.1", "Hold"],
    })
    st.dataframe(upcoming)

# -----------------------------
# Tab: Reports & Exports
# -----------------------------
with tab_reports:
    st.title("📤 Reports & Exports")
    st.info("Download reports, logs, and snapshots (placeholders).")
    report = pd.DataFrame({
        "Pair": st.session_state.selected_pairs,
        "Signal": np.random.choice(["Long", "Short", "Neutral"], len(st.session_state.selected_pairs)),
        "Score": np.random.normal(0, 1, len(st.session_state.selected_pairs)).round(2)
    })
    st.dataframe(report)
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download Signals CSV", csv, file_name="signals.csv", mime="text/csv")

# -----------------------------
# Tab: System Settings & Logs
# -----------------------------
with tab_settings:
    st.title("🔧 System Settings & Logs")
    st.write("**Connections**")
    st.write(f"FRED: {badge(st.session_state.connected_fred)}")
    st.write(f"yfinance: {badge(st.session_state.connected_yf)}")
    st.write(f"MT5: {badge(st.session_state.connected_mt5)}")
    st.write(f"News: {badge(st.session_state.connected_news)}")

    with st.expander("Environment / Keys (Do not commit)"):
        st.code("FRED_API_KEY=...\nNEWS_API_KEY=...\nMT5_HOST=...\nMT5_LOGIN=...\nMT5_PASSWORD=...")

    with st.expander("Logs (Placeholder)"):
        st.text_area("Runtime Log", value="[INFO] AccessAlpha started...\n[INFO] UI initialized.\n", height=200)

# -------------
# Footer / Tick
# -------------
st.session_state.last_refresh = str(dt.datetime.now())
st.caption(f"AccessAlpha © {dt.date.today().year} — Prototype build. Last refresh: {st.session_state.last_refresh}")
