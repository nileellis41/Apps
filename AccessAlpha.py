import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Streamlit Page Config
st.set_page_config(page_title="AccessAlpha", layout="wide")

# Sidebar
st.sidebar.title("AccessAlpha Dashboard")
st.sidebar.markdown("Navigate through:")
navigation = st.sidebar.radio("Sections", ["Home", "Markets", "Analytics", "Portfolio", "Learning Resources"])

# Header
st.title("AccessAlpha")
st.markdown("### Simplifying access to financial markets for everyone.")

# Home Section
if navigation == "Home":
    st.subheader("Welcome to AccessAlpha")
    st.markdown("""
    **Mission:** Provide insights and tools for better market understanding.
    **Explore** global market stats, analyze trends, and learn trading basics.
    """)

    # Quick Stats Example
    st.write("### Quick Market Stats")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("S&P 500", "4,200", "-0.5%")
    with col2:
        st.metric("USD/CAD", "1.35", "+0.2%")
    with col3:
        st.metric("Gold", "$1,950", "+1.1%")

# Markets Section
elif navigation == "Markets":
    st.subheader("Market Data Viewer")
    ticker = st.text_input("Enter a Stock Ticker (e.g., AAPL):", value="AAPL")
    
    if ticker:
        try:
            stock_data = yf.download(ticker, period="1mo", interval="1d")
            st.line_chart(stock_data["Close"], width=700, height=400)
        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Analytics Section
elif navigation == "Analytics":
    st.subheader("Analytics Tools")
    st.markdown("### RSI and Moving Average Analysis")
    ticker = st.text_input("Enter a Stock Ticker for Analysis (e.g., MSFT):", value="MSFT")

    if ticker:
        try:
            stock_data = yf.download(ticker, period="6mo", interval="1d")
            stock_data["RSI"] = 100 - (100 / (1 + stock_data["Close"].pct_change().apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                                              stock_data["Close"].pct_change().apply(lambda x: abs(x)).rolling(window=14).mean()))
            stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()
            stock_data["SMA_200"] = stock_data["Close"].rolling(window=200).mean()

            st.line_chart(stock_data[["Close", "SMA_50", "SMA_200"]])
            st.markdown("### RSI Over Time")
            st.line_chart(stock_data["RSI"])
        except Exception as e:
            st.error(f"Error analyzing data: {e}")

# Portfolio Section
elif navigation == "Portfolio":
    st.subheader("Simulated Portfolio Tracker")

    # Example: Adding/Removing Assets
    st.markdown("### Add Assets to Your Portfolio")
    portfolio = st.session_state.get("portfolio", [])
    asset = st.text_input("Add an Asset (e.g., AAPL):")
    if st.button("Add Asset"):
        portfolio.append(asset)
        st.session_state["portfolio"] = portfolio

    st.write("### Current Portfolio")
    if portfolio:
        st.write(portfolio)
        # Simulated Data
        performance = pd.DataFrame({
            "Asset": portfolio,
            "ROI (%)": np.random.uniform(-10, 20, size=len(portfolio))
        })
        st.dataframe(performance)
    else:
        st.write("No assets added yet.")

# Learning Resources Section
elif navigation == "Learning Resources":
    st.subheader("Learning Resources")
    st.markdown("### Trading Tutorials")
    st.markdown("""
    - [Understanding RSI and Moving Averages](https://www.investopedia.com/terms/r/rsi.asp)
    - [Basics of Stock Trading](https://www.investopedia.com/stock-basics-4689798)
    - [Getting Started with Forex](https://www.babypips.com/learn)
    """)

    st.markdown("### Video Tutorials")
    st.video("https://www.youtube.com/watch?v=qlQOaMPe7AM")  # Example video link
    st.video("https://www.youtube.com/watch?v=bCN0jg4OlUQ")  # Another example video link

# Footer
st.write("---")
st.write("**AccessAlpha v0.2 - Built with Streamlit**")
