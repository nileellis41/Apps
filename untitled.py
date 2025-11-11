import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit
st.set_page_config(page_title="AccessAlpha: 2-Day Forex Forecast", layout="wide")
st.title("📈 AccessAlpha: 2-Day Forex Forecast")

# Initialize MetaTrader5
if not mt5.initialize():
    st.error("MetaTrader5 initialization failed")
    mt5.shutdown()
    st.stop()

# Function to fetch historical data
def get_forex_data(symbol, timeframe, lookback=500):
    timeframe_map = {
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1
    }
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, lookback)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# User Inputs
pair = st.selectbox("Select a Currency Pair", ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"])
timeframe = "M15"

# Retrieve data
df = get_forex_data(pair, timeframe)

if df is not None:
    # Feature Engineering
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['Volatility'] = df['close'].rolling(window=10).std()
    df['Momentum'] = df['close'] - df['close'].shift(10)
    df.dropna(inplace=True)
    
    # Prepare Training Data
    X = df[['SMA_10', 'SMA_50', 'Volatility', 'Momentum']].values
    y = df['close'].shift(-1).dropna().values
    X = X[:-1]
    
    # Initialize models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    # Train models and evaluate errors
    errors = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X[-48:])  # Predicting next 2 days (48 periods for M15 timeframe)
        actual_prices = df['close'].values[-48:]
        mse = mean_squared_error(actual_prices, y_pred)
        mae = mean_absolute_error(actual_prices, y_pred)
        errors[name] = (mse, mae)
    
    # Select the best model based on lowest MSE
    best_model_name = min(errors, key=lambda k: errors[k][0])
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X[-48:])
    future_dates = [df['time'].iloc[-1] + timedelta(minutes=15 * i) for i in range(1, 49)]
    
    # Forecast DataFrame
    forecast_df = pd.DataFrame({
        'time': future_dates,
        'Forecast': y_pred_best
    })
    
    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"MSE: {errors[best_model_name][0]:.4f}, MAE: {errors[best_model_name][1]:.4f}")
    st.dataframe(forecast_df)
    
    # Visualization
    fig = px.line(df, x='time', y='close', title=f"{pair} 2-Day Forecast with {best_model_name}", labels={'time': 'Time', 'close': 'Price'})
    fig.add_scatter(x=forecast_df['time'], y=forecast_df['Forecast'], mode='lines', name=f'Forecast ({best_model_name})', line=dict(color='red'))
    
    st.plotly_chart(fig)
else:
    st.error("Failed to retrieve forex data.")
