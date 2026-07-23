import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("Stock Market Trend Analysis & Forecasting")
st.write("Analyze and forecast stock price trends using LSTM Deep Learning")

ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.button("Analyze & Forecast"):
    with st.spinner("Downloading data and training model..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        df.columns = df.columns.get_level_values(0)

        st.subheader(f"{ticker} Stock Closing Price")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
        fig1.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig1)

        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data) * 0.80)
        train_data = scaled_data[:train_size]

        X_train, y_train = [], []
        for i in range(60, len(train_data)):
            X_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        X_train = np.array(X_train).reshape(-1, 60, 1)
        y_train = np.array(y_train)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        test_data = scaled_data[train_size-60:]
        X_test = []
        for i in range(60, len(test_data)):
            X_test.append(test_data[i-60:i, 0])
        X_test = np.array(X_test).reshape(-1, 60, 1)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(scaled_data[train_size:])

        st.subheader("Actual vs Predicted Price")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index[train_size:], y=actual.flatten(), name='Actual Price', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=df.index[train_size:], y=predictions.flatten(), name='Predicted Price', line=dict(color='red')))
        fig2.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig2)

        st.subheader("Model Performance")
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"${mae:.2f}")
        col2.metric("RMSE", f"${rmse:.2f}")
