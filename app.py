
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import yfinance as yf
from tensorflow.keras.models import load_model

model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Stock Market Trend Analysis & Forecasting")
st.write("Analyze and forecast stock price trends using LSTM Deep Learning")

ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.button("Analyze & Forecast"):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

    st.subheader(f"{ticker} Stock Closing Price")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    fig1.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig1)

    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    sequence_length = 60
    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
    X = np.array(X).reshape(-1, sequence_length, 1)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    st.subheader("Actual vs Predicted Price")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index[sequence_length:], y=df['Close'][sequence_length:], name='Actual Price', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=df.index[sequence_length:], y=predictions.flatten(), name='Predicted Price', line=dict(color='red')))
    fig2.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig2)

    st.subheader("Model Performance")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(df['Close'][sequence_length:], predictions)
    rmse = np.sqrt(mean_squared_error(df['Close'][sequence_length:], predictions))
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"${mae:.2f}")
    col2.metric("RMSE", f"${rmse:.2f}")
