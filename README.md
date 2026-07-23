# Stock Market Trend Analysis & Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![yfinance](https://img.shields.io/badge/yfinance-Real%20Time%20Data-green)

A complete end-to-end Deep Learning project to analyze and forecast stock market price trends using LSTM neural networks and real time data from Yahoo Finance.

## Problem Statement
Stock markets are highly volatile and complex. This project builds a system to analyze historical stock price patterns and forecast future trends using Long Short-Term Memory (LSTM) deep learning models.

## Features
- Analyze any stock (AAPL, GOOGL, TSLA, RELIANCE.NS etc.)
- Real time data fetching via yfinance API
- Interactive candlestick and price charts
- LSTM based price trend forecasting
- Actual vs Predicted price visualization
- Model performance metrics (MAE, RMSE)

## Tech Stack
- Python, Pandas, NumPy
- TensorFlow, Keras (LSTM Deep Learning)
- yfinance (Real time stock data API)
- Scikit-learn (Preprocessing, Metrics)
- Streamlit (Web App Deployment)
- Plotly (Interactive Charts)

## Project Pipeline
1. Real time data collection via yfinance API
2. Exploratory Data Analysis (EDA)
3. Feature Scaling using MinMaxScaler
4. Sequence creation (60 day lookback window)
5. LSTM Model building (3 layers + Dropout)
6. Model Training & Evaluation
7. Visualization of predictions
8. Deployment as Streamlit web app

## Model Architecture
| Layer | Type | Units |
|---|---|---|
| Layer 1 | LSTM + Dropout | 50 |
| Layer 2 | LSTM + Dropout | 50 |
| Layer 3 | LSTM + Dropout | 50 |
| Layer 4 | Dense | 25 |
| Output | Dense | 1 |

## Results
- MAE: $7.54
- RMSE: $8.71
- Trained on 9 years of Apple stock data (2015-2024)

## Supported Stocks
- US Stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA
- Indian Stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
- Crypto: BTC-USD, ETH-USD

## How to Run Locally
1. Clone this repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Project Structure
stock-market-forecasting/
├── app.py
├── lstm_model.keras
├── scaler.pkl
├── requirements.txt
└── README.md

