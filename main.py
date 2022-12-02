
from time import time
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout="wide")
st.title("Stock Prediction Platform")
user_input = st.text_input("Enter The Stock Ticker", "AAPL")
stocks = (user_input, "AAPL", "GOOG", "BC94.L",  "MSFT", "GME", "TSLA", "BTC-USD", "ETH-USD", "DOGE-USD", "SHIB-USD", "TWTR", "META", "RELIANCE.NS", "TATASTEEL.NS", "TATAMOTORS.NS", "TATAPOWER.NS"
, "PEP", "COKE", "IOC.NS")
selected_stocks = st.selectbox("Select Stock for Prediction", stocks)

n_years = st.slider("Years of Prediction:", 0, 10)
period = n_years * 365

#n_months = st.slider("Months of Prediction:", 1, 12)
#period1 = n_months * 30

#n_weeks = st.slider("Weeks of Prediction:", 1, 5)
#period = n_weeks * 7

#n_days = st.slider("Days of Prediction:", 0, 30)
#period = n_days * 1

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data....")
data = load_data(selected_stocks)
data_load_state.text("Loading data....done!")

st.subheader('Raw Data')
st.write(data.tail(10))

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period,)
forecast1 = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast1.tail(4))

st.subheader('Forecast Chart')
fig1 = plot_plotly(m, forecast1)
st.plotly_chart(fig1)

st.subheader('Forecast Components')
fig2 = m.plot_components(forecast1)
st.write(fig2)

