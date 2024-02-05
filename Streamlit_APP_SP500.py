# %%
# Import libraries
import pandas as pd
import streamlit as st 
import numpy as np
import yfinance as yf
from datetime import datetime

# Scrap table indicators from Wikipedia

def get_sp500_components():
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = df[0]
    tickers = df['Symbol'].to_list()
    tickers_companies_dict = dict(zip(df['Symbol'], df['Security']))
    return df, tickers, tickers_companies_dict

df,tickers, tick_dict = get_sp500_components()
# Add indicators

indicators = ['SMA_Short', 'SMA_Long', 'EWMA', 'Relative_Strength_Index']

def apply_indicator(indicator, data, window_short = 7, window_long= 90):
    assert(window_long > window_short, f" window_long smaller than window_short")
    if (indicator == 'SMA_Short') | (indicator == 'SMA_Long'):
        sma1 = data['Close'].rolling(window=window_short).mean()
        sma2 = data['Close'].rolling(window=window_long).mean()
        return pd.DataFrame({'Close': data['Close'], 'SMA_Short':sma1, 'SMA_Long':sma2})
    elif indicator == 'EWMA':
        ewma = data['Close'].ewm(halflife = 0.5, min_periods = window_long).mean()
        return pd.DataFrame({'Close': data['Close'], 'EWMA':ewma})
    elif indicator == 'Relative_Strength_Index':
        rsi = calc_rsi(data, period = window_short)
        return pd.DataFrame({'Close': data['Close'], 'RSI':rsi})
    
   
def calc_rsi(df, period):
    delta = df['Close'].diff()
    # Calculate the average gain and average loss for the specified period, with 2 masks
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    # Calculate the Relative Strength (RS) by dividing the average gain by the average loss
    rs = avg_gain / avg_loss

    # Calculate the Relative Strength Index (RSI)
    rsi = 100 - (100 / (1 + rs))

    # Add the RSI to the dataframe
    return rsi
# Apps Framework

st.title('Stock Data Analysis on the S&P500')
st.write('Sample App to download stock data and apply technical analysis indicators')

st.sidebar.header('Stock Parameters')

ticker = st.sidebar.selectbox(
    'Ticker', tickers, format_func = tick_dict.get
)
start = st.sidebar.date_input('Start_Date', pd.Timestamp('2021-01-01'))
# end day default is today´s date
end = st.sidebar.date_input('End_Date (default is today´s date)', pd.Timestamp(datetime.today().strftime('%Y-%m-%d')))
window_short = st.sidebar.number_input('Window Short', min_value=1, 
                                       max_value=252, value = 20, step = 2)
window_long = st.sidebar.number_input('Window Long', min_value=1, max_value=252, 
                                      value = 120, step= 5)

# Download data and create the charts with yfinance
data = yf.download(ticker, start, end)


selected_indicator = st.selectbox('Select a technical analysis indicator', indicators)
indicator_data = apply_indicator(selected_indicator, data, window_short, window_long)

st.write(f"{selected_indicator} for {ticker}")
st.line_chart(indicator_data)

st.write("stock data for", ticker)
st.dataframe(data)

# %% [markdown]
# # Instructions
# 
# Once you have finished your functions, save the file as .py, then go to your terminal and launch the follwing command
# 
# streamlit run streamlit_SP500_indicators.py


