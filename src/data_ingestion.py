import yfinance as yf
import pandas as pd



def download_price(tickers, start_date, end_date):


    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")


    data = yf.download(tickers, start=start_date, end=end_date, progress=False)


    prices = data['Adj Close']



    print(f"Download complete. Data shape: {prices.shape}")

    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")


    return prices




