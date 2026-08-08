"""Download price data from Yahoo Finance"""

import yfinance as yf
import pandas as pd


def download_prices(tickers, start_date, end_date):
    """
    Download adjusted close prices for a list of tickers.
    
    Parameters:
    - tickers: list of strings, e.g., ['AAPL', 'MSFT']
    - start_date: string, 'YYYY-MM-DD'
    - end_date: string, 'YYYY-MM-DD'
    
    Returns:
    - pd.DataFrame with dates as index, tickers as columns
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close']
    
    print(f"Download complete. Shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    return prices


def download_vix(start_date, end_date):
    """
    Download VIX index data from Yahoo Finance.
    
    Parameters:
    - start_date: string, 'YYYY-MM-DD'
    - end_date: string, 'YYYY-MM-DD'
    
    Returns:
    - pd.Series with VIX closing prices
    """
    print(f"Downloading VIX data from {start_date} to {end_date}...")
    
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix_series = vix['Adj Close']
    
    print(f"VIX download complete. Shape: {vix_series.shape}")
    
    return vix_series