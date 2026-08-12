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
    import time
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download all tickers at once with retry logic
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Initial download failed: {e}")
        print("Retrying with 2-second delay...")
        time.sleep(2)
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Check if data is empty
    if data.empty:
        print("WARNING: No data downloaded. Check tickers or date range.")
        return pd.DataFrame(columns=tickers)
    
    # Handle different column formats
    # Case 1: Standard 'Adj Close' column
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    # Case 2: MultiIndex with ('Close', 'Ticker') format
    elif isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0):
        prices = data['Close']
    # Case 3: Try 'Close' if available
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        print(f"WARNING: Unexpected column format. Available: {data.columns}")
        return pd.DataFrame(columns=tickers)
    
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
    
    if vix.empty:
        print("WARNING: No VIX data downloaded.")
        return pd.Series()
    
    # Handle different column formats
    if 'Adj Close' in vix.columns:
        vix_series = vix['Adj Close']
    elif 'Close' in vix.columns:
        vix_series = vix['Close']
    else:
        print(f"WARNING: Unexpected VIX column format. Available: {vix.columns}")
        return pd.Series()
    
    # Convert to Series (drop column name)
    vix_series = vix_series.squeeze()
    
    print(f"VIX download complete. Shape: {vix_series.shape}")
    
    return vix_series