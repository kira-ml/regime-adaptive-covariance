"""Preprocess price data: compute returns, handle missing values"""

import pandas as pd
import numpy as np


def compute_returns(prices):
    """
    Compute daily returns from adjusted close prices.
    
    Parameters:
    - prices: pd.DataFrame with dates as index, tickers as columns
    
    Returns:
    - pd.DataFrame of daily returns (pct_change)
    """
    returns = prices.pct_change().dropna()
    print(f"Returns computed. Shape: {returns.shape}")
    return returns


def validate_data(returns):
    """
    Basic validation checks on returns data.
    
    Parameters:
    - returns: pd.DataFrame of daily returns
    
    Returns:
    - dict with validation results
    """
    # Handle empty DataFrame
    if returns.empty or returns.shape[0] == 0:
        print("WARNING: Returns data is empty. Skipping validation.")
        return {
            'n_assets': 0,
            'n_days': 0,
            'n_missing': 0,
            'zero_var_assets': [],
            'n_inf': 0,
            'valid': False
        }
    
    n_assets = returns.shape[1]
    n_days = returns.shape[0]
    
    # Check for missing values
    n_missing = returns.isnull().sum().sum()
    
    # Check for zero variance (constant price)
    variances = returns.var()
    zero_var_assets = variances[variances == 0].index.tolist()
    
    # Check for infinite values (convert to float first)
    try:
        n_inf = np.isinf(returns.values).sum()
    except TypeError:
        # If values are not numeric, try converting
        try:
            returns_numeric = returns.astype(float)
            n_inf = np.isinf(returns_numeric.values).sum()
        except:
            n_inf = 0
    
    results = {
        'n_assets': n_assets,
        'n_days': n_days,
        'n_missing': n_missing,
        'zero_var_assets': zero_var_assets,
        'n_inf': n_inf,
        'valid': (n_missing == 0 and len(zero_var_assets) == 0 and n_inf == 0 and n_days > 0 and n_assets > 0)
    }
    
    print(f"Validation: {results}")
    return results


def clean_returns(returns):
    """
    Clean returns by removing any remaining missing or infinite values.
    
    Parameters:
    - returns: pd.DataFrame of daily returns
    
    Returns:
    - pd.DataFrame of cleaned returns
    """
    # Handle empty DataFrame
    if returns.empty:
        print("WARNING: Returns data is empty. Nothing to clean.")
        return returns
    
    # Remove any rows with missing values
    returns = returns.dropna()
    
    # Remove any columns with zero variance (if any columns remain)
    if not returns.empty and returns.shape[1] > 0:
        returns = returns.loc[:, returns.var() > 0]
    
    # Replace infinite with NaN and drop
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Cleaned returns. Shape: {returns.shape}")
    return returns


def align_vix_with_returns(returns, vix):
    """
    Align VIX data with returns index.
    
    Parameters:
    - returns: pd.DataFrame of daily returns
    - vix: pd.Series of VIX closing prices
    
    Returns:
    - pd.Series of VIX aligned to returns index
    """
    # Handle empty returns
    if returns.empty:
        print("WARNING: Returns data is empty. Cannot align VIX.")
        return returns, pd.Series()
    
    # Reindex VIX to returns index, forward fill missing values
    vix_aligned = vix.reindex(returns.index).ffill()
    
    # Check for any remaining missing values
    if vix_aligned.isnull().any():
        print(f"Warning: {vix_aligned.isnull().sum()} missing VIX values remain")
        # Drop rows where VIX is missing
        valid_mask = vix_aligned.notna()
        if valid_mask.any():
            vix_aligned = vix_aligned[valid_mask]
            returns = returns[valid_mask]
        else:
            print("WARNING: No valid VIX data. Using empty returns.")
            return pd.DataFrame(), pd.Series()
    
    print(f"VIX aligned. Shape: {vix_aligned.shape}")
    
    return returns, vix_aligned