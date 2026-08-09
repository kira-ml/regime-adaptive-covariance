"""Create rolling windows and compute covariance matrices"""

import numpy as np
import pandas as pd


def create_windows(returns, window_size, horizon):
    """
    Generate rolling window indices.
    
    Parameters:
    - returns: pd.DataFrame of daily returns
    - window_size: int, number of days in estimation window (e.g., 120)
    - horizon: int, number of days ahead for realized covariance (e.g., 20)
    
    Returns:
    - list of dicts with 'train_start', 'train_end', 'test_start', 'test_end'
    """
    n = len(returns)
    windows = []
    
    for i in range(window_size, n - horizon):
        windows.append({
            'train_start': i - window_size,
            'train_end': i,
            'test_start': i,
            'test_end': i + horizon
        })
    
    print(f"Created {len(windows)} rolling windows")
    return windows


def compute_covariance_matrices(returns, windows):
    """
    For each window, compute sample covariance S and realized covariance.
    Target T is identity matrix (handled in optimal_lambda.py).
    
    Parameters:
    - returns: pd.DataFrame of daily returns
    - windows: list of window dicts from create_windows()
    
    Returns:
    - list of dicts with 'S', 'realized', 'dates'
    """
    window_data = []
    
    for w in windows:
        # Training period (estimation window)
        train_returns = returns.iloc[w['train_start']:w['train_end']]
        
        # Test period (for realized covariance)
        test_returns = returns.iloc[w['test_start']:w['test_end']]
        
        # Sample covariance matrix (S)
        S = train_returns.cov().values
        
        # Realized covariance matrix (Sigma_realized)
        realized = test_returns.cov().values
        
        # Get the index dates for reference
        train_dates = (returns.index[w['train_start']], returns.index[w['train_end']-1])
        test_dates = (returns.index[w['test_start']], returns.index[w['test_end']-1])
        
        window_data.append({
            'S': S,
            'realized': realized,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'n_assets': S.shape[0]
        })
    
    print(f"Computed covariance matrices for {len(window_data)} windows")
    return window_data


def compute_regime_features(returns, vix, windows):
    """
    Compute regime features for each window.
    
    Parameters:
    - returns: pd.DataFrame of daily returns
    - vix: pd.Series of VIX aligned with returns
    - windows: list of window dicts from create_windows()
    
    Returns:
    - list of dicts with regime features for each window
    """
    features = []
    
    for w in windows:
        # Get returns and VIX for this window
        window_returns = returns.iloc[w['train_start']:w['train_end']]
        window_vix = vix.iloc[w['train_start']:w['train_end']]
        
        # 1. VIX level (end of window)
        vix_level = window_vix.iloc[-1]
        
        # 2. VIX percentile over past year (252 trading days)
        # Use all available history up to window end
        vix_history = vix.iloc[:w['train_end']]
        vix_percentile = (vix_history <= vix_level).mean() * 100
        
        # 3. Realized volatility (20-day) of equally-weighted portfolio
        portfolio_returns = window_returns.mean(axis=1)
        realized_vol = portfolio_returns.tail(20).std() * np.sqrt(252)
        
        # 4. Average pairwise correlation
        corr_matrix = window_returns.corr().values
        n = corr_matrix.shape[0]
        # Get upper triangle excluding diagonal
        upper_tri_indices = np.triu_indices(n, k=1)
        avg_correlation = np.mean(corr_matrix[upper_tri_indices])
        
        # 5. Cross-sectional dispersion
        cross_sectional_dispersion = window_returns.iloc[-1].std()
        
        # 6. Market return over window
        market_return = (window_returns.mean(axis=1).iloc[-1] - 
                        window_returns.mean(axis=1).iloc[0])
        
        # 7. Maximum drawdown over window
        cumulative = (1 + window_returns.mean(axis=1)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 8. Condition number of sample covariance
        S = window_returns.cov().values
        eigvals = np.linalg.eigvalsh(S)
        condition_number = eigvals.max() / eigvals.min()
        
        # 9. Trace of sample covariance (total variance)
        trace = np.trace(S)
        
        # 10. Average eigenvalue magnitude
        avg_eigenvalue = eigvals.mean()
        
        features.append({
            'window_id': len(features),
            'vix_level': vix_level,
            'vix_percentile': vix_percentile,
            'realized_vol': realized_vol,
            'avg_correlation': avg_correlation,
            'cross_sectional_dispersion': cross_sectional_dispersion,
            'market_return': market_return,
            'max_drawdown': max_drawdown,
            'condition_number': condition_number,
            'trace': trace,
            'avg_eigenvalue': avg_eigenvalue
        })
    
    print(f"Computed regime features for {len(features)} windows")
    return features





def split_windows_by_date(windows, returns, train_end='2015-12-31', val_end='2019-12-31'):
    """
    Split windows chronologically into train/validation/test sets.
    
    Parameters:
    - windows: list of window dicts from create_windows()
    - returns: pd.DataFrame with returns index
    - train_end: str, date for end of training period
    - val_end: str, date for end of validation period
    
    Returns:
    - dict with 'train', 'val', 'test' indices
    """
    import pandas as pd
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)
    
    for idx, w in enumerate(windows):
        # Get the end date of the training window
        train_end_idx = w['train_end']
        window_end_date = returns.index[train_end_idx - 1]  # Last day of training window
        
        if window_end_date <= train_end_date:
            train_indices.append(idx)
        elif window_end_date <= val_end_date:
            val_indices.append(idx)
        else:
            test_indices.append(idx)
    
    print(f"Split windows:")
    print(f"  Training: {len(train_indices)} windows (through {train_end})")
    print(f"  Validation: {len(val_indices)} windows ({train_end} to {val_end})")
    print(f"  Test: {len(test_indices)} windows (after {val_end})")
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }