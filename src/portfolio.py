"""Portfolio construction and evaluation"""

import numpy as np
import pandas as pd


def minimum_variance_portfolio(cov_matrix):
    """
    Compute minimum variance portfolio weights.
    
    Parameters:
    - cov_matrix: covariance matrix (numpy array)
    
    Returns:
    - weights: numpy array of portfolio weights
    """
    n = cov_matrix.shape[0]
    
    # Solve for minimum variance portfolio
    # min w' Σ w s.t. w' 1 = 1
    
    # Use analytical solution: w = Σ^(-1) 1 / (1' Σ^(-1) 1)
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n)
        weights = inv_cov @ ones
        weights = weights / weights.sum()
        return weights
    except np.linalg.LinAlgError:
        # If singular, use equal weights
        return np.ones(n) / n


def portfolio_volatility(weights, cov_matrix):
    """Compute portfolio volatility from covariance matrix."""
    return np.sqrt(weights.T @ cov_matrix @ weights)


def evaluate_portfolio_performance(window_data, lambdas_df, methods, returns_data=None):
    """
    Evaluate portfolio performance for different covariance estimators.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambdas_df: DataFrame with optimal lambdas
    - methods: dict of method_name -> (lambda_func, lambda_func_args)
    - returns_data: pd.DataFrame of daily returns (optional, for Sharpe & drawdown)
    
    Returns:
    - DataFrame with portfolio metrics
    """
    results = []
    
    for method_name, (lambda_func, func_kwargs) in methods.items():
        volatilities = []
        turnovers = []
        portfolio_returns = []  # For Sharpe and drawdown
        prev_weights = None
        
        for idx, w in enumerate(window_data):
            S = w['S']
            n = w['n_assets']
            I = np.eye(n)
            
            # Get lambda for this method
            if method_name == 'Optimal':
                lam = lambdas_df.iloc[idx]['lambda_opt']
            elif method_name == 'Constant':
                lam = func_kwargs.get('lambda_const', 0.0)
            elif method_name == 'Ledoit-Wolf':
                from sklearn.covariance import LedoitWolf
                if 'train_returns' in w and w['train_returns'] is not None:
                    lw = LedoitWolf()
                    lw.fit(w['train_returns'].values)
                    cov_est = lw.covariance_
                else:
                    cov_est = S
                lam = None
            else:
                lam = func_kwargs.get('lambda_pred', 0.0)
            
            if method_name != 'Ledoit-Wolf' and lam is not None:
                cov_est = (1 - lam) * S + lam * I
            
            # Compute minimum variance portfolio
            weights = minimum_variance_portfolio(cov_est)
            
            # --- Realized volatility ---
            realized_cov = w['realized']
            vol = portfolio_volatility(weights, realized_cov)
            volatilities.append(vol)
            
            # --- Turnover ---
            if prev_weights is not None:
                turnover = np.sum(np.abs(weights - prev_weights))
                turnovers.append(turnover)
            prev_weights = weights.copy()
            
            # --- Portfolio returns (for Sharpe and drawdown) ---
            if returns_data is not None:
                test_start_idx = w['test_dates'][0]
                test_end_idx = w['test_dates'][1]
                period_returns = returns_data.loc[test_start_idx:test_end_idx]
                daily_returns = period_returns.values @ weights
                portfolio_returns.extend(daily_returns)
        
        # --- Aggregate metrics ---
        mean_vol = np.mean(volatilities)
        std_vol = np.std(volatilities)
        
        mean_turnover = np.mean(turnovers) if turnovers else np.nan
        
        # Sharpe Ratio and Max Drawdown (if returns data available)
        if returns_data is not None and portfolio_returns:
            returns_series = np.array(portfolio_returns)
            excess_returns = returns_series
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
            cumulative = np.cumprod(1 + returns_series)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe = np.nan
            max_drawdown = np.nan
        
        results.append({
            'method': method_name,
            'mean_volatility': mean_vol,
            'std_volatility': std_vol,
            'mean_turnover': mean_turnover,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_windows': len(volatilities)
        })
    
    return pd.DataFrame(results)