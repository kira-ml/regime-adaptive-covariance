"""Sub-period analysis for regime-dependent performance"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_sub_periods():
    """
    Define market sub-periods for analysis.
    
    Returns:
    - dict: {name: (start_date, end_date)}
    """
    return {
        'Normal (2000-2007)': ('2000-01-01', '2007-07-01'),
        'GFC Crash (2007-2009)': ('2007-07-01', '2009-03-01'),
        'Recovery (2009-2014)': ('2009-03-01', '2014-12-31'),
        'Low Vol (2015-2019)': ('2015-01-01', '2019-12-31'),
        'COVID Crash (2020)': ('2020-01-01', '2020-06-30'),
        'Recovery (2020-2021)': ('2020-07-01', '2021-12-31'),
        'Bear Market (2022)': ('2022-01-01', '2022-12-31'),
        'Recovery (2023-2024)': ('2023-01-01', '2024-12-31'),
    }


def filter_windows_by_date(windows, returns, start_date, end_date):
    """
    Filter windows that end within a specific date range.
    
    Parameters:
    - windows: list of window dicts
    - returns: pd.DataFrame with returns index
    - start_date: str, 'YYYY-MM-DD'
    - end_date: str, 'YYYY-MM-DD'
    
    Returns:
    - list of window indices
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    indices = []
    for idx, w in enumerate(windows):
        window_end_idx = w['train_end']
        window_end_date = returns.index[window_end_idx - 1]
        
        if start <= window_end_date <= end:
            indices.append(idx)
    
    return indices


def evaluate_sub_periods(window_data, lambdas_df, windows, returns, split_indices, methods, returns_data=None):
    """
    Evaluate methods on each sub-period.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambdas_df: DataFrame with optimal lambdas
    - windows: list of window dicts from create_windows()
    - returns: pd.DataFrame with returns index
    - split_indices: dict with 'train', 'val', 'test' indices
    - methods: dict of method_name -> (lambda_func, kwargs)
    
    Returns:
    - DataFrame with results
    """
    from src.portfolio import evaluate_portfolio_performance
    
    # Get test indices (2020-2025)
    test_indices = split_indices['test']
    
    if len(test_indices) == 0:
        print("No test indices available.")
        return pd.DataFrame()
    
    # Get sub-periods
    sub_periods = get_sub_periods()
    
    results = []
    
    for period_name, (start_date, end_date) in sub_periods.items():
        # Find windows in this sub-period
        period_indices = []
        for idx in test_indices:
            # Use the windows list (not window_data) to get train_end
            w = windows[idx]
            train_end_idx = w['train_end']
            # Use the returns index to get the date
            window_date = returns.index[train_end_idx - 1]  # Last day of training window
            
            if start_date <= str(window_date) <= end_date:
                period_indices.append(idx)
        
        if len(period_indices) == 0:
            continue
        
        print(f"\n{period_name}: {len(period_indices)} windows")
        
        # Subset data
        window_data_period = [window_data[i] for i in period_indices]
        lambdas_period = lambdas_df.iloc[period_indices]
        
        # Evaluate methods on this period
        period_results = evaluate_portfolio_performance(
            window_data_period, 
            lambdas_period,
            methods, 
            returns_data=returns_data
        )
        
        # Add period information
        period_results['period'] = period_name
        period_results['n_windows'] = len(period_indices)
        
        results.append(period_results)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def analyze_sub_period_results(results_df):
    """
    Analyze and summarize sub-period results.
    
    Parameters:
    - results_df: DataFrame from evaluate_sub_periods()
    
    Returns:
    - summary DataFrame with rankings
    """
    if results_df.empty:
        print("No results to analyze.")
        return pd.DataFrame()
    
    # Find best method per period
    summary = []
    
    for period in results_df['period'].unique():
        period_data = results_df[results_df['period'] == period]
        
        # Find method with lowest volatility
        best_idx = period_data['mean_volatility'].idxmin()
        best_method = period_data.loc[best_idx, 'method']
        best_vol = period_data.loc[best_idx, 'mean_volatility']
        
        # Get constant and optimal for comparison
        const_vol = period_data[period_data['method'] == 'Constant']['mean_volatility'].values[0]
        optimal_vol = period_data[period_data['method'] == 'Optimal']['mean_volatility'].values[0]
        
        # Calculate improvements
        const_improvement = (const_vol - best_vol) / const_vol * 100
        optimal_improvement = (optimal_vol - best_vol) / optimal_vol * 100
        
        summary.append({
            'period': period,
            'best_method': best_method,
            'best_volatility': best_vol,
            'constant_volatility': const_vol,
            'optimal_volatility': optimal_vol,
            'improvement_vs_constant': const_improvement,
            'improvement_vs_optimal': optimal_improvement,
            'n_windows': period_data['n_windows'].iloc[0]
        })
    
    return pd.DataFrame(summary)


def plot_sub_period_comparison(results_df, save_path=None):
    """
    Plot sub-period performance comparison.
    
    Parameters:
    - results_df: DataFrame from evaluate_sub_periods()
    - save_path: str, path to save figure
    """
    if results_df.empty:
        print("No data to plot.")
        return
    
    # Pivot table for heatmap
    pivot = results_df.pivot(index='period', columns='method', values='mean_volatility')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Heatmap of volatilities
    im = ax1.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(np.arange(len(pivot.columns)))
    ax1.set_yticks(np.arange(len(pivot.index)))
    ax1.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax1.set_yticklabels(pivot.index)
    ax1.set_title('Portfolio Volatility by Period and Method')
    plt.colorbar(im, ax=ax1)
    
    # Plot 2: Relative performance vs Constant
    for method in pivot.columns:
        if method == 'Constant':
            continue
        relative = (pivot[method] - pivot['Constant']) / pivot['Constant'] * 100
        ax2.plot(relative.index, relative.values, marker='o', label=method)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Volatility vs Constant (%)')
    ax2.set_title('Relative Performance vs Constant Shrinkage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sub-period analysis to: {save_path}")
    
    plt.show()