"""Quick feature exploration for Day 2"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_features():
    """Load data and generate correlation analysis"""
    
    # Load data
    features = pd.read_csv('data/processed/regime_features.csv')
    lambdas = pd.read_csv('data/processed/optimal_lambdas.csv')
    
    # Merge
    df = pd.merge(features, lambdas, on='window_id')
    
    # Compute correlations with lambda
    feature_cols = ['vix_level', 'vix_percentile', 'realized_vol', 
                    'avg_correlation', 'cross_sectional_dispersion',
                    'market_return', 'max_drawdown', 'condition_number',
                    'trace', 'avg_eigenvalue']
    
    correlations = {}
    for col in feature_cols:
        if col in df.columns:
            correlations[col] = df[col].corr(df['lambda_opt'])
    
    # Print results
    print("=== Feature Correlations with Optimal λ ===\n")
    for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature:30s}: {corr:.4f}")
    
    # Quick plot: λ vs VIX
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: λ vs VIX
    axes[0,0].scatter(df['vix_level'], df['lambda_opt'], alpha=0.3, s=1)
    axes[0,0].set_xlabel('VIX Level')
    axes[0,0].set_ylabel('Optimal λ')
    axes[0,0].set_title(f'λ vs VIX (corr: {correlations.get("vix_level", 0):.3f})')
    
    # Plot 2: λ vs Realized Vol
    axes[0,1].scatter(df['realized_vol'], df['lambda_opt'], alpha=0.3, s=1)
    axes[0,1].set_xlabel('Realized Volatility')
    axes[0,1].set_ylabel('Optimal λ')
    axes[0,1].set_title(f'λ vs Realized Vol (corr: {correlations.get("realized_vol", 0):.3f})')
    
    # Plot 3: λ vs Avg Correlation
    axes[1,0].scatter(df['avg_correlation'], df['lambda_opt'], alpha=0.3, s=1)
    axes[1,0].set_xlabel('Avg Correlation')
    axes[1,0].set_ylabel('Optimal λ')
    axes[1,0].set_title(f'λ vs Avg Corr (corr: {correlations.get("avg_correlation", 0):.3f})')
    
    # Plot 4: λ over time
    dates = pd.to_datetime(df['test_end'])
    axes[1,1].plot(dates, df['lambda_opt'], linewidth=0.5, alpha=0.7)
    axes[1,1].set_xlabel('Date')
    axes[1,1].set_ylabel('Optimal λ')
    axes[1,1].set_title('λ Over Time')
    
    plt.tight_layout()
    plt.savefig('results/figures/feature_analysis.png', dpi=150)
    print("\nSaved figure to: results/figures/feature_analysis.png")
    
    return df, correlations

if __name__ == "__main__":
    df, correlations = analyze_features()