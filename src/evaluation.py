"""Evaluation metrics and visualization"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_lambdas_over_time(lambdas_df, save_path=None):
    """
    Plot optimal lambda over time.
    
    Parameters:
    - lambdas_df: DataFrame with 'test_end' and 'lambda_opt' columns
    - save_path: str, path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert test_end to datetime for plotting
    dates = pd.to_datetime(lambdas_df['test_end'])
    
    ax.plot(dates, lambdas_df['lambda_opt'], linewidth=1.5, color='darkblue')
    ax.axhline(y=lambdas_df['lambda_opt'].mean(), color='red', linestyle='--', 
               label=f'Mean λ = {lambdas_df["lambda_opt"].mean():.3f}')
    ax.fill_between(dates, 0, 1, alpha=0.05)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Optimal Shrinkage Intensity (λ)')
    ax.set_title('Optimal Shrinkage Intensity Over Time (120-day window, 20-day horizon)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary stats as text
    stats_text = f"Mean: {lambdas_df['lambda_opt'].mean():.3f}\n"
    stats_text += f"Std: {lambdas_df['lambda_opt'].std():.3f}\n"
    stats_text += f"Min: {lambdas_df['lambda_opt'].min():.3f}\n"
    stats_text += f"Max: {lambdas_df['lambda_opt'].max():.3f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_frobenius_comparison(lambdas_df, baseline_metrics, save_path=None):
    """
    Plot comparison of optimal vs constant shrinkage Frobenius distances.
    
    Parameters:
    - lambdas_df: DataFrame with optimal lambdas and min_frobenius
    - baseline_metrics: dict from evaluate_constant_baseline()
    - save_path: str, path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distribution of Frobenius distances
    ax1.hist(lambdas_df['min_frobenius'], bins=30, alpha=0.7, label='Optimal', color='green')
    
    ax1.axvline(baseline_metrics['mean_frobenius'], color='red', linestyle='--', 
                linewidth=2, label=f"Constant (mean = {baseline_metrics['mean_frobenius']:.3f})")
    ax1.axvline(baseline_metrics['optimal_mean_frobenius'], color='blue', linestyle='--', 
                linewidth=2, label=f"Optimal (mean = {baseline_metrics['optimal_mean_frobenius']:.3f})")
    
    ax1.set_xlabel('Frobenius Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Frobenius Distances')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparison
    labels = ['Optimal λ', 'Constant λ']
    values = [baseline_metrics['optimal_mean_frobenius'], baseline_metrics['mean_frobenius']]
    colors = ['green', 'red']
    
    ax2.bar(labels, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Mean Frobenius Distance')
    ax2.set_title(f"Improvement: {baseline_metrics['improvement_pct']:.1f}% worse than optimal")
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_feature_correlation(features_df, save_path=None):
    """
    Plot correlation matrix of regime features.
    
    Parameters:
    - features_df: DataFrame with regime features
    - save_path: str, path to save figure (optional)
    """
    feature_cols = ['vix_level', 'vix_percentile', 'realized_vol', 
                    'avg_correlation', 'cross_sectional_dispersion',
                    'market_return', 'max_drawdown', 'condition_number',
                    'trace', 'avg_eigenvalue']
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in features_df.columns]
    corr_matrix = features_df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(available_cols)))
    ax.set_yticks(np.arange(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=45, ha='right')
    ax.set_yticklabels(available_cols)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    ax.set_title('Correlation Matrix of Regime Features')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def save_metrics(metrics, lambdas_df, filepath):
    """
    Save metrics to CSV.
    
    Parameters:
    - metrics: dict from evaluate_constant_baseline()
    - lambdas_df: DataFrame with optimal lambdas
    - filepath: str, path to save CSV
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create summary DataFrame
    summary = pd.DataFrame([{
        'metric': 'mean_frobenius_optimal',
        'value': metrics['optimal_mean_frobenius']
    }, {
        'metric': 'mean_frobenius_constant',
        'value': metrics['mean_frobenius']
    }, {
        'metric': 'improvement_pct',
        'value': metrics['improvement_pct']
    }, {
        'metric': 'lambda_const',
        'value': metrics['lambda_const']
    }, {
        'metric': 'lambda_mean',
        'value': lambdas_df['lambda_opt'].mean()
    }, {
        'metric': 'lambda_std',
        'value': lambdas_df['lambda_opt'].std()
    }, {
        'metric': 'n_windows',
        'value': len(lambdas_df)
    }])
    
    summary.to_csv(filepath, index=False)
    print(f"Metrics saved to: {filepath}")