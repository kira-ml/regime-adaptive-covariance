"""Week 1: Regime-Adaptive Covariance Estimation - Minimal Pipeline"""

import numpy as np
import pandas as pd
import os

from src.data_ingestion import download_prices, download_vix
from src.data_preprocessing import compute_returns, validate_data, clean_returns, align_vix_with_returns
from src.rolling_windows import create_windows, compute_covariance_matrices, compute_regime_features
from src.optimal_lambda import compute_optimal_lambdas
from src.baselines import constant_shrinkage, evaluate_constant_baseline
from src.baselines import vix_threshold_baseline, rolling_average_baseline
from src.evaluation import (plot_lambdas_over_time, plot_frobenius_comparison, 
                           plot_feature_correlation, save_metrics)


def main():
    print("=" * 60)
    print("Regime-Adaptive Covariance Estimation - Week 1 Pipeline")
    print("=" * 60)
    
    # ============================================
    # CONFIGURATION (hardcoded for Week 1)
    # ============================================
    TICKERS = [
        # Technology
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'CSCO',
        # Financials
        'JPM', 'BAC', 'GS', 'V',
        # Healthcare
        'JNJ', 'PFE', 'UNH',
        # Consumer
        'PG', 'KO', 'WMT',
        # Industrials
        'XOM', 'CAT', 'BA',
        # Other
        'DIS', 'NKE'
    ]
    START_DATE = '2000-01-01'
    END_DATE = '2025-01-01'
    WINDOW_SIZE = 120      # trading days (~6 months)
    HORIZON = 20           # trading days (~1 month)
    LAMBDA_GRID = np.linspace(0, 0.5, 51)  # 0.00, 0.01, 0.02, ..., 0.50
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {TICKERS}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Window: {WINDOW_SIZE} days")
    print(f"  Horizon: {HORIZON} days")
    print(f"  Lambda grid: {len(LAMBDA_GRID)} values from {LAMBDA_GRID[0]:.2f} to {LAMBDA_GRID[-1]:.2f}")
    
    # ============================================
    # 1. DATA INGESTION
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 1: Data Ingestion")
    print("=" * 60)
    
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    vix = download_vix(START_DATE, END_DATE)
    
    # ============================================
    # 2. DATA PREPROCESSING
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)
    
    returns = compute_returns(prices)
    validation = validate_data(returns)
    
    if not validation['valid']:
        print("Warning: Data issues found. Cleaning...")
        returns = clean_returns(returns)
        validation = validate_data(returns)
    
    # Align VIX with returns
    returns, vix_aligned = align_vix_with_returns(returns, vix)
    
    print(f"Final data shape: {returns.shape}")
    
    # ============================================
    # 3. ROLLING WINDOWS
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 3: Rolling Windows")
    print("=" * 60)
    
    windows = create_windows(returns, WINDOW_SIZE, HORIZON)
    window_data = compute_covariance_matrices(returns, windows)
    features = compute_regime_features(returns, vix_aligned, windows)
    
    # Save features for analysis
    features_df = pd.DataFrame(features)
    features_df.to_csv('data/processed/regime_features.csv', index=False)
    print("Saved regime features to: data/processed/regime_features.csv")
    
    # ============================================
    # 4. OPTIMAL LAMBDA
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 4: Finding Optimal Lambda")
    print("=" * 60)
    
    lambdas_df = compute_optimal_lambdas(window_data, LAMBDA_GRID)
    
    # Save lambdas to CSV
    lambdas_df.to_csv('data/processed/optimal_lambdas.csv', index=False)
    print("Saved optimal lambdas to: data/processed/optimal_lambdas.csv")
    
    # ============================================
    # 5. BASELINE MODELS
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 5: Baseline Models")
    print("=" * 60)
    
    # Baseline 1: Constant Shrinkage
    lambda_const = constant_shrinkage(lambdas_df)
    baseline_metrics = evaluate_constant_baseline(window_data, lambdas_df, lambda_const)
    
    # Baseline 2: VIX Threshold Rule
    vix_metrics = vix_threshold_baseline(window_data, lambdas_df, features, LAMBDA_GRID)
    
    # Baseline 3: Rolling Average
    rolling_metrics = rolling_average_baseline(window_data, lambdas_df, window_size=10)
    
    # ============================================
    # 6. EVALUATION & VISUALIZATION
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation & Visualization")
    print("=" * 60)
    
    # Plot 1: Lambda over time
    plot_lambdas_over_time(lambdas_df, save_path='results/figures/lambda_over_time.png')
    
    # Plot 2: Frobenius comparison
    plot_frobenius_comparison(lambdas_df, baseline_metrics, 
                              save_path='results/figures/frobenius_comparison.png')
    
    # Plot 3: Feature correlation
    plot_feature_correlation(features_df, save_path='results/figures/feature_correlation.png')
    
    # Save metrics
    save_metrics(baseline_metrics, lambdas_df, 'results/metrics.csv')
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    # Check if we have data before printing summary
    if len(lambdas_df) == 0:
        print("\nWARNING: No windows were processed. No data available.")
        print("This is likely due to Yahoo Finance not returning data.")
        print("Please check:")
        print("  1. Your internet connection")
        print("  2. Try with a smaller date range (e.g., START_DATE = '2020-01-01')")
        print("  3. Try a single ticker: TICKERS = ['AAPL']")
        print("  4. The available columns printed above show what data was returned")
        print("\n" + "=" * 60)
        return
    
    print(f"\nSummary:")
    print(f"  Total windows processed: {len(lambdas_df)}")
    print(f"  Mean optimal lambda: {lambdas_df['lambda_opt'].mean():.4f}")
    print(f"  Constant shrinkage lambda: {lambda_const:.4f}")
    print(f"  Constant baseline mean Frobenius: {baseline_metrics['mean_frobenius']:.4f}")
    print(f"  Optimal mean Frobenius: {baseline_metrics['optimal_mean_frobenius']:.4f}")
    print(f"  Relative difference: {baseline_metrics['improvement_pct']:.1f}%")
    print(f"  VIX threshold baseline mean Frobenius: {vix_metrics['mean_frobenius']:.4f}")
    print(f"  Rolling average baseline mean Frobenius: {rolling_metrics['mean_frobenius']:.4f}")
    print("\nResults saved to:")
    print("  - data/processed/optimal_lambdas.csv")
    print("  - data/processed/regime_features.csv")
    print("  - results/metrics.csv")
    print("  - results/figures/lambda_over_time.png")
    print("  - results/figures/frobenius_comparison.png")
    print("  - results/figures/feature_correlation.png")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()