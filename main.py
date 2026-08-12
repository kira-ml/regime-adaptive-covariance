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
from src.feature_engineering import run_feature_engineering
from src.elastic_net import evaluate_elastic_net_on_covariance
from src.statistical_tests import run_statistical_tests
from src.xgboost_model import evaluate_xgboost_on_covariance


def main():
    print("=" * 60)
    print("Regime-Adaptive Covariance Estimation - Week 1 Pipeline")
    print("=" * 60)
    
    # ============================================
    # CONFIGURATION (hardcoded for Week 1)
    # ============================================
    TICKERS = [
        # Technology (10)
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'CSCO', 'ADBE', 'INTC',
        # Financials (8)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'BLK',
        # Healthcare (8)
        'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'AMGN', 'GILD', 'BMY',
        # Consumer (8)
        'PG', 'KO', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX',
        # Industrials (8)
        'XOM', 'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX',
        # Other (8)
        'DIS', 'T', 'VZ', 'CVX', 'IBM', 'QCOM', 'TXN', 'NFLX'
    ]
    START_DATE = '2000-01-01'
    END_DATE = '2025-01-01'
    WINDOW_SIZE = 120      # trading days (~6 months)
    HORIZON = 20           # trading days (~1 month)
    LAMBDA_GRID = np.linspace(0, 0.2, 101)  # 0.00, 0.002, 0.004, ..., 0.20
    
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
    
    # Split windows chronologically
    from src.rolling_windows import split_windows_by_date
    split = split_windows_by_date(windows, returns, train_end='2015-12-31', val_end='2019-12-31')
    
    # Get training indices
    train_indices = split['train']
    
    if len(train_indices) > 0:
        # Subset data to training set only
        lambdas_train = lambdas_df.iloc[train_indices]
        window_data_train = [window_data[i] for i in train_indices]
        
        # Baseline 1: Constant Shrinkage (trained on training set)
        lambda_const = constant_shrinkage(lambdas_train)
        baseline_metrics = evaluate_constant_baseline(window_data, lambdas_df, lambda_const)
        
        # Baseline 2: VIX Threshold Rule (trained on training set)
        features_train = [features[i] for i in train_indices]
        lambdas_train = lambdas_df.iloc[train_indices]
        # Pass full lambdas for evaluation, training lambdas for threshold search
        vix_metrics = vix_threshold_baseline(window_data, lambdas_train, lambdas_df, features_train, features, LAMBDA_GRID)
    else:
        print("WARNING: No training windows found. Using all data.")
        lambda_const = constant_shrinkage(lambdas_df)
        baseline_metrics = evaluate_constant_baseline(window_data, lambdas_df, lambda_const)
        vix_metrics = vix_threshold_baseline(window_data, lambdas_df, features, LAMBDA_GRID)
    
    # Baseline 3: Rolling Average (still uses all data for now)
    rolling_metrics = rolling_average_baseline(window_data, lambdas_df, window_size=10)
    
    # Baseline 4: Ledoit-Wolf
    from src.baselines import ledoit_wolf_baseline
    lw_metrics = ledoit_wolf_baseline(window_data)



    # ============================================
    # FEATURE ENGINEERING (NEW STEP)
    # ============================================
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING: Baseline vs Advanced Sets")
    print("=" * 60)
    
    from src.feature_engineering import run_feature_engineering
    
    comparison, engineer = run_feature_engineering(
        features_path='data/processed/regime_features.csv',
        lambdas_path='data/processed/optimal_lambdas.csv',
        train_indices=split['train'],
        val_indices=split['val'],
        test_indices=split['test']
    )
    
    # Save comparison
    comparison.to_csv('results/feature_set_comparison.csv', index=False)
    print("Saved feature set comparison to: results/feature_set_comparison.csv")


    # ============================================
    # ELASTIC NET (NEW STEP)
    # ============================================
    print("\n" + "=" * 60)
    print("ELASTIC NET: Using Best Feature Set (VIX-Only)")
    print("=" * 60)

    from src.elastic_net import evaluate_elastic_net_on_covariance

    # Use the best feature set from comparison
    best_set_name = comparison.loc[comparison['rmse'].idxmin(), 'set_name']
    feature_cols = {
        'VIX-Only': ['vix_level'],
        'Vol+Corr': ['vix_level', 'realized_vol', 'avg_correlation'],
        'Market': ['vix_level', 'realized_vol', 'avg_correlation',
                   'cross_sectional_dispersion', 'vix_percentile', 'max_drawdown'],
        'Covariance': ['condition_number', 'trace', 'avg_eigenvalue', 'avg_correlation'],
        'All': ['vix_level', 'vix_percentile', 'realized_vol', 'avg_correlation',
                'cross_sectional_dispersion', 'market_return', 'max_drawdown',
                'condition_number', 'trace', 'avg_eigenvalue']
    }[best_set_name]

    en_metrics = evaluate_elastic_net_on_covariance(
        window_data=window_data,
        lambdas_df=lambdas_df,
        feature_cols=feature_cols,
        train_idx=split['train'],
        val_idx=split['val'],
        test_idx=split['test'],
        features_df=features_df
    )

    print(f"\n=== Elastic Net Results (Test Set) ===")
    print(f"Best hyperparameters: {en_metrics['best_params']}")
    print(f"RMSE of λ prediction: {en_metrics['rmse']:.6f}")
    print(f"R² of λ prediction: {en_metrics['r2']:.4f}")
    print(f"Mean Frobenius distance: {en_metrics['mean_frobenius']:.4f} (+/- {en_metrics['std_frobenius']:.4f})")

    # Save results
    pd.DataFrame([{
        'model': 'ElasticNet',
        'feature_set': best_set_name,
        'rmse_lambda': en_metrics['rmse'],
        'r2_lambda': en_metrics['r2'],
        'mean_frobenius': en_metrics['mean_frobenius'],
        'std_frobenius': en_metrics['std_frobenius'],
        'best_alpha': en_metrics['best_params']['alpha'],
        'best_l1_ratio': en_metrics['best_params']['l1_ratio']
    }]).to_csv('results/elastic_net_results.csv', index=False)
    print("Saved Elastic Net results to: results/elastic_net_results.csv")



    # ============================================
    # XGBOOST (NEW STEP - For Testing/Completeness)
    # ============================================
    print("\n" + "=" * 60)
    print("XGBOOST: Using Best Feature Set (VIX-Only)")
    print("=" * 60)

    xgb_metrics = evaluate_xgboost_on_covariance(
        window_data=window_data,
        lambdas_df=lambdas_df,
        feature_cols=feature_cols,
        train_idx=split['train'],
        val_idx=split['val'],
        test_idx=split['test'],
        features_df=features_df
    )

    print(f"\n=== XGBoost Results (Test Set) ===")
    print(f"Best hyperparameters: {xgb_metrics['best_params']}")
    print(f"RMSE of λ prediction: {xgb_metrics['rmse']:.6f}")
    print(f"R² of λ prediction: {xgb_metrics['r2']:.4f}")
    print(f"Mean Frobenius distance: {xgb_metrics['mean_frobenius']:.4f} (+/- {xgb_metrics['std_frobenius']:.4f})")

    # Save results
    pd.DataFrame([{
        'model': 'XGBoost',
        'feature_set': best_set_name,
        'rmse_lambda': xgb_metrics['rmse'],
        'r2_lambda': xgb_metrics['r2'],
        'mean_frobenius': xgb_metrics['mean_frobenius'],
        'std_frobenius': xgb_metrics['std_frobenius'],
        'best_max_depth': xgb_metrics['best_params']['max_depth'],
        'best_learning_rate': xgb_metrics['best_params']['learning_rate'],
        'best_n_estimators': xgb_metrics['best_params']['n_estimators'],
        'best_subsample': xgb_metrics['best_params']['subsample']
    }]).to_csv('results/xgboost_results.csv', index=False)
    print("Saved XGBoost results to: results/xgboost_results.csv")

    # ============================================
    # PORTFOLIO EVALUATION (TEST SET ONLY)
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 6: Portfolio Evaluation (Test Set Only)")
    print("=" * 60)
    
    from src.portfolio import evaluate_portfolio_performance
    
    # Get test indices
    test_indices = split['test']
    
    if len(test_indices) > 0:
        # Subset to test set only
        window_data_test = [window_data[i] for i in test_indices]
        lambdas_test = lambdas_df.iloc[test_indices]
        
        # Define methods to evaluate (using test set only)
        methods = {
            'Optimal': (None, {}),
            'Constant': (None, {'lambda_const': lambda_const}),
            'VIX Threshold': (None, {}),
            'Rolling Average': (None, {}),
            'Ledoit-Wolf': (None, {})
        }
        
        portfolio_results = evaluate_portfolio_performance(window_data_test, lambdas_test, methods)
        print(portfolio_results.to_string(index=False))
        
        # Save portfolio results (test set only)
        portfolio_results.to_csv('results/portfolio_metrics_test.csv', index=False)
        print("Saved portfolio metrics (test set) to: results/portfolio_metrics_test.csv")
    else:
        print("WARNING: No test windows available for portfolio evaluation.")
        portfolio_results = None

    # ============================================
    # STATISTICAL TESTS (NEW STEP)
    # ============================================
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS: Diebold-Mariano & Bootstrap")
    print("=" * 60)

    from src.statistical_tests import run_statistical_tests

    stats_results = run_statistical_tests(
        window_data=window_data,
        lambdas_df=lambdas_df,
        baseline_metrics=baseline_metrics,
        lw_metrics=lw_metrics,
        test_idx=split['test']
    )

    # Print results
    print("\n=== Diebold-Mariano Test (Frobenius Distance) ===")
    for name, res in stats_results['diebold_mariano'].items():
        print(f"{name}:")
        print(f"  DM statistic: {res['statistic']:.4f}")
        print(f"  p-value: {res['p_value']:.4f}")
        print(f"  H0: {res['h0']}")

    print("\n=== Bootstrap Test (Volatility Difference) ===")
    alpha = 0.05  # Significance level for 95% CI
    for name, res in stats_results['bootstrap'].items():
        print(f"{name}:")
        print(f"  Mean diff (method1 - method2): {res['mean_diff']:.6f}")
        print(f"  {1-alpha:.0%} CI: [{res['ci_lower']:.6f}, {res['ci_upper']:.6f}]")
        print(f"  p-value (one-sided): {res['p_value']:.4f}")

    # Save results
    import json
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return obj

    stats_json = {}
    for key, val in stats_results.items():
        stats_json[key] = convert_for_json(val)

    with open('results/statistical_tests.json', 'w') as f:
        json.dump(stats_json, f, indent=2)
    print("\nSaved statistical test results to: results/statistical_tests.json")


    # ============================================
    # SUB-PERIOD ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 7: Sub-Period Analysis")
    print("=" * 60)
    
    from src.sub_period_analysis import evaluate_sub_periods, analyze_sub_period_results, plot_sub_period_comparison
    
    # Define methods for sub-period analysis
    methods = {
        'Optimal': (None, {}),
        'Constant': (None, {'lambda_const': lambda_const}),
        'VIX Threshold': (None, {}),
        'Rolling Average': (None, {}),
        'Ledoit-Wolf': (None, {})
    }
    
    sub_period_results = evaluate_sub_periods(
        window_data, lambdas_df, windows, returns, split, methods
    )
    
    if not sub_period_results.empty:
        # Save results
        sub_period_results.to_csv('results/sub_period_results.csv', index=False)
        print("Saved sub-period results to: results/sub_period_results.csv")
        
        # Generate summary
        summary = analyze_sub_period_results(sub_period_results)
        print("\n=== Sub-Period Summary ===")
        print(summary.to_string(index=False))
        summary.to_csv('results/sub_period_summary.csv', index=False)
        
        # Plot
        plot_sub_period_comparison(
            sub_period_results,
            save_path='results/figures/sub_period_comparison.png'
        )
    else:
        print("No sub-period results available.")



    
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
    print(f"  Ledoit-Wolf baseline mean Frobenius: {lw_metrics['mean_frobenius']:.4f}")
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