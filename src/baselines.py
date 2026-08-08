"""Baseline models: constant shrinkage, VIX rule, Ledoit-Wolf, rolling average"""

import numpy as np
from scipy.linalg import norm


def frobenius_distance(A, B):
    """Compute Frobenius distance between two matrices."""
    return norm(A - B, 'fro')


def constant_shrinkage(lambdas_df):
    """
    Compute the constant lambda = mean of optimal lambdas from training period.
    
    Parameters:
    - lambdas_df: DataFrame with 'lambda_opt' column
    
    Returns:
    - float: constant lambda
    """
    lambda_const = lambdas_df['lambda_opt'].mean()
    print(f"Constant shrinkage lambda: {lambda_const:.4f}")
    return lambda_const


def evaluate_constant_baseline(window_data, lambdas_df, lambda_const):
    """
    Apply constant lambda to all windows and compute Frobenius distances.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambdas_df: DataFrame with optimal lambdas
    - lambda_const: float, constant lambda to use
    
    Returns:
    - dict with metrics
    """
    n = window_data[0]['S'].shape[0]
    I = np.eye(n)
    
    distances = []
    
    for idx, w in enumerate(window_data):
        S = w['S']
        realized = w['realized']
        
        # Shrunk covariance using constant lambda
        S_est = (1 - lambda_const) * S + lambda_const * I
        
        # Frobenius distance
        dist = frobenius_distance(S_est, realized)
        distances.append(dist)
    
    # Compute metrics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Compare to optimal lambdas
    optimal_dists = lambdas_df['min_frobenius'].values
    improvement = (mean_dist - np.mean(optimal_dists)) / np.mean(optimal_dists) * 100
    
    metrics = {
        'mean_frobenius': mean_dist,
        'std_frobenius': std_dist,
        'lambda_const': lambda_const,
        'optimal_mean_frobenius': np.mean(optimal_dists),
        'improvement_pct': improvement
    }
    
    print("\n=== Constant Shrinkage Baseline ===")
    print(f"Mean Frobenius distance: {mean_dist:.4f} (+/- {std_dist:.4f})")
    print(f"Optimal lambdas mean distance: {np.mean(optimal_dists):.4f}")
    print(f"Relative improvement: {improvement:.2f}% worse than optimal")
    
    return metrics


def vix_threshold_baseline(window_data, lambdas_df, vix_features, lambda_grid):
    """
    VIX threshold rule baseline.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambdas_df: DataFrame with optimal lambdas
    - vix_features: list of dicts with 'vix_level' for each window
    - lambda_grid: array of lambda values
    
    Returns:
    - dict with metrics and optimal threshold
    """
    n = window_data[0]['S'].shape[0]
    I = np.eye(n)
    
    # Extract VIX levels and optimal lambdas
    vix_levels = np.array([f['vix_level'] for f in vix_features])
    optimal_lambdas = lambdas_df['lambda_opt'].values
    
    # Find optimal threshold on training set
    # For simplicity, use percentiles of VIX distribution
    percentiles = np.linspace(10, 90, 17)  # 10% to 90% in 5% steps
    best_threshold = None
    best_frobenius = np.inf
    
    for pct in percentiles:
        threshold = np.percentile(vix_levels, pct)
        
        # Compute lambda for each window based on threshold
        lambda_pred = np.where(vix_levels < threshold, 
                               optimal_lambdas[vix_levels < threshold].mean(),
                               optimal_lambdas[vix_levels >= threshold].mean())
        
        # Compute Frobenius distances
        distances = []
        for idx, w in enumerate(window_data):
            S = w['S']
            realized = w['realized']
            
            lam = lambda_pred[idx]
            S_est = (1 - lam) * S + lam * I
            dist = frobenius_distance(S_est, realized)
            distances.append(dist)
        
        mean_dist = np.mean(distances)
        
        if mean_dist < best_frobenius:
            best_frobenius = mean_dist
            best_threshold = threshold
    
    # Apply best threshold to all windows
    lambda_pred = np.where(vix_levels < best_threshold,
                           optimal_lambdas[vix_levels < best_threshold].mean(),
                           optimal_lambdas[vix_levels >= best_threshold].mean())
    
    distances = []
    for idx, w in enumerate(window_data):
        S = w['S']
        realized = w['realized']
        
        lam = lambda_pred[idx]
        S_est = (1 - lam) * S + lam * I
        dist = frobenius_distance(S_est, realized)
        distances.append(dist)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    metrics = {
        'mean_frobenius': mean_dist,
        'std_frobenius': std_dist,
        'best_threshold': best_threshold,
        'n_windows': len(window_data)
    }
    
    print("\n=== VIX Threshold Baseline ===")
    print(f"Optimal VIX threshold: {best_threshold:.2f}")
    print(f"Mean Frobenius distance: {mean_dist:.4f} (+/- {std_dist:.4f})")
    
    return metrics


def ledoit_wolf_baseline(window_data):
    """
    Ledoit-Wolf shrinkage estimator.
    Simplified implementation using scikit-learn.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    
    Returns:
    - dict with metrics
    """
    from sklearn.covariance import LedoitWolf
    
    distances = []
    
    for idx, w in enumerate(window_data):
        # Reconstruct returns for this window (we need the original data)
        # Note: This is a simplified implementation
        # In practice, we need the actual returns data
        S = w['S']
        realized = w['realized']
        n = S.shape[0]
        
        # For Week 1, we'll use a simple approximation
        # In Week 2+, we'll implement proper Ledoit-Wolf
        # For now, use a placeholder
        lam_lw = 0.5  # Placeholder
        I = np.eye(n)
        S_est = (1 - lam_lw) * S + lam_lw * I
        dist = frobenius_distance(S_est, realized)
        distances.append(dist)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    metrics = {
        'mean_frobenius': mean_dist,
        'std_frobenius': std_dist,
        'n_windows': len(window_data)
    }
    
    print("\n=== Ledoit-Wolf Baseline ===")
    print(f"Mean Frobenius distance: {mean_dist:.4f} (+/- {std_dist:.4f})")
    print("Note: Using placeholder implementation. Full implementation in Week 2.")
    
    return metrics


def rolling_average_baseline(window_data, lambdas_df, window_size=10):
    """
    Rolling average lambda baseline.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambdas_df: DataFrame with optimal lambdas
    - window_size: int, number of past windows to average
    
    Returns:
    - dict with metrics
    """
    n = window_data[0]['S'].shape[0]
    I = np.eye(n)
    
    optimal_lambdas = lambdas_df['lambda_opt'].values
    distances = []
    
    for idx, w in enumerate(window_data):
        S = w['S']
        realized = w['realized']
        
        # Use rolling average of past K optimal lambdas
        if idx < window_size:
            lam_rolling = np.mean(optimal_lambdas[:max(1, idx+1)])
        else:
            lam_rolling = np.mean(optimal_lambdas[idx-window_size:idx])
        
        S_est = (1 - lam_rolling) * S + lam_rolling * I
        dist = frobenius_distance(S_est, realized)
        distances.append(dist)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    metrics = {
        'mean_frobenius': mean_dist,
        'std_frobenius': std_dist,
        'rolling_window': window_size,
        'n_windows': len(window_data)
    }
    
    print("\n=== Rolling Average Baseline ===")
    print(f"Rolling window size: {window_size}")
    print(f"Mean Frobenius distance: {mean_dist:.4f} (+/- {std_dist:.4f})")
    
    return metrics