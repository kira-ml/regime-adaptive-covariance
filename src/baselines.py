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
    # Handle empty DataFrame
    if lambdas_df.empty or 'lambda_opt' not in lambdas_df.columns:
        print("WARNING: No lambda data available. Using default lambda = 0.5")
        return 0.5
    
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
    # Handle empty window_data
    if not window_data or len(window_data) == 0:
        print("WARNING: No window data available. Returning empty metrics.")
        return {
            'mean_frobenius': np.nan,
            'std_frobenius': np.nan,
            'lambda_const': lambda_const if not np.isnan(lambda_const) else 0.5,
            'optimal_mean_frobenius': np.nan,
            'improvement_pct': np.nan
        }
    
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
    VIX threshold rule baseline with 3 regimes (low/medium/high).
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambdas_df: DataFrame with optimal lambdas (may be subset for training)
    - vix_features: list of dicts with 'vix_level' for each window (training set)
    - lambda_grid: array of lambda values
    
    Returns:
    - dict with metrics and optimal thresholds
    """
    # Handle empty data
    if not window_data or len(window_data) == 0:
        print("WARNING: No window data available. Returning empty VIX metrics.")
        return {
            'mean_frobenius': np.nan,
            'std_frobenius': np.nan,
            'threshold_low': np.nan,
            'threshold_high': np.nan,
            'n_windows': 0
        }
    
    n = window_data[0]['S'].shape[0]
    I = np.eye(n)
    
    # Store original vix features for all windows
    vix_features_original = vix_features
    
    # Extract VIX levels from training data
    vix_levels = np.array([f['vix_level'] for f in vix_features])
    
    # Extract optimal lambdas - ensure they match the same windows
    if len(lambdas_df) > len(vix_levels):
        # We're using a subset of windows (e.g., training set only)
        # Find the matching window_ids
        window_ids = [f['window_id'] for f in vix_features]
        optimal_lambdas = lambdas_df[lambdas_df['window_id'].isin(window_ids)]['lambda_opt'].values
    else:
        optimal_lambdas = lambdas_df['lambda_opt'].values
    
    # Store full optimal lambdas for all windows (for final evaluation)
    optimal_lambdas_full = lambdas_df['lambda_opt'].values if len(lambdas_df) == len(window_data) else np.array([f['lambda_opt'] for f in lambdas_df.to_dict('records')])
    
    # Verify lengths match
    if len(optimal_lambdas) != len(vix_levels):
        raise ValueError(f"Length mismatch: vix_levels={len(vix_levels)}, optimal_lambdas={len(optimal_lambdas)}")
    
    # Search for optimal two thresholds using grid search
    percentiles_low = np.linspace(15, 45, 7)
    percentiles_high = np.linspace(55, 85, 7)
    
    best_threshold_low = None
    best_threshold_high = None
    best_frobenius = np.inf
    
    for pct_low in percentiles_low:
        for pct_high in percentiles_high:
            if pct_low >= pct_high:
                continue
                
            threshold_low = np.percentile(vix_levels, pct_low)
            threshold_high = np.percentile(vix_levels, pct_high)
            
            # Assign lambda based on 3 regimes (training only)
            lambda_pred = np.zeros_like(vix_levels)
            for i, vix in enumerate(vix_levels):
                if vix < threshold_low:
                    lambda_pred[i] = optimal_lambdas[vix_levels < threshold_low].mean()
                elif vix < threshold_high:
                    lambda_pred[i] = optimal_lambdas[(vix_levels >= threshold_low) & (vix_levels < threshold_high)].mean()
                else:
                    lambda_pred[i] = optimal_lambdas[vix_levels >= threshold_high].mean()
            
            # Compute Frobenius distances for training windows only
            distances = []
            for idx, w in enumerate(window_data[:len(vix_levels)]):
                S = w['S']
                realized = w['realized']
                
                lam = lambda_pred[idx]
                S_est = (1 - lam) * S + lam * I
                dist = frobenius_distance(S_est, realized)
                distances.append(dist)
            
            mean_dist = np.mean(distances)
            
            if mean_dist < best_frobenius:
                best_frobenius = mean_dist
                best_threshold_low = threshold_low
                best_threshold_high = threshold_high
    
    # Apply best thresholds to ALL windows
    all_vix_levels = np.array([f['vix_level'] for f in vix_features_original])
    lambda_pred_full = np.zeros(len(all_vix_levels))
    
    # Use full optimal lambdas for the means
    full_optimal_lambdas = lambdas_df['lambda_opt'].values
    
    for i, vix in enumerate(all_vix_levels):
        if vix < best_threshold_low:
            lambda_pred_full[i] = full_optimal_lambdas[all_vix_levels < best_threshold_low].mean()
        elif vix < best_threshold_high:
            lambda_pred_full[i] = full_optimal_lambdas[(all_vix_levels >= best_threshold_low) & (all_vix_levels < best_threshold_high)].mean()
        else:
            lambda_pred_full[i] = full_optimal_lambdas[all_vix_levels >= best_threshold_high].mean()
    
    distances = []
    for idx, w in enumerate(window_data):
        S = w['S']
        realized = w['realized']
        
        lam = lambda_pred_full[idx]
        S_est = (1 - lam) * S + lam * I
        dist = frobenius_distance(S_est, realized)
        distances.append(dist)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    metrics = {
        'mean_frobenius': mean_dist,
        'std_frobenius': std_dist,
        'threshold_low': best_threshold_low,
        'threshold_high': best_threshold_high,
        'n_windows': len(window_data)
    }
    
    print("\n=== VIX Threshold Baseline (3-Regime) ===")
    print(f"Optimal VIX thresholds: Low = {best_threshold_low:.2f}, High = {best_threshold_high:.2f}")
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
    # Handle empty data
    if not window_data or len(window_data) == 0:
        print("WARNING: No window data available. Returning empty rolling metrics.")
        return {
            'mean_frobenius': np.nan,
            'std_frobenius': np.nan,
            'rolling_window': window_size,
            'n_windows': 0
        }
    
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