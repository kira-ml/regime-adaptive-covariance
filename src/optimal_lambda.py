"""Find optimal shrinkage intensity lambda for each window"""

import numpy as np
from scipy.linalg import norm


def frobenius_distance(A, B):
    """
    Compute Frobenius distance between two matrices.
    
    Parameters:
    - A, B: numpy arrays of same shape
    
    Returns:
    - float: Frobenius norm of (A - B)
    """
    return norm(A - B, 'fro')


def find_optimal_lambda(S, realized, lambda_grid):
    """
    Find the lambda that minimizes Frobenius distance to realized covariance.
    
    Parameters:
    - S: sample covariance matrix (numpy array)
    - realized: realized covariance matrix (numpy array)
    - lambda_grid: array of lambda values to search over, e.g., np.linspace(0, 1, 21)
    
    Returns:
    - optimal_lambda: float, best lambda value
    - min_distance: float, minimum Frobenius distance achieved
    """
    n = S.shape[0]
    I = np.eye(n)  # Identity matrix as target
    
    best_lambda = lambda_grid[0]
    min_dist = np.inf
    
    for lam in lambda_grid:
        # Shrunk covariance estimate: (1-lam)*S + lam*I
        S_est = (1 - lam) * S + lam * I
        
        # Distance to realized covariance
        dist = frobenius_distance(S_est, realized)
        
        if dist < min_dist:
            min_dist = dist
            best_lambda = lam
    
    return best_lambda, min_dist


def compute_optimal_lambdas(window_data, lambda_grid):
    """
    Process all windows and find optimal lambda for each.
    
    Parameters:
    - window_data: list of dicts from compute_covariance_matrices()
    - lambda_grid: array of lambda values to search over
    
    Returns:
    - pd.DataFrame with columns: window_id, lambda_opt, min_frobenius
    """
    import pandas as pd
    
    # Handle empty window_data
    if not window_data or len(window_data) == 0:
        print("WARNING: No window data available. Returning empty DataFrame.")
        return pd.DataFrame(columns=['window_id', 'lambda_opt', 'min_frobenius', 
                                      'train_start', 'train_end', 'test_start', 'test_end'])
    
    results = []
    
    for idx, w in enumerate(window_data):
        S = w['S']
        realized = w['realized']
        
        lam_opt, min_dist = find_optimal_lambda(S, realized, lambda_grid)
        
        results.append({
            'window_id': idx,
            'lambda_opt': lam_opt,
            'min_frobenius': min_dist,
            'train_start': w['train_dates'][0],
            'train_end': w['train_dates'][1],
            'test_start': w['test_dates'][0],
            'test_end': w['test_dates'][1]
        })
    
    df = pd.DataFrame(results)
    print(f"Optimal lambdas computed for {len(df)} windows")
    
    # Only print stats if we have data
    if len(df) > 0:
        print(f"Lambda range: {df['lambda_opt'].min():.3f} to {df['lambda_opt'].max():.3f}")
        print(f"Mean lambda: {df['lambda_opt'].mean():.3f}")
    
    return df