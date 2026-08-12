"""Feature engineering for regime-adaptive covariance estimation.

This module defines baseline and advanced feature sets for predicting
optimal shrinkage intensity λ. All features are computed from the
regime features already generated in rolling_windows.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# Feature set definitions
# ----------------------------------------------------------------------

BASELINE_SETS = {
    'VIX-Only': ['vix_level'],
    'Vol+Corr': ['vix_level', 'realized_vol', 'avg_correlation'],
    'Market': [
        'vix_level',
        'realized_vol',
        'avg_correlation',
        'cross_sectional_dispersion',
        'vix_percentile',
        'max_drawdown'
    ]
}

ADVANCED_SETS = {
    'Covariance': [
        'condition_number',
        'trace',
        'avg_eigenvalue',
        'avg_correlation'
    ],
    'VIX+Interaction': [
        'vix_level',
        'vix_x_realized_vol'
    ],
    'VIX+Rolling': [
        'vix_level',
        'vix_rolling_60'
    ],
    'All': [
        'vix_level',
        'vix_percentile',
        'realized_vol',
        'avg_correlation',
        'cross_sectional_dispersion',
        'market_return',
        'max_drawdown',
        'condition_number',
        'trace',
        'avg_eigenvalue',
        # --- New features ---
        'vix_x_realized_vol',
        'vix_level_lag5',
        'vix_rolling_60'
    ]
}
# ----------------------------------------------------------------------
# Feature engineering class
# ----------------------------------------------------------------------

class FeatureEngineer:
    """
    Manages feature set selection, scaling, and evaluation.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing all regime features (from regime_features.csv).
    target_df : pd.DataFrame
        DataFrame containing optimal lambdas (from optimal_lambdas.csv).
    train_indices : list of int
        Indices of training windows (for fitting scaler).
    val_indices : list of int
        Indices of validation windows.
    test_indices : list of int
        Indices of test windows.
    """

    def __init__(self, features_df, target_df, train_indices, val_indices, test_indices):
        self.features_df = features_df.copy()
        self.target_df = target_df.copy()
        self.train_idx = train_indices
        self.val_idx = val_indices
        self.test_idx = test_indices

        # Merge features and target on window_id
        self.df = pd.merge(self.features_df, self.target_df, on='window_id')

        # Ensure we have only the windows that exist in all splits
        all_idx = set(train_indices) | set(val_indices) | set(test_indices)
        self.df = self.df[self.df['window_id'].isin(all_idx)]

        self.scaler = None

    def get_X_y(self, feature_cols, indices):
        """
        Extract feature matrix X and target vector y for given indices.

        Parameters
        ----------
        feature_cols : list of str
            Column names for features.
        indices : list of int
            Window indices to include.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        subset = self.df[self.df['window_id'].isin(indices)]
        X = subset[feature_cols].values
        y = subset['lambda_opt'].values
        return X, y

    def fit_scaler(self, feature_cols):
        """
        Fit StandardScaler on training data for a given feature set.

        Parameters
        ----------
        feature_cols : list of str
            Column names for features.
        """
        X_train, _ = self.get_X_y(feature_cols, self.train_idx)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

    def transform(self, X):
        """Apply fitted scaler to X."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        return self.scaler.transform(X)

    def prepare_dataset(self, feature_cols, indices, scale=True):
        """
        Get X and y for given indices, optionally scaled.

        Parameters
        ----------
        feature_cols : list of str
            Column names for features.
        indices : list of int
            Window indices to include.
        scale : bool
            Whether to scale features (requires scaler to be fitted).

        Returns
        -------
        X : np.ndarray
        y : np.ndarray
        """
        X, y = self.get_X_y(feature_cols, indices)
        if scale and self.scaler is not None:
            X = self.transform(X)
        return X, y

    def evaluate_feature_set(self, feature_cols, name):
        """
        Evaluate a feature set on test data using RMSE and R².

        Parameters
        ----------
        feature_cols : list of str
            Column names for features.
        name : str
            Name of the feature set (for reporting).

        Returns
        -------
        dict
            {'rmse': float, 'r2': float, 'n_features': int}
        """
        # Fit scaler on training set
        self.fit_scaler(feature_cols)

        # Get test data (scaled)
        X_test, y_test = self.prepare_dataset(feature_cols, self.test_idx, scale=True)

        if len(X_test) == 0:
            return {'rmse': np.nan, 'r2': np.nan, 'n_features': len(feature_cols)}

        # Simple linear regression for quick evaluation
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.prepare_dataset(feature_cols, self.train_idx, scale=True)[0],
                  self.prepare_dataset(feature_cols, self.train_idx, scale=True)[1])

        y_pred = model.predict(X_test)

        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_raw = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        r2 = max(0.0, r2_raw)  # Floor at 0 for interpretability

        return {'rmse': rmse, 'r2': r2, 'n_features': len(feature_cols)}

    def compare_all_sets(self):
        """
        Evaluate all baseline and advanced feature sets.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns:
            set_name, n_features, rmse, r2
        """
        results = []

        # Baseline sets
        for name, cols in BASELINE_SETS.items():
            # Only include columns that actually exist
            available_cols = [c for c in cols if c in self.df.columns]
            if not available_cols:
                continue
            metrics = self.evaluate_feature_set(available_cols, name)
            metrics['set_name'] = name
            results.append(metrics)

        # Advanced sets
        for name, cols in ADVANCED_SETS.items():
            available_cols = [c for c in cols if c in self.df.columns]
            if not available_cols:
                continue
            metrics = self.evaluate_feature_set(available_cols, name)
            metrics['set_name'] = name
            results.append(metrics)

        df_results = pd.DataFrame(results)
        # Reorder columns
        df_results = df_results[['set_name', 'n_features', 'rmse', 'r2']]
        return df_results

    def get_best_set(self, metric='rmse'):
        """
        Return the name of the best-performing feature set.

        Parameters
        ----------
        metric : str, 'rmse' or 'r2'
            Metric to optimize.

        Returns
        -------
        str
            Name of the best feature set.
        """
        comparison = self.compare_all_sets()
        if metric == 'rmse':
            best_idx = comparison['rmse'].idxmin()
        elif metric == 'r2':
            best_idx = comparison['r2'].idxmax()
        else:
            raise ValueError("metric must be 'rmse' or 'r2'")
        return comparison.loc[best_idx, 'set_name']


# ----------------------------------------------------------------------
# Convenience function for standalone usage
# ----------------------------------------------------------------------

def run_feature_engineering(features_path, lambdas_path, train_indices, val_indices, test_indices):
    """
    Load data, run feature set comparison, and return results.

    Parameters
    ----------
    features_path : str
        Path to regime_features.csv.
    lambdas_path : str
        Path to optimal_lambdas.csv.
    train_indices, val_indices, test_indices : list of int
        Window indices for each split.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    FeatureEngineer
        Fitted engineer object (for further use).
    """
    features = pd.read_csv(features_path)
    lambdas = pd.read_csv(lambdas_path)

    engineer = FeatureEngineer(features, lambdas, train_indices, val_indices, test_indices)
    comparison = engineer.compare_all_sets()

    print("\n=== Feature Set Comparison ===")
    print(comparison.to_string(index=False))

    best = engineer.get_best_set('rmse')
    print(f"\n🏆 Best feature set (by RMSE): {best}")

    return comparison, engineer