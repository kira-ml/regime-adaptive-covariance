"""XGBoost model for regime-adaptive covariance estimation."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from src.baselines import frobenius_distance


class XGBoostRegime:
    """
    XGBoost model that predicts optimal shrinkage intensity λ
    using a selected feature set.
    """

    def __init__(self, feature_cols, param_grid=None):
        """
        Parameters
        ----------
        feature_cols : list of str
            Column names for features (e.g., ['vix_level']).
        param_grid : dict, optional
            Hyperparameter grid for tuning.
        """
        self.feature_cols = feature_cols
        self.param_grid = param_grid or {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }
        self.scaler = StandardScaler()
        self.model = None
        self.best_params_ = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit XGBoost with hyperparameter tuning on validation set.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets (optimal λ).
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation targets.
        """
        # Scale features using training statistics
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Grid search over hyperparameters
        best_score = np.inf
        best_model = None
        best_params = None

        for max_depth in self.param_grid['max_depth']:
            for learning_rate in self.param_grid['learning_rate']:
                for n_estimators in self.param_grid['n_estimators']:
                    for subsample in self.param_grid['subsample']:
                        model = xgb.XGBRegressor(
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            subsample=subsample,
                            random_state=42,
                            verbosity=0
                        )
                        model.fit(X_train_scaled, y_train)
                        y_pred_val = model.predict(X_val_scaled)
                        rmse_val = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
                        if rmse_val < best_score:
                            best_score = rmse_val
                            best_model = model
                            best_params = {
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'n_estimators': n_estimators,
                                'subsample': subsample
                            }

        self.model = best_model
        self.best_params_ = best_params

    def predict(self, X):
        """Predict λ for given features."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set.

        Returns
        -------
        dict
            {'rmse': float, 'r2': float}
        """
        y_pred = self.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        return {'rmse': rmse, 'r2': r2}


def evaluate_xgboost_on_covariance(window_data, lambdas_df, feature_cols, train_idx, val_idx, test_idx, features_df):
    """
    Train XGBoost and evaluate covariance estimation performance.

    Parameters
    ----------
    window_data : list of dict
        From compute_covariance_matrices().
    lambdas_df : pd.DataFrame
        Optimal lambdas.
    feature_cols : list of str
        Feature column names.
    train_idx, val_idx, test_idx : list of int
        Window indices for each split.
    features_df : pd.DataFrame
        Regime features (with window_id).

    Returns
    -------
    dict
        {'rmse': float, 'r2': float, 'frobenius': float, 'lambda_pred': np.ndarray}
    """
    # Prepare feature matrix and target
    df = pd.merge(features_df, lambdas_df, on='window_id')

    X_train = df[df['window_id'].isin(train_idx)][feature_cols].values
    y_train = df[df['window_id'].isin(train_idx)]['lambda_opt'].values

    X_val = df[df['window_id'].isin(val_idx)][feature_cols].values
    y_val = df[df['window_id'].isin(val_idx)]['lambda_opt'].values

    X_test = df[df['window_id'].isin(test_idx)][feature_cols].values
    y_test = df[df['window_id'].isin(test_idx)]['lambda_opt'].values

    # Train XGBoost
    model = XGBoostRegime(feature_cols)
    model.fit(X_train, y_train, X_val, y_val)

    # Predict on test set
    y_pred = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)

    # Compute Frobenius distances for test set
    frob_dists = []
    for idx, w in enumerate(window_data):
            if idx not in test_idx:
                continue
            S = w['S']
            realized = w['realized']
            n = S.shape[0]
            I = np.eye(n)
            lam = y_pred[list(test_idx).index(idx)] if idx in test_idx else 0.0
            lam = np.clip(lam, 0.0, 1.0)  # <-- ADD THIS LINE
            S_est = (1 - lam) * S + lam * I
            frob_dists.append(frobenius_distance(S_est, realized))

    metrics['mean_frobenius'] = np.mean(frob_dists)
    metrics['std_frobenius'] = np.std(frob_dists)
    metrics['best_params'] = model.best_params_
    metrics['lambda_pred'] = y_pred

    return metrics