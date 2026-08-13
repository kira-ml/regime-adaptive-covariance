"""Statistical tests for regime-adaptive covariance estimation."""

import numpy as np
from scipy import stats


def diebold_mariano_test(e1, e2, h=1, one_sided=False):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    e1 : array-like
        Forecast errors from model 1 (e.g., Frobenius distances).
    e2 : array-like
        Forecast errors from model 2.
    h : int
        Forecast horizon (for autocorrelation adjustment). Default 1.
    one_sided : bool
        If True, test H0: e1 <= e2 (model 1 is not worse). Default False.

    Returns
    -------
    dict
        {'statistic': float, 'p_value': float, 'h0': str}
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    if len(e1) != len(e2):
        raise ValueError("e1 and e2 must have same length.")

    n = len(e1)
    d = e1 - e2  # loss differential
    d_bar = np.mean(d)

    # Compute autocovariance up to lag h-1
    if h > 1:
        # Simple Newey-West type correction
        gamma = np.zeros(h)
        for lag in range(h):
            gamma[lag] = np.mean(d[:(n-lag)] * d[lag:])
        var_d = gamma[0] + 2 * np.sum(gamma[1:])
    else:
        var_d = np.var(d, ddof=1)

    if var_d <= 0:
        # If variance is zero, no evidence of difference
        return {'statistic': 0.0, 'p_value': 1.0, 'h0': 'Equal predictive accuracy'}

    dm_stat = d_bar / np.sqrt(var_d / n)

    # Degrees of freedom (n-1)
    df = n - 1

    if one_sided:
        # H0: e1 <= e2 (model 1 not worse)
        p_value = stats.t.cdf(dm_stat, df)
        h0 = "Model 1 error <= Model 2 error"
    else:
        # Two-sided: H0: e1 == e2
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df))
        h0 = "Equal predictive accuracy"

    return {
        'statistic': dm_stat,
        'p_value': p_value,
        'h0': h0
    }


def bootstrap_volatility_difference(vol1, vol2, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap confidence interval for mean volatility difference (vol1 - vol2).

    Parameters
    ----------
    vol1 : array-like
        Volatility values from method 1.
    vol2 : array-like
        Volatility values from method 2.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).

    Returns
    -------
    dict
        {'mean_diff': float, 'ci_lower': float, 'ci_upper': float, 'p_value': float}
    """
    vol1 = np.asarray(vol1)
    vol2 = np.asarray(vol2)

    if len(vol1) != len(vol2):
        raise ValueError("vol1 and vol2 must have same length.")

    n = len(vol1)
    mean_diff = np.mean(vol1) - np.mean(vol2)

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_diff = np.mean(vol1[idx]) - np.mean(vol2[idx])
        diffs.append(boot_diff)

    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 100 * (alpha / 2))
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    # p-value: proportion of bootstrap samples where diff >= 0 (one-sided)
    # H0: vol1 <= vol2 (method 1 not better)
    p_value = (diffs >= 0).mean()

    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value
    }


def run_statistical_tests(window_data, lambdas_df, baseline_metrics, lw_metrics, test_idx):
    """
    Run Diebold-Mariano and bootstrap tests on test set.

    Parameters
    ----------
    window_data : list of dict
        From compute_covariance_matrices().
    lambdas_df : pd.DataFrame
        Optimal lambdas.
    baseline_metrics : dict
        From evaluate_constant_baseline().
    lw_metrics : dict
        From ledoit_wolf_baseline().
    test_idx : list of int
        Test window indices.

    Returns
    -------
    dict
        Results of statistical tests.
    """
    # Extract Frobenius distances for test set
    frob_constant = []
    frob_lw = []
    frob_optimal = []
    volatilities = {
        'Constant': [],
        'Ledoit-Wolf': [],
        'Optimal': []
    }

    n = window_data[0]['S'].shape[0]
    I = np.eye(n)

    for idx in test_idx:
        w = window_data[idx]
        S = w['S']
        realized = w['realized']

        # Constant
        lambda_const = baseline_metrics['lambda_const']
        S_const = (1 - lambda_const) * S + lambda_const * I
        frob_constant.append(np.linalg.norm(S_const - realized, 'fro'))

        # Ledoit-Wolf (use precomputed from lw_metrics if available, else recompute)
        from sklearn.covariance import LedoitWolf
        if 'train_returns' in w and w['train_returns'] is not None:
            lw = LedoitWolf()
            lw.fit(w['train_returns'].values)
            S_lw = lw.covariance_
        else:
            S_lw = S
        frob_lw.append(np.linalg.norm(S_lw - realized, 'fro'))

        # Optimal
        lambda_opt = lambdas_df.loc[lambdas_df['window_id'] == idx, 'lambda_opt'].values[0]
        S_opt = (1 - lambda_opt) * S + lambda_opt * I
        frob_optimal.append(np.linalg.norm(S_opt - realized, 'fro'))

        # Portfolio volatilities (already computed in portfolio.py, but we recompute here)
        from src.portfolio import minimum_variance_portfolio, portfolio_volatility

        for method_name, S_est in [
            ('Constant', S_const),
            ('Ledoit-Wolf', S_lw),
            ('Optimal', S_opt)
        ]:
            weights = minimum_variance_portfolio(S_est)
            vol = portfolio_volatility(weights, realized)
            volatilities[method_name].append(vol)

    # Convert to arrays
    frob_constant = np.array(frob_constant)
    frob_lw = np.array(frob_lw)
    frob_optimal = np.array(frob_optimal)

    vol_const = np.array(volatilities['Constant'])
    vol_lw = np.array(volatilities['Ledoit-Wolf'])
    vol_opt = np.array(volatilities['Optimal'])

    # --- Diebold-Mariano tests (Frobenius) ---
    dm_lw_vs_const = diebold_mariano_test(frob_lw, frob_constant, h=10)
    dm_opt_vs_const = diebold_mariano_test(frob_optimal, frob_constant, h=10)

    # --- Bootstrap tests (Volatility) ---
    boot_lw_vs_const = bootstrap_volatility_difference(vol_lw, vol_const)
    boot_opt_vs_const = bootstrap_volatility_difference(vol_opt, vol_const)

    return {
        'diebold_mariano': {
            'Ledoit-Wolf vs Constant': dm_lw_vs_const,
            'Optimal vs Constant': dm_opt_vs_const
        },
        'bootstrap': {
            'Ledoit-Wolf vs Constant': boot_lw_vs_const,
            'Optimal vs Constant': boot_opt_vs_const
        },
        'summary': {
            'mean_frobenius_constant': np.mean(frob_constant),
            'mean_frobenius_lw': np.mean(frob_lw),
            'mean_frobenius_optimal': np.mean(frob_optimal),
            'mean_volatility_constant': np.mean(vol_const),
            'mean_volatility_lw': np.mean(vol_lw),
            'mean_volatility_optimal': np.mean(vol_opt)
        }
    }