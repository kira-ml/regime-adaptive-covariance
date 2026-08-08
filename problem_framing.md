# Regime-Adaptive Covariance Estimation

## Problem Framing Document

---

## 1. Project Title

**Regime-Adaptive Covariance Estimation: Testing Whether Market Regime Information Improves Out-of-Sample Portfolio Risk Estimates**

---

## 2. Problem Overview

### The Real-World Problem

Portfolio construction and risk management depend critically on accurate covariance estimates. When markets transition between regimes—for example, from low-volatility to crisis-mode, or from stable correlations to chaotic flight-to-safety—standard covariance estimators react too slowly. The result is that risk models underestimate true portfolio volatility during stress periods, potentially leading to excessive leverage, concentration risk, and drawdowns.

Conversely, if the estimator reacts too quickly to short-term noise, it generates false signals, prompting unnecessary rebalancing, overtrading, and poor risk-adjusted returns.

The core challenge is that conventional approaches apply a single, static shrinkage target (e.g., identity matrix, factor model, or historical average) to all periods. This ignores the fact that optimal shrinkage targets differ across market environments.

### Why This Matters

This problem sits at the intersection of portfolio management, risk modeling, and empirical asset pricing. An improved covariance estimate can:

- Reduce portfolio volatility and drawdown risk
- Improve mean-variance optimization outcomes
- Decrease turnover and transaction costs
- Provide more accurate Value-at-Risk (VaR) and Expected Shortfall (ES) estimates
- Support better capital allocation decisions

The project supports risk managers, portfolio managers, and quantitative analysts seeking to improve the robustness of their risk models without overcomplicating their existing frameworks.

---

## 3. Problem Framing

### Machine Learning Task

**Task Type:** Supervised learning for covariance estimation improvement

**Unit of Observation:** Rolling estimation window of daily returns for a fixed set of assets

**Target Variable:** The ex-post realized covariance matrix for a future horizon (e.g., next 20 trading days)

**Input Variables:** For each estimation window:

- Historical return series (e.g., daily returns over past 60, 120, or 250 trading days)
- Regime classification variables derived from market features:
  - Current market volatility level (VIX or realized volatility)
  - Cross-sectional dispersion of returns
  - Average pairwise correlation
  - Market return trend
  - Volatility term structure (VIX futures curve shape)
  - Credit spreads or other risk indicators
- Features describing the covariance structure:
  - Eigenvalue spectrum shape
  - Condition number of the sample covariance
  - Average correlation level

**Prediction Horizon:** 20 trading days (~1 month) ahead, chosen to match typical portfolio rebalancing frequency

**Information Available at Prediction Time:** All features constructed from data available at the end of the estimation window, using only lagged information with no look-ahead bias

**Information Unavailable at Prediction Time:** Future returns, future realized covariance, or any data not yet observed

**Prediction Target:** Rather than predicting the full covariance matrix directly (which is high-dimensional and challenging), the project predicts the **optimal shrinkage intensity** for a given covariance estimator, where optimality is defined as minimizing the Frobenius distance to the realized covariance.

Specifically, for each estimation window `t`, the target is:

```
λ*_t = argmin_λ || (1-λ)*S_t + λ*T_t - Σ_realized,t+20 ||_F^2
```

Where:
- `S_t` = sample covariance matrix from window t
- `T_t` = chosen shrinkage target (e.g., identity, factor model, or regime-specific target)
- `Σ_realized,t+20` = realized covariance over the next 20 trading days
- `||·||_F` = Frobenius norm

### Key Assumptions

1. There exists a systematic relationship between observable market conditions and the optimal shrinkage intensity
2. This relationship is sufficiently stable over the out-of-sample period to be learnable
3. The 20-day horizon is appropriate for portfolio rebalancing decisions
4. The asset universe is reasonably stable (e.g., S&P 500 constituents with sufficient liquidity)
5. Transaction costs and other frictions are not modeled explicitly but can be considered in interpretation

### Primary Task Formulation

**Predict, for each estimation window, the shrinkage intensity λ ∈ [0,1] that minimizes the expected squared Frobenius distance between the shrunk covariance estimate and the future realized covariance.**

This formulation turns a high-dimensional covariance estimation problem into a more tractable scalar prediction problem while directly addressing the economic objective.

---

## 4. Primary Research Question

**Does incorporating market regime information (via features such as volatility, correlation structure, and market stress indicators) improve out-of-sample shrinkage intensity predictions compared to a baseline model that ignores regime dynamics?**

### Secondary Questions

1. Which regime indicators provide the most predictive power for optimal shrinkage intensity?
2. Does a regime-adaptive covariance estimate lead to lower out-of-sample portfolio variance compared to static shrinkage approaches?
3. Is the improvement economically meaningful after accounting for estimation error and regime detection lags?

---

## 5. Hypotheses

### H1: Regime features contain predictive information
**H₁₀:** All regime features have zero predictive power for optimal shrinkage intensity  
**H₁₁:** At least one regime feature has statistically significant predictive power

### H2: Dynamic shrinkage improves over static approaches
**H₂₀:** Dynamic shrinkage does not reduce out-of-sample Frobenius distance compared to the best static shrinkage parameter  
**H₂₁:** Dynamic shrinkage reduces out-of-sample Frobenius distance

### H3: The improvement is economically meaningful
**H₃₀:** Any improvement in covariance estimation does not reduce realized portfolio volatility  
**H₃₁:** Improved covariance estimation reduces realized portfolio volatility by at least 5-10% out-of-sample

---

## 6. Public Data

### Primary Dataset

**Yahoo Finance / yfinance** (Python package)
- Daily OHLCV and adjusted close prices for S&P 500 constituents
- Period: January 2000 – December 2025 (or most recent available)
- Allows construction of daily returns with proper adjustments

### Supplementary Market Features

**VIX Index** (CBOE Volatility Index)
- Available via Yahoo Finance (`^VIX`)
- Used as a proxy for market-wide volatility expectations

**Fama-French Factors** (Ken French Data Library)
- Daily returns for market, size, value, momentum factors
- Available for free download
- Can be used for robustness checks and factor-adjustment

**Treasury Yields** (Federal Reserve Economic Data / FRED)
- 10-year Treasury yield, 3-month T-bill rate
- Available via FRED API or Yahoo Finance
- Used for risk-free rate and term structure features

**Credit Spreads** (Optional, for robustness)
- BAA-AAA spread from FRED
- High-yield spread (if accessible)

### Data Construction

1. Select a universe of 50-100 liquid S&P 500 stocks with continuous history
2. Ensure no survivorship bias by using historical constituent data or fixed set of large-cap stocks
3. Construct daily returns, adjusted for splits and dividends
4. Create rolling estimation windows (e.g., 60, 120, 250 trading days)
5. For each window, compute features and target shrinkage intensity
6. Ensure strict temporal ordering: features at time t use only data up to t, target uses data from t+1 to t+20

---

## 7. Baseline Models

### Baseline 1: Constant Optimal Shrinkage (Statistical Baseline)

**Method:** Simple average of optimal shrinkage intensities across all training windows

**Approach:**
1. Compute the optimal λ for each training window
2. Take the average: λ_const = mean_t(λ*_t)
3. Apply this constant λ to all test windows

**Purpose:** Establishes whether any dynamic approach can beat a static benchmark

**Why appropriate:** This represents the standard practice in many covariance estimation approaches—choose a single shrinkage target and stick with it. It tests whether regime variation is actually exploitable.

**Limitations:**
- Ignores all regime variation
- May be optimal on average but suboptimal in any specific regime
- Cannot adapt to changing market conditions

### Baseline 2: Historical Rule-Based Shrinkage

**Method:** Simple rule that adjusts λ based on a threshold of a single regime indicator (e.g., VIX level)

**Approach:**
1. Choose VIX as the regime indicator (justified by its widespread use as a fear gauge)
2. Use training data to find a single threshold that minimizes average Frobenius distance
3. Apply rule: λ_low = average optimal λ when VIX < threshold, λ_high = average optimal λ when VIX ≥ threshold

**Purpose:** Provides a simple, interpretable dynamic benchmark that could plausibly be implemented without ML

**Why appropriate:** This tests whether a single, simple rule captures most of the regime-dependent variation, which would imply ML may be unnecessary

**Limitations:**
- Only uses one regime indicator
- Threshold-based rules are discontinuous and may cause regime oscillation
- Cannot capture more complex interactions between multiple regime indicators

### Baseline 3: Ledoit-Wolf Static Shrinkage

**Method:** The standard Ledoit-Wolf (2004) shrinkage estimator

**Approach:**
1. Compute the Ledoit-Wolf shrinkage estimator for each estimation window
2. This estimator uses an adaptive but static formula based on the data structure
3. Compare to a regime-adaptive approach

**Purpose:** Represents the industry-standard approach to covariance shrinkage

**Why appropriate:** Widely used in practice; provides a strong, established benchmark

**Limitations:**
- The shrinkage target (identity matrix) is fixed
- The formula, while adaptive to sample properties, does not account for market regimes
- May still react too slowly to regime changes

---

## 8. Advanced Machine Learning Models

### Advanced 1: Gradient Boosting (XGBoost/LightGBM)

**Method:** Tree-based gradient boosting with regularization

**Capability added:** Can capture non-linear relationships and interactions between multiple regime indicators simultaneously

**Baseline limitation addressed:** Baselines cannot capture complex, non-linear relationships between multiple regime indicators and optimal shrinkage

**Why it may help:**
- Handles non-linearities naturally
- Provides feature importance measures
- Robust to outliers and missing data
- Can model interactions between e.g., VIX level and correlation structure
- Well-suited for relatively small datasets (few thousand windows)

**Evidence that would justify using it:**
- Baseline models show systematic prediction errors in specific regimes
- Feature importance analysis on simpler models suggests interactions matter
- Single-indicator threshold models underperform relative to average baseline

### Advanced 2: Penalized Linear Model (Elastic Net)

**Method:** Linear regression with L1 and L2 regularization

**Capability added:** Automated feature selection and regularization

**Baseline limitation addressed:** Simple rules ignore the combined predictive power of multiple regime indicators

**Why it may help:**
- Very interpretable
- Can handle correlated features
- Avoids overfitting
- Provides coefficient estimates that can be interpreted economically
- If this performs as well as gradient boosting, simplicity would favor it

**Evidence that would justify using it:**
- Linear models show predictive power but need regularization
- Simple threshold models underperform, suggesting multiple indicators matter
- The relationship appears approximately linear in transformed features

---

## 9. Experimental Progression

### Stage 1: Data Construction and Validation (Fixed)
1. Download and clean data
2. Construct rolling estimation windows
3. Compute features and targets for each window
4. Verify no look-ahead bias (strict temporal ordering)
5. Split data: 70% training, 30% test (chronological, not random)

### Stage 2: Baseline Models
1. Compute Constant Optimal Shrinkage baseline
2. Compute Historical Rule-Based baseline (optimize threshold on training)
3. Compute Ledoit-Wolf baseline
4. Evaluate all baselines on test set
5. Document performance and identify systematic patterns in prediction errors

**Decision Point:** If dynamic baselines (Rule-Based) do not significantly outperform the constant baseline, there may be little regime-dependent signal. Proceed cautiously and consider feature engineering.

### Stage 3: Simple Dynamic Models
1. Fit Elastic Net on training data (with cross-validation)
2. Evaluate on test set and compare to baselines
3. Analyze feature coefficients to understand which indicators matter

**Decision Point:** If Elastic Net does not meaningfully improve over baselines, the signal may be too weak or non-linear. This would justify either stopping or exploring non-linear methods.

### Stage 4: Non-Linear Models
1. If Elastic Net shows signal but underperforms expectations, fit Gradient Boosting
2. Tune hyperparameters via time-series cross-validation
3. Evaluate on test set and compare to all previous models
4. Analyze feature importance and partial dependence plots

**Decision Point:** If Gradient Boosting does not substantially outperform Elastic Net, the simpler linear model should be preferred. The evidence would suggest that non-linear relationships are not the primary source of predictive power.

### Stage 5: Economic Evaluation
1. Construct minimum-variance portfolios using each covariance estimate
2. Compare realized out-of-sample portfolio volatility and turnover
3. Assess whether differences are economically meaningful (e.g., 5-10% volatility reduction)

### Stage 6: Robustness Checks
1. Sensitivity to estimation window length (60 vs 120 vs 250 days)
2. Sensitivity to prediction horizon (10 vs 20 vs 30 days)
3. Performance in sub-periods (e.g., pre-2008 vs post-2008)
4. Robustness to different asset universes (if time permits)

---

## 10. Evaluation Framework

### Primary Metrics

**Covariance Estimation Accuracy:**
- **Frobenius Distance:** Mean squared Frobenius distance between estimated and realized covariance
  
  ```
  d_F = ||Σ_est - Σ_real||_F
  ```

- **Kullback-Leibler Divergence (for covariance matrices):**
  
  ```
  D_KL = 0.5 * [tr(Σ_real^{-1} Σ_est) - log(det(Σ_real^{-1} Σ_est)) - n]
  ```

**Portfolio Evaluation Metrics:**
- **Realized Volatility:** Standard deviation of daily portfolio returns over test period
- **Turnover:** Average absolute change in portfolio weights across rebalancing dates
- **Sharpe Ratio:** Annualized return / annualized volatility (using realized portfolio returns)
- **Maximum Drawdown:** Largest peak-to-trough decline

**Prediction-Specific Metrics:**
- **RMSE of λ Prediction:** Root mean squared error between predicted and optimal λ
- **MAE of λ Prediction:** Mean absolute error
- **R²:** Coefficient of determination for λ predictions

### Data Splitting Strategy

**Time-Based Split:**
- Training: January 2000 – December 2015 (approximately 15 years)
- Validation: January 2016 – December 2019 (for hyperparameter tuning)
- Test: January 2020 – December 2025 (approximately 5 years)

**Reasoning:**
- Strict chronological order prevents look-ahead bias
- Includes multiple market regimes in each split:
  - Training: Dot-com bubble aftermath, 2008 crisis, 2011 Euro crisis, 2013 taper tantrum
  - Validation: Normal period with small drawdowns
  - Test: COVID-19 pandemic, 2022 bear market, post-pandemic recovery

### Validation Methodology

**Time-Series Cross-Validation:**
For hyperparameter tuning, use rolling windows (e.g., expanding window with 5-year test periods):
1. Train: Jan 2000–Dec 2006, Test: Jan 2007–Dec 2009
2. Train: Jan 2000–Dec 2009, Test: Jan 2010–Dec 2012
3. Train: Jan 2000–Dec 2012, Test: Jan 2013–Dec 2015

This simulates how the model would be used in practice and provides multiple test periods.

### Leakage Controls

1. **Feature Construction:** Features must use only data from the estimation window (ending at time t)
2. **Target Construction:** Target uses data from t+1 to t+20 only
3. **Splitting:** No randomization; strict temporal ordering
4. **Standardization:** Means and standard deviations computed on training data only and applied to test data
5. **Asset Selection:** Use a fixed set of stocks or carefully handle changes in S&P 500 constituents to avoid survivorship bias

### Statistical Significance

1. **Diebold-Mariano Test:** For comparing forecast accuracy between models
2. **Bootstrapped Confidence Intervals:** For portfolio performance metrics
3. **Sub-period Analysis:** Test whether improvements are consistent across different market conditions
4. **Rolling Performance:** Plot cumulative differences to assess stability

### Finance-Specific Considerations

1. **Transaction Costs:** Consider a simple cost model (e.g., 10-20 bps per trade) when evaluating turnover
2. **Impact of Estimation Error:** Evaluate whether estimation improvement translates to lower tracking error variance
3. **Implementation Lags:** Account for the fact that rebalancing takes time and prices may move
4. **Regime Detection Delay:** Evaluate whether regime changes are detected quickly enough to matter

---

## 11. Expected Failure Modes and Risks

### Methodological Risks

**1. Look-Ahead Bias**
- **Risk:** Accidentally using future information in feature construction
- **Mitigation:** Strict temporal separation; code review; manual verification of window alignment

**2. Overfitting to In-Sample Regimes**
- **Risk:** The model learns patterns specific to training period regimes
- **Mitigation:** Include multiple regimes in training; use regularization; test on post-2020 data

**3. Non-Stationarity**
- **Risk:** The relationship between regime indicators and optimal shrinkage changes over time
- **Mitigation:** Evaluate sub-periods separately; consider adaptive models; report temporal stability

**4. Temporal Dependence**
- **Risk:** Overlapping estimation windows create dependent observations
- **Mitigation:** Use time-series cross-validation; consider block bootstrapping; be cautious with standard errors

**5. Feature Selection Bias**
- **Risk:** Selecting features based on test performance
- **Mitigation:** Use only training data for feature selection; use regularization for automatic selection

**6. Survivorship Bias**
- **Risk:** Using only currently existing stocks
- **Mitigation:** Use a fixed set of highly liquid stocks; explicitly use historical constituents (if feasible)

**7. Transaction Cost Neglect**
- **Risk:** Ignoring that different covariance estimates imply different turnover
- **Mitigation:** Include turnover in evaluation; consider transaction costs in portfolio construction

**8. Regime Change Detection Lags**
- **Risk:** Regime indicators may detect changes too late
- **Mitigation:** Use multiple regime indicators with different lookback periods; evaluate early warning properties

**9. Estimation of Optimal λ**
- **Risk:** The target λ* itself is noisy due to estimation error
- **Mitigation:** Use robust estimation; consider alternative target definitions; average over multiple windows

**10. Limited Data**
- **Risk:** Only 15-20 years of daily data (~2500-4000 windows) limits statistical power
- **Mitigation:** Use cross-validation; prefer simpler models; be transparent about limitations

**11. Model Mis-Specification**
- **Risk:** The functional form of the model is wrong
- **Mitigation:** Use flexible models (gradient boosting); but also test simpler models; favor parsimony

**12. Multiple Testing**
- **Risk:** Testing many models increases chance of false positives
- **Mitigation:** Pre-register main analysis; limit number of models; use validation set for selection

---

## 12. Success Criteria

### Meaningful Baseline
A baseline is meaningful if it:
- Represents a plausible, implementable approach
- Has a clear theoretical or practical justification
- Provides a reference point for evaluating more complex models

### Evidence of Useful Predictive Signal
Evidence that regime features contain useful information if:
- At least one advanced model significantly outperforms the Constant Optimal Shrinkage baseline on the test set (p < 0.05, Diebold-Mariano test)
- The improvement in RMSE is at least 5-10% (sufficient to be economically meaningful)
- The improvement is consistent across multiple sub-periods

### Meaningful Improvement
An improvement is meaningful if:
- It is statistically significant (p < 0.05)
- It is practically significant (e.g., >5% reduction in Frobenius distance)
- It translates to portfolio volatility reduction of >5% out-of-sample
- The improvement does not come at the cost of significantly higher turnover

### Justified Conclusion
A conclusion is justified if:
- The experimental design rules out alternative explanations
- The results are robust to sensitivity analyses
- The chosen model complexity is supported by evidence
- Limitations and failure modes are clearly stated
- The conclusion acknowledges uncertainty and the need for further validation

### Preferred Outcome
The preferred outcome is **not** that the most complex model wins. The preferred outcome is that we have robust empirical evidence about **whether** and **when** regime-adaptive covariance estimation provides meaningful benefits, and that the chosen model is the simplest one that performs well.

---

## 13. Final Project Scope

### What Should Be Implemented in Python

**Data Pipeline:**
- Data download from Yahoo Finance using yfinance
- Data cleaning and quality checks
- Rolling window construction
- Feature engineering (VIX, correlations, volatilities, eigenvalue features)
- Target computation (optimal λ for each window)

**Model Implementation:**
1. Baseline models (constant shrinkage, rule-based, Ledoit-Wolf)
2. Linear models (Elastic Net with cross-validation)
3. Gradient boosting (XGBoost or LightGBM with hyperparameter tuning via Optuna)
4. Time-series cross-validation implementation

**Evaluation Framework:**
- Metric computation (Frobenius distance, portfolio volatility, turnover)
- Statistical tests (Diebold-Mariano, bootstrapping)
- Visualization (performance over time, feature importance)
- Sensitivity analyses

**Portfolio Construction:**
- Minimum-variance portfolio optimization using each covariance estimate
- Portfolio performance calculation (volatility, Sharpe ratio, turnover)
- Risk contribution analysis

### What Should NOT Be Built

**DO NOT include:**
- Production-grade MLOps infrastructure
- Docker containers or Kubernetes deployment
- CI/CD pipelines
- Real-time data streaming
- Database systems or API servers
- Web applications or dashboards (unless a simple Jupyter notebook visualization)
- Deep learning models (LSTMs, neural networks) - the dataset is too small and the problem does not require it
- High-frequency trading infrastructure
- Distributed computing (Spark, Dask clusters)
- Complex factor models beyond standard Fama-French
- Transaction-cost optimization models beyond a simple cost adjustment
- Cross-asset class modeling (stay within equities)

**DO NOT include unless absolutely necessary:**
- GPU-accelerated training (not needed for this dataset size)
- Ensemble methods beyond a single gradient boosting model
- Bayesian optimization (use simpler hyperparameter tuning)

### Project Structure (Recommended)

```
project/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_download_and_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   ├── 05_portfolio_evaluation.ipynb
│   └── 06_results_and_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── covariance_estimators.py
│   ├── portfolio_optimizer.py
│   └── evaluation_metrics.py
├── tests/
│   └── (unit tests for critical functions)
├── results/
│   ├── figures/
│   └── tables/
└── paper/
    └── (write-up in markdown or Jupyter book format)
```

### Time and Effort Estimate

- **Data acquisition and cleaning:** 1-2 days
- **Feature engineering and target construction:** 2-3 days
- **Baseline implementation:** 1-2 days
- **Advanced models:** 2-3 days
- **Evaluation and portfolio analysis:** 2-3 days
- **Write-up and documentation:** 3-5 days
- **Total:** Approximately 2-3 weeks of full-time effort

This scope is realistic for a portfolio project while demonstrating strong problem-framing, careful methodology, and thoughtful evaluation.

---

## Summary

This project transforms the "Regime-Change Prior Problem" into a well-framed machine learning project by:

1. **Turning a high-dimensional problem into a scalar prediction task:** Instead of predicting the full covariance matrix, we predict the optimal shrinkage intensity—a more tractable and economically meaningful target.

2. **Using a baseline-first approach:** Establishing meaningful benchmarks (constant shrinkage, rule-based, Ledoit-Wolf) before introducing ML.

3. **Increasing complexity only when justified:** Starting with simple linear models and only moving to gradient boosting if linear models underperform.

4. **Evaluating economic relevance:** Moving beyond statistical metrics to portfolio-level evaluation with realistic considerations (volatility, turnover, transaction costs).

5. **Rigorously controlling for common finance pitfalls:** Strict temporal ordering, no look-ahead bias, time-series cross-validation, multiple sub-period analysis.

6. **Maintaining a skeptical, evidence-driven tone:** No claims of certainty; hypotheses are testable and falsifiable; conclusions depend on empirical results.

The project demonstrates strong ML thinking—not by using the most sophisticated models, but by carefully defining the problem, constructing appropriate baselines, testing hypotheses rigorously, and being transparent about limitations and potential failure modes.

---

## References

1. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

2. Ledoit, O., & Wolf, M. (2012). Nonlinear shrinkage estimation of large-dimensional covariance matrices. *Annals of Statistics*, 40(2), 1024-1060.

3. De Nard, G., Ledoit, O., & Wolf, M. (2021). Factor models for portfolio selection in large dimensions: The good, the better, and the ugly. *Journal of Financial Econometrics*, 19(2), 241-270.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

5. Guidolin, M., & Timmermann, A. (2008). International asset allocation under regime switching, skew, and kurtosis preferences. *Review of Financial Studies*, 21(2), 889-935.

6. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

7. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

8. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

9. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

10. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.

---

## Appendix: Mathematical Notation Reference

| Symbol | Description |
|--------|-------------|
| `S_t` | Sample covariance matrix at time t |
| `T_t` | Shrinkage target matrix at time t |
| `λ` | Shrinkage intensity (scalar in [0,1]) |
| `Σ_realized,t+20` | Realized covariance over next 20 days |
| `||·||_F` | Frobenius norm |
| `λ*_t` | Optimal shrinkage intensity for window t |
| `d_F` | Frobenius distance between covariance matrices |
| `D_KL` | Kullback-Leibler divergence between covariance matrices |
| `VIX_t` | CBOE Volatility Index at time t |
| `n` | Number of assets in portfolio |

---

*This problem framing document serves as the foundation for the project implementation. All decisions about model selection, evaluation, and interpretation should be guided by the principles and constraints outlined above.*