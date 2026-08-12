# Regime-Adaptive Covariance Shrinkage: A Supervised Learning Approach to Dynamic Portfolio Risk Estimation

---

## 1. Project Title

**Regime-Adaptive Covariance Shrinkage: A Supervised Learning Approach to Dynamic Portfolio Risk Estimation**

---

## 2. Problem Overview

### The Core Problem

Covariance matrix estimation is fundamental to portfolio construction and risk management. Standard practice uses shrinkage estimators (Ledoit-Wolf, 2004) that apply a single, data-driven shrinkage intensity to all periods. This assumes that the optimal shrinkage intensity is constant or varies only with sample properties (dimension, sample size).

**This is almost certainly false.** Markets exhibit distinct regimes—low volatility, high volatility, crisis, recovery—and the optimal trade-off between sample covariance and a structured target likely varies across these regimes. A static estimator will be systematically wrong in certain market conditions.

### Why This Is a Machine Learning Problem

The relationship between market conditions and optimal shrinkage is:
- **Complex**: Multiple features (volatility, correlation structure, market stress) likely interact.
- **Non-linear**: The effect of VIX on optimal \(\lambda\) is probably not linear.
- **Unknown**: No closed-form solution exists for regime-dependent optimal \(\lambda\).
- **Predictable**: Market conditions are observable and contain forward-looking information.

This makes it a natural candidate for supervised learning—predict the optimal \(\lambda\) from observable market features.

### What This Project Does Not Claim

- This is **not** a new covariance estimator.
- This is **not** a production-ready risk system.
- This is **not** guaranteed to outperform existing methods in all conditions.
- This is **not** financial advice.

### What This Project Actually Does

**Empirically tests whether market regime information improves out-of-sample covariance shrinkage predictions compared to static approaches.** It does so by framing the problem as a supervised regression task, testing multiple feature sets, and evaluating both covariance accuracy and portfolio-level economic impact.

---

## 3. Problem Framing (Machine Learning Formulation)

### Task Type
**Supervised regression** (predicting a continuous target: \(\lambda \in [0,1]\)).

### Unit of Observation
A rolling estimation window of \(T = 120\) trading days ending at time \(t\).

### Target Variable (\(y\))
For each window \(t\), the target is the optimal shrinkage intensity:

$$
\lambda^*_t = \underset{\lambda \in [0,1]}{\text{argmin}} \left\| (1-\lambda) S_t + \lambda I - \Sigma_{t+1:t+H} \right\|_F^2
$$

Where:
- \(S_t\): Sample covariance matrix from window ending at \(t\).
- \(I\): Identity matrix (shrinkage target).
- \(\Sigma_{t+1:t+H}\): Realized covariance over horizon \(H = 20\) days.
- \(\|\cdot\|_F\): Frobenius norm.

**Important:** This is a *proxy* target. We are not predicting the true optimal \(\lambda\) (which is unknowable). We are predicting the \(\lambda\) that would have minimized error relative to the realized covariance. This is the standard approach in this literature.

### Input Features (\(X\))
Features must be **observable at time \(t\)** (no look-ahead).

**Market Regime Features:**
1. VIX level (market volatility expectations).
2. VIX percentile over the past 1 year (relative regime position).
3. Realized volatility (20-day) of the equally-weighted portfolio.
4. Average pairwise correlation of assets in the window.
5. Cross-sectional dispersion of returns (standard deviation of returns across assets).
6. Market return over the window.
7. Maximum drawdown over the window (market stress indicator).

**Covariance Structure Features:**
8. Condition number of sample covariance matrix (ill-conditioning).
9. Trace of sample covariance (total variance).
10. Average eigenvalue magnitude.

**Total features:** 10. Additional engineered features (interaction, lag, rolling mean) were tested but did not improve predictive performance.

### Prediction Horizon
\(H = 20\) trading days (~1 month). Chosen because:
- Matches typical portfolio rebalancing frequency.
- Long enough to be economically meaningful.
- Short enough to estimate reliably.

### Data Split (Chronological)
- **Training:** 2000–2015 (~16 years).
- **Validation:** 2016–2019 (~4 years) for hyperparameter tuning.
- **Test:** 2020–2025 (~5 years) for final evaluation.

*Rationale: Strict temporal ordering prevents look-ahead bias. The test period includes the COVID-19 crash, the 2022 bear market, and recovery phases—providing a realistic stress test.*

---

## 4. Research Questions

### Primary Research Question
**Does incorporating market regime features into a supervised learning model improve out-of-sample predictions of optimal shrinkage intensity compared to a static baseline?**

### Secondary Research Questions
1. Which regime features are most predictive of optimal shrinkage?
2. Does improved covariance estimation translate to lower portfolio volatility?
3. Is the improvement economically meaningful and statistically significant?

---

## 5. Hypotheses

### H1: Predictive Signal Exists
**H₁₀:** All regime features have zero predictive power (\(R^2 = 0\)).  
**H₁₁:** At least one regime feature has non-zero predictive power (\(R^2 > 0\)).

**Test:** Feature set evaluation via linear regression; Elastic Net coefficients.

### H2: Dynamic Shrinkage Improves Accuracy
**H₂₀:** Dynamic shrinkage does not reduce mean Frobenius distance vs. constant shrinkage.  
**H₂₁:** Dynamic shrinkage reduces mean Frobenius distance.

**Test:** Diebold-Mariano test for pairwise forecast comparison.

### H3: Economic Significance
**H₃₀:** Improved covariance estimation does not reduce portfolio volatility.  
**H₃₁:** Improved covariance estimation reduces portfolio volatility by a meaningful margin (\(\geq 5\%\)).

**Test:** Bootstrap confidence intervals for volatility difference; sub-period analysis.

---

## 6. Baseline Models

### Baseline 1: Constant Optimal Shrinkage (Primary Benchmark)
**Method:** Apply the average optimal \(\lambda\) from the training set to all test windows.

$$
\lambda_{\text{pred}} = \text{mean}(\lambda^*_{\text{train}})
$$

**Justification:** Represents standard practice—choose a shrinkage parameter and stick with it. If no model beats this, there is no exploitable regime signal.

**Why it's a strong baseline:** Simple, interpretable, and represents the default approach in many applications.

---

### Baseline 2: VIX Threshold Rule (Simple Dynamic Baseline)
**Method:** A 3-regime rule based on VIX level, with thresholds optimized on training data.

1. Find thresholds \(\tau_{\text{low}}\) and \(\tau_{\text{high}}\) on training data that minimize average Frobenius distance.
2. Assign \(\lambda\) based on regime:
   - \(\mathrm{VIX} < \tau_{\mathrm{low}}\): low-volatility regime.
   - \(\tau_{\mathrm{low}} \leq \mathrm{VIX} < \tau_{\mathrm{high}}\): medium-volatility regime.
   - \(\mathrm{VIX} \geq \tau_{\mathrm{high}}\): high-volatility regime.

**Justification:** Tests whether a simple, interpretable rule captures most of the regime variation. If this performs as well as ML, there's no need for complex models.

---

### Baseline 3: Ledoit-Wolf (Industry Standard)
**Method:** The Ledoit-Wolf (2004) shrinkage estimator.

$$
\Sigma_{LW} = (1 - \lambda_{LW}) S + \lambda_{LW} I
$$

Where \(\lambda_{LW}\) is computed analytically from the data structure, not optimized for regimes.

**Justification:** The most widely used covariance shrinkage method in practice. A strong benchmark.

---

### Baseline 4: Rolling Average \(\lambda\)
**Method:** Use the average optimal \(\lambda\) from the last \(K = 10\) windows as the prediction.

$$
\lambda_{\text{pred}} = \text{mean}(\lambda^*_{t-K:t})
$$

**Justification:** Tests whether simple temporal smoothing captures regime changes.

---

## 7. Advanced Model

### Elastic Net (Regularized Linear Regression)
**Method:** Linear regression with L1 + L2 regularization.

$$
\lambda_{\text{pred}} = \beta_0 + \sum_i \beta_i \cdot \text{feature}_i
$$

**Why this is the advanced model (not Gradient Boosting):**

1. **Simplicity:** Linear models are interpretable, which is valuable for understanding which features matter.
2. **Regularization:** Elastic Net handles correlated features (common in finance).
3. **Performance:** For problems with ~10 features and ~2000 samples, linear models often perform comparably to non-linear models.
4. **Baseline progression:** If Elastic Net fails, non-linear models are unlikely to succeed (Occam's razor).
5. **Economic interpretation:** Coefficients directly tell us which regime features matter and in what direction.

**Model Selection Rationale:**
- **Not Gradient Boosting:** Only introduce non-linearity if linear models systematically underperform.
- **Not Deep Learning:** Dataset is too small; interpretability is lost.
- **Not Random Forest:** Less interpretable than Elastic Net; no clear advantage for this problem.

**Hyperparameter Tuning:**
- \(\alpha\) (L1 ratio): 0.1, 0.3, 0.5, 0.7, 0.9.
- \(\lambda\) (regularization strength): 0.001, 0.01, 0.1, 1.0, 10.0.
- Tuning via validation set (time-series split).

---

### 7.1 Robustness Check: XGBoost

To test whether a non-linear model could capture relationships missed by Elastic Net, **XGBoost** was evaluated as a robustness check using the same VIX-Only feature set and train/validation/test split.

| Model | RMSE (λ) | R² (λ) | Mean Frobenius |
|-------|----------|--------|----------------|
| Elastic Net | 0.000530 | -0.0228 | 0.0121 |
| XGBoost | 0.000530 | -0.0228 | 0.0121 |

**Result:** XGBoost performed identically to Elastic Net, achieving the same RMSE, R², and Frobenius distance. This confirms that the limitation is not model linearity, but the inherent difficulty of predicting near-zero shrinkage intensities from market features.

---

## 8. Evaluation Framework

### Primary Metrics (Covariance Estimation)

| Metric | Definition | Why Used |
|--------|------------|----------|
| **Frobenius Distance** | ‖Σ_est − Σ_real‖_F | Primary metric; directly measures estimation error. |
| **RMSE of λ** | √(mean((λ_pred − λ*)²)) | Measures prediction accuracy of the target. |
| **R² of λ** | 1 − (SS_res / SS_tot) | Measures proportion of variance explained. |



### Secondary Metrics (Portfolio Impact)

| Metric | Definition | Why Used |
|--------|------------|----------|
| **Realized Volatility** | Std dev of portfolio returns | Direct economic measure. |
| **Turnover** | Mean absolute weight change | Practical cost consideration. |
| **Sharpe Ratio** | (R − R_f) / σ | Risk-adjusted performance. |
| **Maximum Drawdown** | Max peak-to-trough decline | Tail risk measure. |

### Statistical Tests

1. **Diebold-Mariano Test:** Compares predictive accuracy of two models (constant vs. dynamic).
2. **Bootstrap Confidence Intervals:** For volatility differences.
3. **Sub-period Analysis:** Performance across distinct market regimes.

### Evaluation Protocol

1. All models trained on training set only.
2. Hyperparameters tuned on validation set.
3. Final evaluation on test set (one forward test).
4. Results reported with p-values and confidence intervals.
5. Feature importance analyzed for interpretability.

---

## 9. Success Criteria

### Statistical Success
- At least one dynamic model significantly outperforms Constant Shrinkage (\(p < 0.05\), Diebold-Mariano).
- Improvement is consistent across sub-periods (not driven by a single event).

### Practical Success
- \(\geq 5\%\) reduction in mean Frobenius distance.
- \(\geq 5\%\) reduction in out-of-sample portfolio volatility.
- Improvement does not increase turnover by \(>20\%\).

### Research Contribution
- Clear, evidence-based answer to the primary research question.
- Transparent reporting of both successes and failures.
- Reproducible code and results.

---

## 10. Risk Controls & Validation

### Look-Ahead Bias
- Strict chronological data split.
- Features use only data available at time \(t\).

### Overfitting
- Regularization (Elastic Net).
- Time-series cross-validation.
- Simple models preferred over complex.

### Survivorship Bias
- Fixed set of large-cap stocks with continuous history.

### Non-Stationarity
- Sub-period analysis across multiple market regimes.
- Results reported separately for each period.

---

## 11. Data Sources

### Primary Dataset

**Yahoo Finance / yfinance**
- Daily adjusted close prices for 50 liquid S&P 500 stocks.
- Period: January 2000 – December 2025.
- Access: Free via `yfinance` API.

### Supplementary Market Features

**VIX Index** (CBOE Volatility Index)
- Available via Yahoo Finance (`^VIX`).
- Used as a proxy for market-wide volatility expectations.

---

## 12. Project Scope & Deliverables

### Completed (Weeks 1–2)
- 50-stock data pipeline (2000–2025).
- 4 baseline models (Constant, VIX Threshold, Rolling Average, Ledoit-Wolf).
- Feature set selection (7 sets tested; VIX-Only best).
- Elastic Net with hyperparameter tuning.
- XGBoost robustness check.
- Portfolio evaluation (test set only).
- Sub-period analysis (4 test periods).
- Statistical tests (Diebold-Mariano, bootstrap).

### What Will NOT Be Built
- Production-grade MLOps infrastructure.
- Docker containers or Kubernetes deployment.
- Real-time data streaming or APIs.
- Deep learning models (LSTMs, neural networks).
- GPU-accelerated training.
- Distributed computing (Spark, Dask).

---

## 13. References

1. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

2. Ledoit, O., & Wolf, M. (2012). Nonlinear shrinkage estimation of large-dimensional covariance matrices. *Annals of Statistics*, 40(2), 1024-1060.

3. De Nard, G., Ledoit, O., & Wolf, M. (2021). Factor models for portfolio selection in large dimensions. *Journal of Financial Econometrics*, 19(2), 241-270.

4. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

5. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.

6. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

---

## Appendix: Mathematical Notation Reference

| Symbol | Description |
|--------|-------------|
| S_t | Sample covariance matrix at time t. |
| I | Identity matrix (shrinkage target). |
| λ | Shrinkage intensity (scalar in [0,1]). |
| Σ_realized, t+H | Realized covariance over next H days. |
| ‖·‖_F | Frobenius norm. |
| λ*_t | Optimal shrinkage intensity for window t. |
| d_F | Frobenius distance between covariance matrices. |
| VIX_t | CBOE Volatility Index at time t. |
| n | Number of assets in portfolio. |
| H | Prediction horizon (20 trading days). |


---

*This problem framing document serves as the foundation for the project implementation. All decisions about model selection, evaluation, and interpretation are guided by the principles and constraints outlined above. The framing is designed to evolve as the project develops and new insights emerge.*