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
- **Complex**: Multiple features (volatility, correlation structure, market stress) likely interact
- **Non-linear**: The effect of VIX on optimal λ is probably not linear
- **Unknown**: No closed-form solution exists for regime-dependent optimal λ
- **Predictable**: Market conditions are observable and contain forward-looking information

This makes it a natural candidate for supervised learning—predict the optimal λ from observable market features.

### What This Project Does Not Claim

- This is **not** a new covariance estimator
- This is **not** a production-ready risk system
- This is **not** guaranteed to outperform existing methods
- This is **not** financial advice

### What This Project Actually Does

**Empirically tests whether market regime information improves out-of-sample covariance shrinkage predictions compared to static approaches.**

---

## 3. Problem Framing (Machine Learning Formulation)

### Task Type
**Supervised regression** (predicting a continuous target: λ ∈ [0,1])

### Unit of Observation
A rolling estimation window of `T` trading days (e.g., 120 days) ending at time `t`.

### Target Variable (y)
For each window `t`, the target is the optimal shrinkage intensity:

```
λ*_t = argmin_{λ ∈ [0,1]} || (1-λ) * S_t + λ * I - Σ_{t+1:t+H} ||_F^2
```

Where:
- `S_t`: Sample covariance matrix from window ending at t
- `I`: Identity matrix (shrinkage target)
- `Σ_{t+1:t+H}`: Realized covariance over horizon H (20 days)
- `||·||_F`: Frobenius norm

**Important:** This is a *proxy* target. We are not predicting the true optimal λ (which is unknowable). We are predicting the λ that would have minimized error relative to the realized covariance. This is the standard approach in this literature.

### Input Features (X)
Features must be **observable at time t** (no look-ahead).

**Market Regime Features:**
1. VIX level (market volatility expectations)
2. VIX percentile over past 1 year (relative regime position)
3. Realized volatility (20-day) of the equally-weighted portfolio
4. Average pairwise correlation of assets in the window
5. Cross-sectional dispersion of returns (standard deviation of returns across assets)
6. Market return (S&P 500 performance) over the window
7. Maximum drawdown over the window (market stress indicator)

**Covariance Structure Features:**
8. Condition number of sample covariance matrix (ill-conditioning)
9. Trace of sample covariance (total variance)
10. Average eigenvalue magnitude

**Total features:** 10 (Week 1: 3-5, Week 2+: full set)

### Prediction Horizon
`H = 20` trading days (~1 month). Chosen because:
- Matches typical portfolio rebalancing frequency
- Long enough to be economically meaningful
- Short enough to estimate reliably

### Data Split (Chronological)
- **Training:** 2000–2015 (~16 years)
- **Validation:** 2016–2019 (~4 years) for hyperparameter tuning
- **Test:** 2020–2025 (~5 years) for final evaluation

*Rationale: Strict temporal ordering prevents look-ahead bias. The test period includes COVID-19 and 2022 bear market—stress tests.*

---

## 4. Research Questions

### Primary Research Question
**Does incorporating market regime features into a supervised learning model improve out-of-sample predictions of optimal shrinkage intensity compared to a static baseline?**

### Secondary Research Questions
1. Which regime features are most predictive of optimal shrinkage?
2. Does improved shrinkage prediction translate to lower portfolio volatility?
3. Is the improvement economically meaningful (≥5% volatility reduction)?

---

## 5. Hypotheses

### H1: Predictive Signal Exists
**H₁₀:** All regime features have zero predictive power (R² = 0)  
**H₁₁:** At least one regime feature has non-zero predictive power (R² > 0)

**Test:** F-test on Elastic Net coefficients; feature importance for Gradient Boosting.

### H2: Dynamic Shrinkage Improves Accuracy
**H₂₀:** Dynamic shrinkage does not reduce mean Frobenius distance vs. constant shrinkage  
**H₂₁:** Dynamic shrinkage reduces mean Frobenius distance

**Test:** Diebold-Mariano test for pairwise forecast comparison.

### H3: Economic Significance
**H₃₀:** Improved covariance estimation does not reduce portfolio volatility  
**H₃₁:** Improved covariance estimation reduces portfolio volatility by ≥5%

**Test:** Bootstrap confidence intervals for volatility difference; sub-period analysis.

---

## 6. Baseline Models

### Baseline 1: Constant Optimal Shrinkage (Primary Benchmark)
**Method:** Apply the average optimal λ from the training set to all test windows.

```
λ_pred = mean(λ*_train)
```

**Justification:** Represents standard practice—choose a shrinkage parameter and stick with it. If no model beats this, there is no exploitable regime signal.

**Why it's a strong baseline:** It's simple, interpretable, and represents the default approach in many applications.

---

### Baseline 2: VIX Threshold Rule (Simple Dynamic Baseline)
**Method:** A two-regime rule based on VIX level.

1. Find threshold `τ` on training data that minimizes average Frobenius distance
2. Apply rule:
   - If VIX_t < τ: λ_pred = mean(λ*_train | VIX < τ)
   - If VIX_t ≥ τ: λ_pred = mean(λ*_train | VIX ≥ τ)

**Justification:** Tests whether a simple, interpretable rule captures most of the regime variation. If this performs as well as ML, there's no need for complex models.

**Why it's useful:** Represents what a practitioner might implement without ML. Sets a realistic bar for ML to clear.

---

### Baseline 3: Ledoit-Wolf (Industry Standard)
**Method:** The Ledoit-Wolf (2004) shrinkage estimator.

```
Σ_LW = (1 - λ_LW) * S + λ_LW * I
```

Where λ_LW is computed analytically from the data structure, not optimized for regimes.

**Justification:** The most widely used covariance shrinkage method in practice. Strong benchmark.

**Why it's useful:** If regime-adaptive methods can't beat Ledoit-Wolf, the project has no practical value.

---

### Baseline 4: Rolling Average λ
**Method:** Use the average optimal λ from the last `K` windows as the prediction.

```
λ_pred = mean(λ*_{t-K:t})
```

**Justification:** Tests whether simple temporal smoothing captures regime changes.

**Why included:** Accounts for the possibility that λ is persistent but not regime-dependent.

---

## 7. Advanced Model

### Elastic Net (Regularized Linear Regression)
**Method:** Linear regression with L1 + L2 regularization.

```
λ_pred = β_0 + Σ β_i * feature_i
```

**Why this is the advanced model (not Gradient Boosting):**

1. **Simplicity:** Linear models are interpretable, which is valuable for understanding which features matter
2. **Regularization:** Elastic Net handles correlated features (common in finance)
3. **Performance:** For problems with ~10 features and ~2000 samples, linear models often perform comparably to non-linear models
4. **Baseline progression:** If Elastic Net fails, non-linear models are unlikely to succeed (Occam's razor)
5. **Economic interpretation:** Coefficients directly tell us which regime features matter and in what direction

**Model Selection Rationale:**
- **Not Gradient Boosting (Week 1-2):** Only introduce non-linearity if linear models systematically underperform
- **Not Deep Learning:** Dataset is too small; interpretability is lost
- **Not Random Forest:** Less interpretable than Elastic Net; no clear advantage for this problem

**Hyperparameter Tuning:**
- α (L1 ratio): 0.1, 0.3, 0.5, 0.7, 0.9
- λ (regularization strength): 0.001, 0.01, 0.1, 1.0, 10.0
- Tuning via time-series cross-validation on validation set

**Decision Point for Gradient Boosting:**
If Elastic Net shows predictive signal (R² > 0.05, p < 0.05) but leaves systematic residuals (patterns in prediction errors across regimes), then introduce Gradient Boosting as a robustness check.

---

## 8. Evaluation Framework

### Primary Metrics (Covariance Estimation)

| Metric | Definition | Why Used |
|--------|------------|----------|
| **Frobenius Distance** | `||Σ_est - Σ_real||_F` | Primary metric; directly measures estimation error |
| **RMSE of λ** | `sqrt(mean((λ_pred - λ*)²))` | Measures prediction accuracy of target |
| **R² of λ** | `1 - (SS_res / SS_tot)` | Measures proportion of variance explained |

### Secondary Metrics (Portfolio Impact)

| Metric | Definition | Why Used |
|--------|------------|----------|
| **Realized Volatility** | Std dev of portfolio returns | Direct economic measure |
| **Turnover** | Mean absolute weight change | Practical cost consideration |
| **Sharpe Ratio** | (Return - Rf) / Volatility | Risk-adjusted performance |
| **Maximum Drawdown** | Max peak-to-trough decline | Tail risk measure |

### Statistical Tests

1. **Diebold-Mariano Test:** Compares predictive accuracy of two models (constant vs. dynamic)
2. **Bootstrap Confidence Intervals:** For volatility and Sharpe ratio differences
3. **Sub-period Analysis:** Performance in high-volatility (VIX > 25) vs. low-volatility (VIX < 15) periods

### Evaluation Protocol

1. All models trained on training set only
2. Hyperparameters tuned on validation set (time-series CV)
3. Final evaluation on test set (one forward test)
4. Results reported with standard errors
5. Feature importance analyzed for interpretability

---

## 9. Success Criteria

### Statistical Success
- At least one dynamic model significantly outperforms Constant Shrinkage baseline (p < 0.05, Diebold-Mariano)
- Improvement is consistent across sub-periods (not driven by one event)

### Practical Success
- ≥5% reduction in mean Frobenius distance
- ≥5% reduction in out-of-sample portfolio volatility
- Improvement does not increase turnover by >20%

### Research Contribution
- Clear answer to the primary research question
- Feature importance analysis reveals which regime indicators matter
- Reproducible code and results
- Transparent discussion of limitations and failure modes

---

## 10. Risk Controls & Validation

### Look-Ahead Bias
- Strict chronological data split
- Features use only data available at time t
- Manual verification of window alignment

### Overfitting
- Regularization (Elastic Net)
- Time-series cross-validation
- Simple models preferred over complex

### Survivorship Bias
- Use fixed set of large-cap stocks (continuous history)
- Document any stock removals

### Non-Stationarity
- Sub-period analysis (2000-2007, 2008-2012, 2013-2019, 2020-2025)
- Report stability of results across periods

---

## 11. Data Sources

### Primary Dataset

**Yahoo Finance / yfinance**
- Daily adjusted close prices for 5-100 liquid S&P 500 stocks
- Period: January 2000 – December 2025
- Access: Free via `yfinance` API

### Supplementary Market Features

**VIX Index** (CBOE Volatility Index)
- Available via Yahoo Finance (`^VIX`)
- Used as a proxy for market-wide volatility expectations

**Fama-French Factors** (Ken French Data Library)
- Daily returns for market, size, value, momentum factors
- Available for free download
- Optional: Used for robustness checks

**Treasury Yields** (FRED)
- 10-year Treasury yield, 3-month T-bill rate
- Available via FRED API or pandas-datareader
- Optional: Used for risk-free rate in Sharpe ratio

---

## 12. Project Scope & Deliverables

### Week 1 (MVP) - In Progress ✓
- Data pipeline (5 stocks, 2000-2025)
- Optimal λ computation (120-day window, 20-day horizon)
- Constant shrinkage baseline
- Basic visualization

### Week 2 (Planned)
- Feature engineering (VIX, volatility, correlation, dispersion)
- Elastic Net model with cross-validation
- Rule-based VIX threshold baseline
- Evaluation framework

### Week 3 (If justified)
- Gradient Boosting comparison
- Portfolio construction and evaluation
- Robustness checks
- Final write-up

### What Will NOT Be Built
- Production-grade MLOps infrastructure
- Docker containers or Kubernetes deployment
- Real-time data streaming or APIs
- Deep learning models (LSTMs, neural networks)
- GPU-accelerated training
- Distributed computing (Spark, Dask)

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
| `S_t` | Sample covariance matrix at time t |
| `I` | Identity matrix (shrinkage target) |
| `λ` | Shrinkage intensity (scalar in [0,1]) |
| `Σ_realized,t+H` | Realized covariance over next H days |
| `||·||_F` | Frobenius norm |
| `λ*_t` | Optimal shrinkage intensity for window t |
| `d_F` | Frobenius distance between covariance matrices |
| `VIX_t` | CBOE Volatility Index at time t |
| `n` | Number of assets in portfolio |
| `H` | Prediction horizon (20 trading days) |

---

*This problem framing document serves as the foundation for the project implementation. All decisions about model selection, evaluation, and interpretation should be guided by the principles and constraints outlined above. The framing is designed to evolve as the project develops and new insights emerge.*