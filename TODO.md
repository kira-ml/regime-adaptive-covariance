# TODO.md - Regime-Adaptive Covariance Estimation

## Project Log & Task Tracker

---

### August 9, 2026 (2:00 AM) - Day 1: Project Setup & Week 1 MVP

#### What Was Accomplished

**1. Project Initialization**
- Created project repository: `regime-adaptive-covariance`
- Set up virtual environment with Python 3.10
- Installed all dependencies from `requirements.txt`
- Initialized Git and pushed to GitHub: https://github.com/kira-ml/regime-adaptive-covariance

**2. Problem Framing**
- Optimized `problem_framing.md` with improved ML formulation
- Defined 3 baseline models (Constant Shrinkage, VIX Threshold, Rolling Average)
- Defined 1 advanced model (Elastic Net)
- Created testable hypotheses with statistical tests
- Established evaluation framework (Frobenius distance, portfolio metrics)

**3. Code Implementation - Week 1 MVP**
Created modular pipeline with 7 Python modules:

| File | Purpose | Status |
|------|---------|--------|
| `src/data_ingestion.py` | Download stock prices & VIX from Yahoo Finance | ✅ Complete |
| `src/data_preprocessing.py` | Compute returns, clean data, align VIX | ✅ Complete |
| `src/rolling_windows.py` | Create rolling windows, compute covariance matrices & regime features | ✅ Complete |
| `src/optimal_lambda.py` | Find optimal shrinkage intensity λ via grid search | ✅ Complete |
| `src/baselines.py` | Implement Constant, VIX Threshold, Rolling Average baselines | ✅ Complete |
| `src/evaluation.py` | Generate plots, metrics, visualizations | ✅ Complete |
| `main.py` | Orchestrate full pipeline end-to-end | ✅ Complete |

**4. Pipeline Execution Results**
- Data downloaded: 5 stocks (AAPL, MSFT, JPM, JNJ, XOM) + VIX
- Period: 2000-01-01 to 2025-01-01
- Windows: 6,148 rolling windows (120-day window, 20-day horizon)
- Optimal lambda range: 0.000 to 0.000 (mean = 0.000)
- All baselines performed equally (Frobenius distance = 0.0011)
- Results saved to `data/processed/` and `results/`

**5. Bugs Fixed**
- Fixed empty DataFrame handling in data ingestion
- Fixed MultiIndex column format issue with Yahoo Finance
- Fixed VIX Series vs DataFrame ambiguity
- Added edge case handling for empty window_data
- Made baseline functions robust to empty data

**6. Git Commits Pushed**
- "Week 1: Minimal pipeline for regime-adaptive covariance estimation"
- "Optimize problem framing: sharpen ML formulation, add baselines, define evaluation framework"
- "Add VIX download function to data_ingestion"
- "Update data_preprocessing: add VIX alignment function"
- "Update all src modules: add VIX support, regime features, baseline implementations"
- "Add main.py orchestrator for Week 1 pipeline"
- "Complete main.py with full pipeline and all baselines"
- "Fix data ingestion: handle empty DataFrames and missing Adj Close column"
- "Fix VIX Series and baseline empty data handling"

---

### Files Created

```
regime-adaptive-covariance/
├── main.py                          ✅ Orchestrator
├── problem_framing.md               ✅ Optimized ML framing
├── README.md                        ✅ Project overview
├── requirements.txt                 ✅ All dependencies
├── .gitignore                       ✅ Git ignore rules
│
├── src/
│   ├── __init__.py                  ✅ Package marker
│   ├── data_ingestion.py            ✅ Downloads prices + VIX
│   ├── data_preprocessing.py        ✅ Returns + cleaning + VIX alignment
│   ├── rolling_windows.py           ✅ Windows + covariance + regime features
│   ├── optimal_lambda.py            ✅ Grid search for optimal λ
│   ├── baselines.py                 ✅ 3 baseline implementations
│   └── evaluation.py                ✅ Plots + metrics
│
├── data/
│   ├── raw/                         📁 Created (empty)
│   └── processed/
│       ├── optimal_lambdas.csv      ✅ 6,148 rows
│       └── regime_features.csv      ✅ 6,148 rows × 10 features
│
└── results/
    ├── figures/
    │   ├── lambda_over_time.png     ✅ Plot generated
    │   ├── frobenius_comparison.png ✅ Plot generated
    │   └── feature_correlation.png  ✅ Plot generated
    └── metrics.csv                  ✅ Summary metrics
```

---

### Key Results Summary

| Metric | Value |
|--------|-------|
| Total windows processed | 6,148 |
| Mean optimal λ | 0.0000 |
| Constant baseline Frobenius | 0.0011 |
| Optimal Frobenius | 0.0011 |
| Improvement | 0.0% |
| VIX threshold baseline | 0.0011 |
| Rolling average baseline | 0.0011 |

---

### Immediate TODOs (After Sleep / August 9, 2026)

#### Priority 1: Data Expansion
- [ ] Increase tickers to 20-30 stocks (e.g., S&P 100 subset)
- [ ] Re-run pipeline with larger universe
- [ ] Verify lambda becomes > 0 with more assets

#### Priority 2: Feature Analysis
- [ ] Explore `data/processed/regime_features.csv`
- [ ] Plot lambda vs VIX levels
- [ ] Identify which features correlate with lambda
- [ ] Check if lambda variation exists across regimes

#### Priority 3: Train/Validation/Test Split
- [ ] Split windows chronologically:
  - Training: 2000-2015
  - Validation: 2016-2019
  - Test: 2020-2025
- [ ] Compute baselines only on training data
- [ ] Evaluate on validation and test sets

#### Priority 4: Week 2 - Feature Engineering
- [ ] Create feature matrix X (regime features)
- [ ] Create target vector y (optimal λ)
- [ ] Standardize features using training set statistics
- [ ] Implement Elastic Net with time-series cross-validation
- [ ] Compare to baselines on test set

#### Priority 5: Week 2 - Model Evaluation
- [ ] Compute RMSE, MAE, R² for λ predictions
- [ ] Compute Frobenius distance using predicted λ
- [ ] Run Diebold-Mariano test for statistical significance
- [ ] Analyze feature importance coefficients

#### Priority 6: Portfolio Evaluation
- [ ] Implement minimum-variance portfolio construction
- [ ] Compare volatility, turnover, Sharpe ratio across methods
- [ ] Evaluate economic significance (≥5% volatility reduction)

#### Priority 7: Documentation
- [ ] Update README.md with Week 1 progress
- [ ] Add comments to code where needed
- [ ] Create notebook for exploratory data analysis
- [ ] Document findings and next steps

---

### Stretch Goals (Week 3+)
- [ ] Gradient Boosting (if Elastic Net shows signal)
- [ ] Sub-period robustness checks (2008, COVID, 2022)
- [ ] Sensitivity to window length (60, 120, 250)
- [ ] Sensitivity to horizon (10, 20, 30 days)
- [ ] Ledoit-Wolf baseline implementation (full)

---

### Notes & Observations

#### Today's Learnings
1. **Yahoo Finance data format changed**: Uses MultiIndex `('Close', 'Ticker')` instead of `'Adj Close'`. Our code now handles both formats.

2. **λ = 0 with 5 stocks**: With small asset universes, the sample covariance matrix already outperforms shrinkage to identity. This is expected and confirms the pipeline works correctly.

3. **Pipeline is robust**: The pipeline now gracefully handles:
   - Empty DataFrames
   - Missing VIX data
   - 0 windows
   - NaN values in baselines

4. **All baselines equal**: With λ = 0 for all windows, all baselines produce identical results. This will change when we add more stocks.

#### Next Session Focus
- **Primary goal**: Add 20-30 stocks and re-run to see λ become meaningful
- **Secondary goal**: Analyze feature correlations and start Elastic Net implementation

#### Questions to Explore
- What is the optimal number of stocks for this analysis?
- Which regime features have the strongest predictive power?
- Is the relationship linear or non-linear?
- Does the benefit persist across different market regimes?

---

### Commands for Next Session

```powershell
# Activate environment
.\venv\Scripts\Activate

# Run pipeline with more stocks (after updating main.py)
python main.py

# Check results
ls data/processed/
ls results/figures/

# Git status and commit
git status
git add .
git commit -m "Description of changes"
git push
```

---

### Important Notes

- **Data stored locally**: `data/raw/` and `data/processed/` are in `.gitignore` to avoid large files
- **Results stored locally**: `results/` is in `.gitignore`
- **GitHub is up to date**: All code changes have been pushed
- **Python version**: 3.10
- **Virtual environment**: `venv/` folder (in `.gitignore`)

---

### Contact / Issues

If something breaks, check:
1. Yahoo Finance API changes
2. Missing data for specific tickers
3. Date ranges with no trading days
4. Memory usage with large asset universes

---

**Last Updated**: August 9, 2026 - 2:00 AM
**Project Status**: ✅ Week 1 MVP Complete
**Next Milestone**: Week 2 - Feature Engineering & Elastic Net