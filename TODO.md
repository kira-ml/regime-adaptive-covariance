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

### August 10, 2026 - Day 2 & 3: Week 1 Expansion & Validation

#### What Was Accomplished

**1. Asset Universe Expansion**
- Expanded from 5 to 20 sector-balanced stocks
- Added tech (NVDA, GOOGL, CSCO), financials (BAC, GS, V), healthcare (PFE, UNH), consumer (PG, KO, WMT), industrials (CAT, BA), and others (DIS, NKE)
- Re-ran pipeline: 4085 rolling windows (down from 6148 due to data availability)
- **Result:** λ range now 0.000 to 0.006 (mean = 0.0001) — non-zero values appear!

**2. Lambda Grid Refinement**
- Changed grid from 21 points (0-1) to 101 points (0-0.2)
- **Result:** Captured finer λ values (0.002, 0.004, 0.006) that were previously missed

**3. Feature Analysis**
- Created `src/feature_analysis.py` to explore correlations
- **Key findings:**
  - VIX Level: 0.346 (strongest predictor)
  - Realized Volatility: 0.334
  - Cross-sectional Dispersion: 0.267
  - VIX Percentile: 0.232
- **Insight:** Regime signal exists — λ increases with VIX and volatility

**4. VIX Threshold Enhancement (3-Regime)**
- Upgraded from 2-regime to 3-regime VIX rule
- **Optimal thresholds (trained on 2000-2015):**
  - Low: 13.48 (calm markets)
  - High: 30.58 (stress/crisis)
- **Result:** More granular regime segmentation

**5. Train/Validation/Test Split**
- Implemented chronological split in `src/rolling_windows.py`
- **Split results:**
  - Training: 1842 windows (2000-2015)
  - Validation: 1006 windows (2016-2019)
  - Test: 1237 windows (2020-2025)
- **Importance:** Eliminates look-ahead bias, enables proper out-of-sample testing

**6. Ledoit-Wolf Baseline (Full Implementation)**
- Replaced placeholder with proper `sklearn.covariance.LedoitWolf`
- Added `train_returns` to window_data for fitting
- **Result:** Mean shrinkage intensity = 0.1095, Frobenius = 0.0042 (matches optimal!)

**7. Portfolio Evaluation (Test Set Only)**
- Implemented minimum-variance portfolio construction
- Evaluated all methods on test set (2020-2025) only
- **Test-set results (2020-2025):**
  | Method | Mean Volatility | vs Constant |
  |--------|----------------|-------------|
  | Constant | 0.008693 | 🏆 BEST |
  | Ledoit-Wolf | 0.008719 | +0.3% |
  | VIX Threshold | 0.009369 | +7.8% |
  | Rolling Average | 0.009369 | +7.8% |
  | Optimal | 0.009615 | +10.6% |

**8. Sub-Period Analysis (Key Finding!)**
- Analyzed performance across market regimes
- **Results:**
  | Period | Best Method | Improvement vs Constant |
  |--------|-------------|------------------------|
  | COVID Crash (2020) | Ledoit-Wolf | **+3.17%** |
  | Recovery (2020-2021) | Constant | 0% (tie) |
  | Bear Market (2022) | Constant | 0% (tie) |
  | Recovery (2023-2024) | Constant | 0% (tie) |
- **Critical Insight:** Ledoit-Wolf outperforms Constant during extreme stress (COVID crash) but not in normal markets

**9. New Files Created**
- `src/feature_analysis.py` - Feature correlation exploration
- `src/portfolio.py` - Minimum-variance portfolio construction
- `src/sub_period_analysis.py` - Regime-dependent performance testing

#### Key Results Summary (20 Stocks, Test Set 2020-2025)

| Metric | Value |
|--------|-------|
| Total windows processed | 4,085 |
| Mean optimal λ | 0.0001 |
| Constant baseline Frobenius | 0.0043 |
| Ledoit-Wolf Frobenius | 0.0042 |
| Ledoit-Wolf shrinkage | 0.1095 |
| Constant volatility (test) | 0.008693 |
| Ledoit-Wolf volatility (test) | 0.008719 |
| COVID crash improvement (LW) | +3.17% |

#### Bugs Fixed
- Fixed VIX threshold baseline to handle training subset properly (IndexError)
- Fixed length mismatch in VIX threshold baseline (boolean index)
- Fixed portfolio evaluation: test set only (removed look-ahead bias)
- Fixed sub-period analysis: used windows list for train_end dates
- Removed duplicate portfolio save (NoneType error)

#### Git Commits Pushed (21 total)
- "Expand asset universe to 20 sector-balanced stocks"
- "Refine lambda grid to 0-0.2 with 101 points"
- "Add feature analysis script for Day 2 exploration"
- "Enhance VIX threshold baseline to 3 regimes"
- "Add chronological train/val/test split"
- "Fix VIX threshold baseline to handle training subset properly"
- "Fix VIX threshold baseline: clean up variable names and logic"
- "Implement proper Ledoit-Wolf baseline with analytical shrinkage"
- "Add portfolio evaluation module and fix Ledoit-Wolf baseline"
- "Fix Ledoit-Wolf: use sklearn on actual returns data"
- "Add portfolio evaluation for economic significance testing"
- "Fix portfolio evaluation: use test set only to prevent look-ahead bias"
- "Finalize Week 1: clean portfolio evaluation with test set only"
- "Add sub-period analysis for regime-dependent performance testing"
- "Fix sub-period analysis: use windows list for train_end dates"
- (and 6 additional commits for various fixes)

---

### Planned: Baseline-First Feature Engineering (Next Session)

#### Goal
Test feature sets systematically — baseline first, advanced only if needed.

#### Feature Sets to Test

**Baseline Sets (Test First):**
| Set | Features | Rationale |
|-----|----------|-----------|
| **Set 1: VIX-Only (1)** | `vix_level` | Simplest regime indicator, strongest correlation (0.346) |
| **Set 2: Vol+Corr (3)** | `vix_level`, `realized_vol`, `avg_correlation` | Market stress + correlation structure |
| **Set 3: Market (6)** | `vix_level`, `realized_vol`, `avg_correlation`, `cross_sectional_dispersion`, `vix_percentile`, `max_drawdown` | All market regime features |

**Advanced Sets (Test Only if Baselines Underperform):**
| Set | Features | Rationale |
|-----|----------|-----------|
| **Set 4: Covariance (4)** | `condition_number`, `trace`, `avg_eigenvalue`, `avg_correlation` | Covariance matrix properties |
| **Set 5: All (10)** | All 10 features | Full information set |

#### Implementation Plan

1. **Create `src/feature_engineering.py`**
   - `FeatureSet` class with all sets defined
   - `FeatureEngineer` class for evaluation
   - `prepare_features()` - X, y creation with scaling
   - `evaluate_feature_sets()` - Test all sets on test data
   - `compare_feature_sets()` - RMSE, R² comparison
   - `get_best_feature_set()` - Return best performing set

2. **Add to `main.py`**
   - Initialize feature engineer
   - Evaluate all sets on test data
   - Print comparison table
   - Save results to CSV
   - Generate comparison plot

3. **Evaluation Criteria**
   - **RMSE** (lower is better) - Primary metric
   - **R²** (higher is better) - Secondary metric
   - **Improvement threshold**: >5% improvement = advanced sets add value

#### Expected Output

```
=== Feature Set Comparison ===
feature_set                    n_features     rmse       r2
Baseline VIX-Only (1)          1              0.0008     0.120
Baseline Vol+Corr (3)          3              0.0007     0.230
Baseline Market (6)            6              0.0007     0.250
Advanced Covariance (4)        4              0.0008     0.100
Advanced All (10)              10             0.0007     0.260

🏆 Best Feature Set: Baseline Market (6)
   RMSE: 0.0007
   R²: 0.2500
```

#### Key Questions to Answer
- Do baseline sets provide good predictive power?
- Do advanced features add value (>5% improvement)?
- Which features are most important for predicting λ?
- Does test-set RMSE confirm correlation analysis?

#### Why This Matters
- **Defensible**: Baseline-first approach shows you didn't over-engineer
- **Statistical**: Out-of-sample comparison on test set
- **Practical**: Clear answer on what features to use for Elastic Net
- **Fair**: All sets evaluated on same test data

---

### Updated TODOs (Next Session)

#### Priority 1: Expand to 50 Stocks
- [ ] Test if Ledoit-Wolf's COVID advantage holds with larger universe
- [ ] Add delay to avoid Yahoo Finance rate limits
- [ ] Compare results with 20-stock findings

#### Priority 2: Statistical Significance Tests
- [ ] Implement Diebold-Mariano test for Ledoit-Wolf vs Constant
- [ ] Add bootstrap confidence intervals for volatility differences
- [ ] Test if COVID crash improvement is statistically significant

#### Priority 3: Simple Regime-Switch Strategy
- [ ] Test rule: Use Ledoit-Wolf when VIX > 30, Constant otherwise
- [ ] Compare performance vs Constant and Ledoit-Wolf individually
- [ ] Evaluate if simple rule provides best of both worlds

#### Priority 4: Baseline-First Feature Engineering
- [ ] Create `src/feature_engineering.py` with 3 baseline + 2 advanced sets
- [ ] Add feature engineering section to `main.py`
- [ ] Evaluate all feature sets on test data
- [ ] Compare RMSE and R² across sets
- [ ] Determine if advanced sets add value (>5% improvement)
- [ ] Save results and plots

#### Priority 5: Elastic Net (Week 2)
- [ ] Use best feature set from feature engineering
- [ ] Create feature matrix X (selected features)
- [ ] Create target vector y (optimal λ)
- [ ] Standardize features using training set statistics
- [ ] Implement Elastic Net with time-series cross-validation
- [ ] Compare to baselines on test set

#### Priority 6: Documentation
- [ ] Update README.md with Week 1 findings
- [ ] Add comments to code where needed
- [ ] Begin formal research paper draft

---

### Notes & Observations (Day 2-3)

#### Key Learnings
1. **Frobenius ≠ Portfolio Volatility**: Ledoit-Wolf minimizes Frobenius (0.0042) but Constant has lower volatility (0.008693). Covariance estimation accuracy doesn't translate to portfolio performance.

2. **Dynamic Methods Work in Crises**: During COVID crash, Ledoit-Wolf reduced volatility by 3.17% vs Constant. This is the key finding — regime-adaptation matters most when markets are stressed.

3. **VIX Thresholds are Useful**: Data-driven thresholds (13.48, 30.58) provide interpretable regime boundaries for practitioners.

4. **Sample Covariance is Robust**: With 20 stocks × 120 days, sample covariance is already well-conditioned. Shrinkage provides minimal benefit in normal markets.

5. **Test-Set Validation is Critical**: Full-sample evaluation masked the COVID effect. Sub-period analysis revealed what aggregate metrics hid.

#### Questions for Next Session
- Does the COVID effect hold with 50 stocks?
- Is the 3.17% improvement statistically significant?
- Can a simple VIX > 30 rule beat Constant overall?
- Which feature set works best for Elastic Net?
- Will Elastic Net capture the regime signal better than Ledoit-Wolf?

---

### Commands for Feature Engineering Session

```powershell
# Activate environment
.\venv\Scripts\Activate

# Create feature engineering module
New-Item -Path src\feature_engineering.py -ItemType File

# Update main.py with feature engineering section

# Run pipeline
python main.py

# Check results
cat results/feature_set_comparison.csv

# Git status and commit
git status
git add .
git commit -m "Add baseline-first feature engineering with 3 baseline and 2 advanced sets"
git push
```

---

### Decision Points

| Decision | Criteria | Action |
|----------|----------|--------|
| **Universe Size** | If 50 stocks shows similar results | Stick with 20 stocks for faster iteration |
| **Feature Set** | If baseline sets have R² > 0.2 | Use best baseline set for Elastic Net |
| **Feature Set** | If advanced sets show >5% improvement | Use best advanced set for Elastic Net |
| **Statistical Test** | If p < 0.05 for Ledoit-Wolf vs Constant | Include in paper as significant finding |
| **Regime-Switch** | If VIX > 30 rule beats Constant | Add as practical recommendation |

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
5. Feature engineering import errors

---
 

## ✅ Updated August 13, 2026 Entry (Corrected)

---

### August 13, 2026 - Day 5: Statistical Fixes, Pipeline Re-Run, Data Validation & LinkedIn PDF Finalization

#### What Was Accomplished

**1. Statistical & Mathematical Bug Fixes (Critical)**
- Fixed ML model predictions: added `np.clip(lam, 0.0, 1.0)` to Elastic Net and XGBoost to ensure λ stays within the valid [0,1] range.
- Removed look-ahead bias from VIX Threshold baseline: regime averages are now computed **only from training data** and applied to all windows.
- Removed artificial R² floor (`max(0.0, r2_raw)`) from feature engineering to report true negative R² (-0.0228).
- Updated Diebold-Mariano test to use **Newey-West correction (`h=10`)** to account for overlapping windows.

**2. Full Pipeline Re-Run (50 Stocks)**
- Re-ran `python main.py` with the corrected codebase.
- Total windows processed: **2,879**.
- Confirmed λ* mean = `0.000034` (near-zero).
- Elastic Net and XGBoost RMSE = `0.000530`; R² = `-0.0228`.
- Ledoit-Wolf mean Frobenius = `0.01172`; Constant = `0.01214`.
- Ledoit-Wolf volatility (test set) = `0.008385`; Constant = `0.010099` ( **~17% reduction** ).
- Diebold-Mariano (Ledoit-Wolf vs Constant): statistic = `-3.14`, p = `0.0017` (statistically significant).
- Bootstrap (volatility): Ledoit-Wolf vs Constant p = `0.0000`.

**3. Data Validation (Cross-Checked Against Results Files)**
- Verified every number in the output against the actual CSV/JSON files:
  - `metrics.csv` → mean λ*, std λ*
  - `portfolio_metrics_test.csv` → Ledoit-Wolf and Constant volatility
  - `elastic_net_results.csv` / `xgboost_results.csv` → ML Frobenius distances
  - `statistical_tests.json` → DM statistic, p-values, bootstrap confidence intervals
- **Result:** All numbers were confirmed accurate. No discrepancies found.

**4. Problem Framing & README Updates**
- Updated `problem_framing.md` to reflect the validated results:
  - Added a new **Core Empirical Finding** section (λ* is near-zero).
  - Reframed research questions and hypotheses to be realistic.
  - Aligned success criteria with actual outcomes (Ledoit-Wolf passes; ML fails).
- Updated `README.md` with corrected statistical tests, negative R², and honest success criteria.

**5. LinkedIn PDF Summary Generator (ReportLab)**
- Created `src/generate_linkedin_pdf.py` to generate a **3-page, modern, academic-style PDF** for LinkedIn posting.
- Used **ReportLab** (pure Python, no external GTK dependencies).
- Embedded 3 key figures:
  - `linkedin_plot1_lambda_over_time.png` (λ* over time)
  - `linkedin_plot2_frobenius_comparison.png` (Covariance accuracy)
  - `linkedin_plot3_sub_period_heatmap.png` (Portfolio volatility by regime)
- Implemented robust Times New Roman font registration (multiple Windows paths + fallback).
- PDF dynamically reads from:
  - `metrics.csv` (λ mean, std)
  - `portfolio_metrics_test.csv` (volatility)
  - `elastic_net_results.csv` / `xgboost_results.csv` (Frobenius)
  - `statistical_tests.json` (p-values)
- **Result:** PDF automatically updates if results change — no hardcoded numbers.

**6. Final Validation of PDF Content**
- Cross-checked every number in the PDF against the actual CSV/JSON files:
  - **Mean λ*:** `metrics.csv` → `3.40e-05` ✅
  - **Std λ*:** `metrics.csv` → `0.00035` ✅
  - **Constant Volatility:** `portfolio_metrics_test.csv` → `0.01010` ✅
  - **Ledoit-Wolf Volatility:** `portfolio_metrics_test.csv` → `0.00839` ✅
  - **Volatility Reduction:** Calculated → `~17%` ✅
  - **DM p-value:** `statistical_tests.json` → `0.0017` ✅
  - **Frobenius Distances:** Verified against `elastic_net_results.csv`, `xgboost_results.csv`, and `stat_tests.json` ✅
- **GitHub Link:** Updated to `https://github.com/kira-ml/regime-adaptive-covariance.git` ✅

**7. Git Commits Pushed**
- "Fix statistical and mathematical validity issues: clip ML predictions to [0,1], remove VIX baseline look-ahead bias, stop flooring R² at 0, and add autocorrelation correction (h=10) to Diebold-Mariano tests"
- "Update problem_framing.md to reflect validated results: add core finding (λ* near-zero), reframe hypotheses, and align success criteria with actual outcomes"
- "Update README.md with corrected results: reflect validated statistical tests, near-zero λ* finding, negative R², and honest success criteria"
- "Create LinkedIn PDF generator using ReportLab with validated data and modern academic styling"
- "Fix Unicode rendering and GitHub link in LinkedIn PDF generator"

---

### Summary of the Day

| Milestone | Status |
|-----------|--------|
| Statistical bugs fixed (clipping, VIX look-ahead, R² floor, DM correction) | ✅ Complete |
| Full pipeline re-run with 50 stocks (2,879 windows) | ✅ Complete |
| Data validation against CSV/JSON files | ✅ Complete |
| Problem framing & README updated | ✅ Complete |
| LinkedIn PDF generator created | ✅ Complete |
| PDF content validated against source data | ✅ Complete |
| All changes pushed to GitHub | ✅ Complete |

---

**Last Updated**: August 13, 2026 - 11:30 PM  
**Project Status**: ✅ Statistical validity restored; pipeline re-run; LinkedIn PDF finalized  
**Next Milestone**: Paper drafting and submission preparation


