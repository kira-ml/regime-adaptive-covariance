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

#### Priority 4: Elastic Net (Week 2)
- [ ] Create feature matrix X (10 regime features)
- [ ] Create target vector y (optimal λ)
- [ ] Standardize features using training set statistics
- [ ] Implement Elastic Net with time-series cross-validation
- [ ] Compare to baselines on test set

#### Priority 5: Documentation
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
- Will Elastic Net capture the regime signal better than Ledoit-Wolf?

---

### Important Notes

- **Data stored locally**: `data/raw/` and `data/processed/` are in `.gitignore` to avoid large files
- **Results stored locally**: `results/` is in `.gitignore`
- **GitHub is up to date**: All code changes have been pushed
- **Python version**: 3.10
- **Virtual environment**: `venv/` folder (in `.gitignore`)

---

**Last Updated**: August 10, 2026
**Project Status**: ✅ Week 1 Complete — Sub-Period Analysis Reveals Key Finding
**Next Milestone**: Week 2 — 50 Stocks + Statistical Tests
