# Regime-Adaptive Covariance Estimation

> An evidence-driven investigation into whether market regime information improves out-of-sample covariance shrinkage for portfolio risk management

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Development](https://img.shields.io/badge/Status-Development-red.svg)]()

---

## 📋 Overview

This project investigates whether market regime information can improve out-of-sample covariance estimation for portfolio risk management. Traditional covariance estimators apply a static shrinkage target (e.g., identity matrix) regardless of market conditions, potentially leading to poor risk estimates during regime transitions.

We frame this as a supervised learning problem where we predict the optimal shrinkage intensity for each estimation window based on observable market features. The project follows a **baseline-first, evidence-driven** approach:

1. Establish meaningful statistical and rule-based baselines
2. Test simple linear models (Elastic Net)
3. Only increase complexity to non-linear models (Gradient Boosting) if empirically justified
4. Evaluate economic significance through portfolio construction

---

## 🎯 Problem Statement

### Core Challenge
Most risk models assume a stable, stationary covariance structure. In reality, markets transition between distinct volatility/correlation regimes. A single, static shrinkage target is almost certainly wrong for a significant portion of the investment horizon.

### Key Questions
1. Does incorporating market regime information improve shrinkage intensity predictions?
2. Which regime indicators provide the most predictive power?
3. Does improved covariance estimation lead to lower out-of-sample portfolio volatility?
4. Is the improvement economically meaningful?

---

## 🗓️ Project Status

**Current Phase:** 📝 Problem Framing & Data Collection

This project is in the initial planning and setup phase. The following milestones are planned:

- [x] Problem framing and research questions
- [x] Experimental design and evaluation framework
- [ ] Data collection and cleaning
- [ ] Feature engineering and target construction
- [ ] Baseline model implementation
- [ ] Advanced model implementation
- [ ] Portfolio evaluation
- [ ] Results analysis and documentation

---

## 📁 Project Structure

```
regime-adaptive-covariance/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── problem_framing.md            # Full problem framing document
│
├── notebooks/                    # Jupyter notebooks (development)
│   ├── 01_data_download_and_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   ├── 05_portfolio_evaluation.ipynb
│   └── 06_results_and_analysis.ipynb
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Data download and preprocessing
│   ├── feature_engineering.py   # Feature and target construction
│   ├── covariance_estimators.py # Covariance estimation methods
│   ├── portfolio_optimizer.py   # Minimum-variance portfolio construction
│   └── evaluation_metrics.py    # Metrics and statistical tests
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_covariance_estimators.py
│
├── results/                      # Output and analysis
│   ├── figures/
│   ├── tables/
│   └── models/                   # Saved model artifacts
│
└── paper/                        # Write-up
    └── project_report.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/regime-adaptive-covariance.git
   cd regime-adaptive-covariance
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**
   ```bash
   # Run the data download notebook
   jupyter notebook notebooks/01_data_download_and_exploration.ipynb
   ```

---

## 📊 Data Sources

| Data Source | Description | Period | Access |
|-------------|-------------|--------|--------|
| Yahoo Finance | S&P 500 daily prices | 2000–present | `yfinance` API |
| Yahoo Finance | VIX Index | 2000–present | `yfinance` API |
| Ken French Library | Fama-French factors | 2000–present | Free download |
| FRED | Treasury yields | 2000–present | `fredapi` or pandas-datareader |
| FRED | Credit spreads (optional) | 2000–present | `fredapi` or pandas-datareader |

---

## 🔬 Methodology

### 1. Problem Formulation

**Unit of Observation:** Rolling estimation window (e.g., 120 trading days)

**Target Variable:** Optimal shrinkage intensity `λ*_t` that minimizes Frobenius distance to future realized covariance:

```
λ*_t = argmin_λ || (1-λ)*S_t + λ*T_t - Σ_realized,t+20 ||_F^2
```

**Prediction Horizon:** 20 trading days (approximately 1 month)

### 2. Baseline Models

| Baseline | Description |
|----------|-------------|
| **Constant Shrinkage** | Average optimal λ across training set |
| **Rule-Based** | VIX threshold: λ_low when VIX < threshold, λ_high otherwise |
| **Ledoit-Wolf** | Industry-standard static shrinkage estimator |

### 3. Machine Learning Models

| Model | Purpose | Complexity |
|-------|---------|------------|
| **Elastic Net** | Linear model with regularization and feature selection | Low |
| **Gradient Boosting** | Non-linear model with interaction effects | Medium |

> **Note:** Complexity is increased only if empirical evidence shows simpler models have meaningful limitations.

### 4. Evaluation Metrics

**Covariance Estimation:**
- Frobenius Distance
- Kullback-Leibler Divergence
- RMSE of λ predictions

**Portfolio Performance:**
- Realized Volatility
- Turnover
- Sharpe Ratio
- Maximum Drawdown

### 5. Experimental Design

1. **Data Split** (chronological):
   - Training: 2000–2015
   - Validation: 2016–2019
   - Test: 2020–2025

2. **Validation**: Time-series cross-validation with rolling windows

3. **Robustness Checks**:
   - Different window lengths (60, 120, 250 days)
   - Different horizons (10, 20, 30 days)
   - Sub-period analysis
   - Asset universe sensitivity

---

## 📈 Expected Outcomes

### Hypotheses

**H1:** Regime features contain predictive information for optimal shrinkage

**H2:** Dynamic shrinkage reduces out-of-sample Frobenius distance vs. static approaches

**H3:** Improved covariance estimation reduces realized portfolio volatility by ≥5%

### Success Criteria

The project is successful if:
- At least one ML model significantly outperforms the Constant Shrinkage baseline (p < 0.05)
- Improvement translates to ≥5% out-of-sample portfolio volatility reduction
- The chosen model is the simplest one that performs well
- Results are robust across different market regimes and sensitivity tests

---

## 🧪 Key Risk Controls

| Risk | Mitigation |
|------|------------|
| Look-ahead bias | Strict temporal separation; only lagged features |
| Overfitting | Time-series CV; regularization; multiple regimes in training |
| Survivorship bias | Fixed asset universe or historical constituents |
| Non-stationarity | Sub-period analysis; rolling evaluation |
| Temporal dependence | Time-series cross-validation; cautious inference |

---

## 📚 References

1. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

2. Ledoit, O., & Wolf, M. (2012). Nonlinear shrinkage estimation of large-dimensional covariance matrices. *Annals of Statistics*, 40(2), 1024-1060.

3. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

4. De Nard, G., Ledoit, O., & Wolf, M. (2021). Factor models for portfolio selection in large dimensions. *Journal of Financial Econometrics*, 19(2), 241-270.

---

## 🤝 Contributing

This is a portfolio project. While contributions are not expected, suggestions and feedback are welcome. Please open an issue for discussion before submitting any pull requests.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback, please open an issue or reach out via [GitHub](https://github.com/yourusername).

---

## 🙏 Acknowledgments

- Ken French for maintaining the factor data library
- Yahoo Finance for providing accessible financial data
- The open-source Python ecosystem (pandas, numpy, scikit-learn, etc.)

---

## ⚠️ Disclaimer

**This project is for educational and research purposes only.** It does not constitute financial advice, and no investment decisions should be based on its outputs. Past performance is not indicative of future results.