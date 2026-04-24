# Equity Return Prediction Using Earnings Fundamentals

**Author:** Raz Kropveld &nbsp;|&nbsp; **Paper:** [equity_return_prediction_paper.html](equity_return_prediction_paper.html)

A systematic, walk-forward framework that predicts 60-trading-day forward equity returns using
point-in-time accounting fundamentals and XGBoost. The model is trained on 1999–2011 data and
evaluated entirely out-of-sample on 2012–2023, spanning multiple market regimes.

**Test-phase results (2012–2023):**
- Annualised return: **14.2%** (top-30 strategy) vs **6.9%** (equal-weighted universe benchmark)
- Single-period outperformance probability: **61%**; rolling 2-year: **77.5%**
- Mean rank IC: **0.024** | Worst single-period return: −26.3%

---

## Project structure

```
equity_return_prediction/
├── data/
│   ├── raw/                  # Raw inputs (not tracked — place files here, see below)
│   ├── processed/
│   │   ├── sample/           # 10%-ticker sample of processed intermediate files
│   │   └── train/            # model_dataset.csv for the train phase (1999–2011)
│   ├── predictions/train/    # Per-period predictions and feature importances
│   └── evaluation/           # runs_protocol.csv — one row per modeling run
├── results/
│   ├── figures/              # Paper figures (PNG)
│   └── tables/               # Full results tables (not tracked — too large)
├── scripts/
│   ├── data_preprocessing.py        # Stage 1: raw → processed panel → model_dataset
│   ├── modeling_and_prediction.py   # Stage 2: TSCV → predictions + feature importance
│   └── run_evaluation.py            # Stage 3: predictions → evaluation metrics + protocol row
├── src/
│   ├── config.py             # All paths, phase settings, model hyperparameters
│   ├── data_preparation.py   # Raw data ingestion and panel construction
│   ├── feature_engineering.py# Feature construction and model_dataset assembly
│   ├── modeling.py           # Walk-forward TSCV loop (XGBoost)
│   └── evaluation.py         # Metric computation and protocol logging
├── exploration/
│   ├── evaluation.ipynb      # Out-of-sample performance analysis
│   ├── eda_raw_data.ipynb    # EDA on raw prices and fundamentals
│   └── eda_processed_data.ipynb  # EDA on the engineered model_dataset
├── paper_gen.py              # Regenerates equity_return_prediction_paper.html
├── paper_figs.py             # Regenerates the three paper figures (requires results)
└── equity_return_prediction_paper.html  # Working paper
```

---

## How to run

All scripts are run from the project root. Set `phase` and `data_mode` in `src/config.py`
before running.

```bash
# Stage 1 — build processed panel and model_dataset
python scripts/data_preprocessing.py

# Stage 2 — run walk-forward TSCV, save predictions and feature importances
python scripts/modeling_and_prediction.py

# Stage 3 — compute evaluation metrics and append a row to runs_protocol.csv
python scripts/run_evaluation.py
```

Set `phase = "train"` to work with 1999–2011 data; `phase = "test"` for the 2012–2023
out-of-sample period. Set `data_mode = "sample"` to run on a 10%-ticker sample for
rapid iteration.

---

## Pipeline overview

### Stage 1 — Data preprocessing (`scripts/data_preprocessing.py`)

Reads raw daily prices and Sharadar SF1 point-in-time fundamentals, then builds:

1. **Daily price panel** — closing prices with 20-, 30-, and 60-trading-day forward prices
   and trailing 60-day returns.
2. **Fundamentals snapshot** — carries each quarterly/TTM report forward until the next
   filing date, creating a point-in-time panel that is strictly free of look-ahead bias.
3. **Daily panel** — fundamentals joined to prices on (ticker, date).
4. **Model dataset** — feature engineering applied to the panel:
   - Prediction target: log(price₊₆₀ / price₀), the 60-trading-day forward log-return.
   - Earnings yield and cash-flow yield (net income and free cash flow scaled by market cap).
   - Relative valuation signals: firm yield minus contemporaneous cross-sectional and
     sector means, capturing the Novy-Marx (2013) profitability signal structure.
   - Lagged fundamental changes (one-period lags of earnings and free cash flow).
   - Price momentum: 60-trading-day trailing return as a short-term control.
   - Report timing: business days since the most recent earnings release, capturing
     post-announcement drift (Ball & Brown, 1968; Bernard & Thomas, 1989).
   - Continuous features are log-transformed where appropriate; all features and the
     label are z-scored within each training window.

   Output: `data/processed/{phase}/model_dataset.csv`

### Stage 2 — Walk-forward TSCV (`scripts/modeling_and_prediction.py`)

Runs the three-block walk-forward cross-validation procedure over all monthly test periods:

- **Validation block** — placed *before* the training window. Used for early stopping
  (tree-count selection via XGBoost's `early_stopping_rounds`). This decouples model
  selection from the market regime immediately preceding the test period, eliminating
  an indirect form of temporal leakage present in conventional TSCV designs.
- **Training block** — up to 90 calendar days of history, capped at 100,000 observations.
  Sample weights combine time-decay and absolute-return components.
- **Buffer** — a minimum 61-trading-day gap separates the training block from the test
  period, ensuring no forward-return overlap.
- **Test block** — the forward month being evaluated. Predictions are made only after
  model selection is complete.

The model is XGBoost (`gbtree`, `eta=0.02`, `max_depth=3`, `gamma=0.1`, `lambda=0.3`).
At each period, predictions are made for all universe stocks with a recent earnings release
(`bdfr=1`: business day 1 after the filing date) and a market cap above ~$100 million.

Outputs:
- `data/predictions/{phase}/predictions_{run_id}.csv`
- `data/predictions/{phase}/feature_importance_{run_id}.csv`

### Stage 3 — Evaluation (`scripts/run_evaluation.py`)

Selects the top-30 stocks by predicted return each period and computes:

- Cumulative and annualised returns for the strategy and equal-weighted benchmark.
- Single-period and rolling 2-year outperformance probabilities.
- Mean rank information coefficient (Spearman IC between predicted and realised return).
- Worst and best single-period returns.

Results are appended as one row to `data/evaluation/runs_protocol.csv`, enabling
comparison across hyperparameter configurations (grid search is supported via
`GRID_SEARCH_CONFIGS` in `src/config.py`).

---

## Key design choices

| Choice | Rationale |
|---|---|
| Validation block placed *before* training window | Eliminates temporal leakage in tree-count selection |
| Point-in-time fundamentals (Sharadar SF1) | No look-ahead bias from restatements |
| Delisted firms included | Avoids survivorship bias |
| Prediction at `bdfr=1` (day after earnings release) | Targets post-announcement drift |
| 60-trading-day horizon | Matches quarterly reporting cadence; low turnover |
| Market-cap filter (~$100M) | Limits micro-cap illiquidity |

---

## Raw data

Place the following files in `data/raw/` before running Stage 1:

| File | Description |
|---|---|
| `prices.csv` | Daily closing prices for U.S.-listed equities (including delisted) |
| `SF1.csv` | Sharadar SF1 point-in-time fundamentals (ARQ and TTM) |
| `tickers.csv` | Ticker metadata (sector, exchange, etc.) |

These files are not tracked in the repository due to size and licensing.
