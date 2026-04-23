# Equity Price Prediction Using Fundamentals Data

This project predicts future equity price movement using accounting fundamentals, valuation-style features, and a custom walk-forward validation design.

## Project structure

```text
earnings_return_project/
├── data/
│   ├── raw/
│   │   ├── prices.csv
│   │   ├── SF1.csv
│   │   └── tickers.csv
│   └── processed/
├── results/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── build_processed_data.py
│   └── run_equity_price_prediction.py
└── src/
    ├── config.py
    ├── data_preparation.py
    ├── evaluation.py
    ├── feature_engineering.py
    └── modeling.py
```

## What the code is doing

### Stage 1: Build the processed data
The script `scripts/build_processed_data.py` creates the datasets used by the
rest of the project.

It:
1. reads raw daily prices,
2. removes unsuitable price histories,
3. creates future price columns for 20, 30, and 60 trading days,
4. computes the trailing 60-day return,
5. reads quarterly and TTM fundamentals,
6. merges in ticker metadata,
7. joins the fundamentals to daily prices,
8. carries each report forward until the next report,
9. computes time-from-report variables used later in the modeling stage.

The main outputs are:
- `data/processed/daily_prices_with_forward_prices.csv`
- `data/processed/fundamentals_snapshot.csv`
- `data/processed/daily_price_fundamentals_panel.csv`

### Stage 2: Build the modeling dataset
The script `scripts/run_equity_price_prediction.py` first converts the processed
panel into the final modeling table.

It:
1. defines the prediction target as `next_60_days_close / close`,
2. removes the unlabeled tail of the sample,
3. keeps positive revenue, net income, and market cap rows,
4. log-transforms selected variables,
5. builds market-relative and sector-relative valuation features,
6. builds lagged firm-history features,
7. scales selected features by market cap,
8. creates the final spread-style valuation signals used by the model.

The modeling dataset is saved to:
- `data/processed/model_dataset.csv`

### Stage 3: Run the custom walk-forward evaluation
For each monthly test window, the code:
1. defines a forward test month,
2. defines a historical training window,
3. defines an earlier validation window that comes before the training window,
4. uses that earlier validation block for early stopping and tree-count selection,
5. scores the forward month only after model selection is complete.

This means the project uses two separate evaluation layers:
- an earlier validation block for local model selection,
- and a later forward test month for out-of-sample evaluation.

### Stage 4: Evaluate the ranked predictions
After generating predictions, the code:
1. merges them with realized return decomposition,
2. ranks stocks by prediction within each month,
3. compares top-ranked predictions, the full universe, and bottom-ranked names,
4. summarizes realized forward returns by prediction-rank bucket,
5. saves the resulting tables under `results/tables/`.

## How to run

From the project root:

```bash
python scripts/build_processed_data.py
python scripts/run_equity_price_prediction.py
```

## Expected raw data locations

Place the raw files here:
- `data/raw/prices.csv`
- `data/raw/SF1.csv`
- `data/raw/tickers.csv`
