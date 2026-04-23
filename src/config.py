"""
Equity Return Prediction

What this file defines
----------------------
Project folders, input/output file paths, and all experiment settings.

Project workflow
----------------
1. data_preprocessing.py   — raw -> processed -> data/processed/{phase}/model_dataset.csv
2. modeling_and_prediction.py — TSCV -> data/predictions/{phase}/predictions_{run_id}.csv
3. run_evaluation.py        — eval metrics -> data/evaluation/runs_protocol.csv

Phase definition
----------------
- "train": rows with date <= 2011-12-31
- "test" : rows with date >= 2012-01-01

Custom validation design
------------------------
For each monthly test period:
- the test block is the forward month being evaluated,
- the train block is an earlier history window,
- the validation block is placed even earlier than the train block.
That validation block selects the effective num_boost_rounds via early stopping.
"""

from pathlib import Path


PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DATA_DIR  = DATA_DIR / "raw"

PROCESSED_DATA_DIR        = DATA_DIR / "processed"
PROCESSED_SAMPLE_DATA_DIR = PROCESSED_DATA_DIR / "sample"

PREDICTIONS_DIR = DATA_DIR / "predictions"
EVALUATION_DIR  = DATA_DIR / "evaluation"

RAW_FILES = {
    "prices":  RAW_DATA_DIR / "prices.csv",
    "sf1":     RAW_DATA_DIR / "SF1.csv",
    "tickers": RAW_DATA_DIR / "tickers.csv",
}

PROCESSED_FILES = {
    "daily_prices":      PROCESSED_DATA_DIR / "daily_prices_with_forward_prices.csv",
    "fundamentals_snapshot": PROCESSED_DATA_DIR / "fundamentals_snapshot.csv",
    "daily_panel":       PROCESSED_DATA_DIR / "daily_price_fundamentals_panel.csv",
    "prices_with_monthly_reference_returns": PROCESSED_DATA_DIR / "prices_with_monthly_reference_returns.csv",
}

RESULT_FILES = {
    "protocol": EVALUATION_DIR / "runs_protocol.csv",
}

PREPARATION_SETTINGS = {
    "data_mode":                  "full",     # "full" | "sample"
    "sample_fraction":            0.1,
    "sample_random_state":        123,
    "label_horizon_days":         60,
    "extra_forward_price_horizons": [20, 30],
    "minimum_price_history_days": 180,
    "panel_business_day_positions": [0, 1, 15, 30, 45, 60],
}

# Phase controls which date slice of the data is modeled.
# "train" = up to 2011-12-31 inclusive; "test" = 2012-01-01 onward.
PHASE_SETTINGS = {
    "phase": "train",
}

MODEL_SETTINGS = {
    "xgb_params": {
        "booster":    "gbtree",
        "eval_metric": "rmse",
        "eta":        0.02,
        "max_depth":  3,
        "gamma":      0.1,
        "objective":  "reg:squarederror",
        "lambda":     0.3,
    },
    "num_boost_round":              1000,
    "early_stopping_rounds":        30,
    "validation_size":              5000,
    "validation_and_test_bdfr":     [1],
    "validation_and_test_days_from_report": [1, 2, 3, 4, 5],
    "time_weights_lambda":          1,
    "train_max_days":               90,
    "train_max_size":               100000,
    "prediction_model_source":      "val",
    "remove_validation_tickers_from_train": False,
    "period_start":                 18,
    "period_end":                   -4,
    "zscore_inputs_and_label":      True,
    "validation_before_train":      True,
    "minimum_marketcap_log":        18.420680743952367,
}

# Evaluation strategy settings — can be changed independently of MODEL_SETTINGS
# and applied to any existing predictions file via run_evaluation.py.
EVAL_SETTINGS = {
    "top_n":               30,    # stocks picked long per quarter
    "max_holding_quarters": 20,   # horizon for holding-period analysis
}

# Grid search: each dict overrides specific MODEL_SETTINGS keys for one run.
# Empty list = single run with the default MODEL_SETTINGS above.
# Example: [{"eta": 0.01, "max_depth": 4}, {"eta": 0.05, "max_depth": 3}]
GRID_SEARCH_CONFIGS = []

for _folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, EVALUATION_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)
