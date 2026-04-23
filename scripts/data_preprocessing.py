"""
Equity Return Prediction

What this script does
---------------------
Builds all processed datasets up to and including the phase-split model_dataset.csv
that is consumed directly by modeling_and_prediction.py.

Step-by-step logic
------------------
1. Build the cleaned daily prices table with forward prices and the trailing 60-day return.
2. Build the report-date fundamentals snapshot from quarterly and TTM accounting data.
3. Merge them into the daily price-fundamentals panel.
4. Run feature engineering on the panel to produce the final model_dataset.
5. Apply the phase date filter and the minimum market-cap filter.
6. Save the model_dataset under data/processed/{phase}/model_dataset.csv.

Expected raw data paths
-----------------------
- data/raw/prices.csv
- data/raw/SF1.csv
- data/raw/tickers.csv
"""

import time
import pandas as pd

from src.config import (
    PREPARATION_SETTINGS,
    PHASE_SETTINGS,
    MODEL_SETTINGS,
    PROCESSED_FILES,
    PROCESSED_DATA_DIR,
    PROCESSED_SAMPLE_DATA_DIR,
    RAW_FILES,
)
from src.data_preparation import (
    build_daily_price_fundamentals_panel,
    build_daily_prices_dataset,
    build_fundamentals_snapshot,
    save_dataframe,
)
import src.feature_engineering

_start_time = None
_last_time = None


def _log(message: str) -> None:
    global _start_time, _last_time
    now = time.time()
    if _start_time is None:
        _start_time = now
        _last_time = now
    total_elapsed = now - _start_time
    step_elapsed = now - _last_time
    print(f"[{total_elapsed:.1f}s | +{step_elapsed:.1f}s] {message}", flush=True)
    _last_time = now


def main() -> None:
    """Build and save all processed datasets up to model_dataset.csv."""
    global _start_time, _last_time
    _start_time = None
    _last_time = None

    _log("data_preprocessing.py is running")
    data_mode = PREPARATION_SETTINGS.get("data_mode", "full").lower()
    sample_random_state = PREPARATION_SETTINGS.get("sample_random_state", 123)

    if data_mode == "sample":
        is_sample_mode = True
        sample_fraction = PREPARATION_SETTINGS.get("sample_fraction", 0.1)
    elif data_mode == "full":
        is_sample_mode = False
        sample_fraction = 1.0
    else:
        raise ValueError("PREPARATION_SETTINGS['data_mode'] must be 'full' or 'sample'")

    output_root = PROCESSED_SAMPLE_DATA_DIR if is_sample_mode else PROCESSED_DATA_DIR
    suffix = "_sample" if is_sample_mode else ""

    processed_files = {
        key: output_root / (path.name.replace(".csv", f"{suffix}.csv"))
        for key, path in PROCESSED_FILES.items()
    }

    # ── Step 1: daily prices ──────────────────────────────────────────────────
    daily_prices = build_daily_prices_dataset(
        prices_csv=RAW_FILES["prices"],
        sf1_csv=RAW_FILES["sf1"],
        label_horizon_days=PREPARATION_SETTINGS["label_horizon_days"],
        extra_forward_price_horizons=PREPARATION_SETTINGS["extra_forward_price_horizons"],
        minimum_price_history_days=PREPARATION_SETTINGS["minimum_price_history_days"],
        sample_fraction=sample_fraction,
        sample_random_state=sample_random_state,
    )
    _log("Exporting daily_prices")
    save_dataframe(daily_prices, processed_files["daily_prices"])
    _log("daily_prices exported successfully")

    # ── Step 2: fundamentals snapshot ─────────────────────────────────────────
    fundamentals_snapshot = build_fundamentals_snapshot(
        sf1_csv=RAW_FILES["sf1"],
        tickers_csv=RAW_FILES["tickers"],
        sample_fraction=sample_fraction,
        sample_random_state=sample_random_state,
    )
    _log("Exporting fundamentals_snapshot")
    save_dataframe(fundamentals_snapshot, processed_files["fundamentals_snapshot"])
    _log("fundamentals_snapshot exported successfully")

    # ── Step 3: daily panel ───────────────────────────────────────────────────
    try:
        daily_panel = build_daily_price_fundamentals_panel(
            daily_prices=daily_prices,
            fundamentals_snapshot=fundamentals_snapshot,
            panel_business_day_positions=PREPARATION_SETTINGS["panel_business_day_positions"],
        )
        _log("Exporting daily_panel")
        save_dataframe(daily_panel, processed_files["daily_panel"])
        _log("daily_panel exported successfully")
    except ValueError as e:
        _log(f"ERROR during panel building: {e}")
        if data_mode == "sample":
            _log("Sample mode may have produced too small a dataset. Try increasing sample_fraction.")
        raise

    # ── Step 4: feature engineering, phase + marketcap filter ────────────────
    phase = PHASE_SETTINGS["phase"]
    _log(f"Building model_dataset for phase='{phase}'")
    try:
        model_dataset = src.feature_engineering.build_model_dataset(daily_panel)
    except ValueError as e:
        _log(f"ERROR during feature engineering: {e}")
        raise

    model_dataset = model_dataset.reset_index()
    model_dataset["date"] = pd.to_datetime(model_dataset["date"])

    if phase == "train":
        model_dataset = model_dataset[model_dataset["date"] <= pd.Timestamp("2011-12-31")]
    elif phase == "test":
        model_dataset = model_dataset[model_dataset["date"] >= pd.Timestamp("2012-01-01")]
    else:
        raise ValueError(f"PHASE_SETTINGS['phase'] must be 'train' or 'test', got '{phase}'")

    model_dataset = model_dataset[
        model_dataset["marketcap"] >= MODEL_SETTINGS["minimum_marketcap_log"]
    ].copy()

    phase_dir = PROCESSED_DATA_DIR / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    model_dataset_path = phase_dir / "model_dataset.csv"
    save_dataframe(model_dataset, model_dataset_path)
    _log(f"model_dataset exported ({len(model_dataset):,} rows) -> {model_dataset_path}")

    _log("Saved processed files:")
    print(f"  {processed_files['daily_prices']}")
    print(f"  {processed_files['fundamentals_snapshot']}")
    print(f"  {processed_files['daily_panel']}")
    print(f"  {model_dataset_path}")
    _log("data_preprocessing.py ran successfully")


if __name__ == "__main__":
    main()
