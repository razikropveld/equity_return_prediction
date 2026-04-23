"""
Equity Return Prediction

What this script does
---------------------
TSCV execution stage. Expects model_dataset.csv from data_preprocessing.py.

Step-by-step logic
------------------
1. Load data/processed/{phase}/model_dataset.csv.
2. Build the run config list (single run or grid over GRID_SEARCH_CONFIGS).
3. For each config:
   a. Generate a unique run_id (timestamp).
   b. Run the walk-forward TSCV procedure.
   c. Save predictions_{run_id}.csv under data/predictions/{phase}/.
   d. Save feature_importance_{run_id}.csv under data/predictions/{phase}/.
   e. Append a partial protocol row (modeling columns only, eval columns empty)
      to data/evaluation/runs_protocol.csv.

Evaluation is a separate step — run run_evaluation.py to fill the eval columns.
"""

import copy
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import src.config
import src.evaluation
import src.modeling

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


def _build_config_list() -> list[dict]:
    base = copy.deepcopy(src.config.MODEL_SETTINGS)
    if not src.config.GRID_SEARCH_CONFIGS:
        return [base]
    configs = []
    for override in src.config.GRID_SEARCH_CONFIGS:
        cfg = copy.deepcopy(base)
        for key, value in override.items():
            if key in cfg["xgb_params"]:
                cfg["xgb_params"][key] = value
            else:
                cfg[key] = value
        configs.append(cfg)
    return configs


def _run_id_for(index: int, total: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{index:02d}" if total > 1 else ts


def _model_protocol_row(run_id: str, cfg: dict, best_iterations: list, n_periods: int) -> dict:
    """Build the modeling-stage columns of the protocol row (eval cols left None)."""
    xgb = cfg["xgb_params"]
    valid_iters = [i for i in best_iterations if i is not None]
    import numpy as np
    nbr_mean   = round(float(np.mean(valid_iters)),   1) if valid_iters else None
    nbr_median = round(float(np.median(valid_iters)), 1) if valid_iters else None
    return {
        "run_id":                         run_id,
        "eval_id":                        None,
        "datetime_model":                 datetime.now().isoformat(timespec="seconds"),
        "datetime_eval":                  None,
        "phase":                          src.config.PHASE_SETTINGS["phase"],
        "data_mode":                      src.config.PREPARATION_SETTINGS["data_mode"],
        "eta":                            xgb.get("eta"),
        "max_depth":                      xgb.get("max_depth"),
        "gamma":                          xgb.get("gamma"),
        "lambda_reg":                     xgb.get("lambda"),
        "num_boost_round":                cfg.get("num_boost_round"),
        "early_stopping_rounds":          cfg.get("early_stopping_rounds"),
        "validation_size":                cfg.get("validation_size"),
        "time_weights_lambda":            cfg.get("time_weights_lambda"),
        "train_max_days":                 cfg.get("train_max_days"),
        "train_max_size":                 cfg.get("train_max_size"),
        "zscore_inputs_and_label":        cfg.get("zscore_inputs_and_label"),
        "prediction_model_source":        cfg.get("prediction_model_source"),
        "validation_before_train":        cfg.get("validation_before_train"),
        "remove_validation_tickers_from_train": cfg.get("remove_validation_tickers_from_train"),
        "minimum_marketcap_log":          cfg.get("minimum_marketcap_log"),
        "period_start":                   cfg.get("period_start"),
        "period_end":                     cfg.get("period_end"),
        "n_periods":                      n_periods,
        "optimal_num_boost_rounds_mean":  nbr_mean,
        "optimal_num_boost_rounds_median": nbr_median,
        # Eval columns — filled later by run_evaluation.py
        "top_n":                          None,
        "cum_return_strategy":            None,
        "cum_return_benchmark":           None,
        "ann_return_strategy":            None,
        "ann_return_benchmark":           None,
        "prob_beat_bench_1q":             None,
        "min_q_90pct_beat_bench":         None,
        "max_q_negative_worst_return":    None,
        "worst_1q_return":                None,
        "spearman_ic_mean":               None,
    }


def main() -> None:
    global _start_time, _last_time
    _start_time = None
    _last_time = None

    _log("modeling_and_prediction.py is running")

    phase = src.config.PHASE_SETTINGS["phase"]

    # ── Load model dataset ────────────────────────────────────────────────────
    model_dataset_path = src.config.PROCESSED_DATA_DIR / phase / "model_dataset.csv"
    if not model_dataset_path.exists():
        raise FileNotFoundError(
            f"model_dataset not found at {model_dataset_path}. "
            "Run data_preprocessing.py first."
        )
    _log(f"Loading model_dataset from {model_dataset_path}")
    model_dataset = pd.read_csv(model_dataset_path, parse_dates=["date"])
    model_dataset = model_dataset.set_index("date")
    _log(f"Loaded {len(model_dataset):,} rows  |  phase={phase}")

    # ── Build config list ─────────────────────────────────────────────────────
    configs = _build_config_list()
    _log(f"Running {len(configs)} config(s)")

    # Output directories
    pred_phase_dir = src.config.PREDICTIONS_DIR / phase
    pred_phase_dir.mkdir(parents=True, exist_ok=True)

    # ── Grid loop ─────────────────────────────────────────────────────────────
    for i, cfg in enumerate(configs):
        run_id = _run_id_for(i, len(configs))
        _log(f"--- Run {i+1}/{len(configs)}  run_id={run_id} ---")

        (
            test_preds_df,
            fitted_models,
            best_iterations,
            validation_scores,
            _val_preds_df,
            feature_importance_df,
        ) = src.modeling.run_walk_forward_evaluation(model_dataset, cfg)

        n_periods = len(fitted_models)
        avg_iter  = pd.Series(best_iterations).dropna().astype(float).mean()
        avg_score = pd.Series(validation_scores).dropna().astype(float).mean()
        _log(f"TSCV done: {n_periods} periods | avg best_iter={avg_iter:.1f} | avg val_score={avg_score:.4f}")

        # ── Save predictions ──────────────────────────────────────────────────
        preds_path = pred_phase_dir / f"predictions_{run_id}.csv"
        src.evaluation.save_dataframe(test_preds_df.reset_index(), preds_path)
        _log(f"Saved predictions -> {preds_path.name}")

        # ── Save feature importance ───────────────────────────────────────────
        fi_path = pred_phase_dir / f"feature_importance_{run_id}.csv"
        src.evaluation.save_dataframe(feature_importance_df, fi_path)
        _log(f"Saved feature_importance -> {fi_path.name}")

        # ── Write partial protocol row ────────────────────────────────────────
        row = _model_protocol_row(run_id, cfg, best_iterations, n_periods)
        src.evaluation.append_to_protocol(src.config.RESULT_FILES["protocol"], row)
        _log(f"Protocol row (modeling) appended -> {src.config.RESULT_FILES['protocol'].name}")
        _log(f"Next step: run run_evaluation.py with RUN_ID='{run_id}'")

    _log("modeling_and_prediction.py ran successfully")


if __name__ == "__main__":
    main()
