"""
Equity Return Prediction

What this script does
---------------------
Evaluation stage — runs independently of TSCV and can be re-run with different
EVAL_SETTINGS on the same predictions file.

Step-by-step logic
------------------
1. Resolve the predictions file:
   - if RUN_ID is set, load data/predictions/{phase}/predictions_{RUN_ID}.csv,
   - otherwise use the most recent file in that folder.
2. Compute all evaluation metrics using EVAL_SETTINGS.
3. Update the protocol table:
   - if the existing row for this run_id has no eval columns filled, update it,
   - otherwise append a new row with a fresh eval_id (allows multiple strategies).
4. Print a summary.

To evaluate with a different strategy, change EVAL_SETTINGS in src/config.py
and re-run this script. Each distinct strategy produces its own protocol row.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import src.config
import src.evaluation

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


# Set RUN_ID to target a specific run, or leave "" to use the most recent.
RUN_ID = ""


def _resolve_predictions_path(run_id: str, phase: str) -> Path:
    pred_dir = src.config.PREDICTIONS_DIR / phase
    if run_id:
        p = pred_dir / f"predictions_{run_id}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Predictions file not found: {p}")
        return p
    candidates = sorted(pred_dir.glob("predictions_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No predictions files found in {pred_dir}. "
            "Run modeling_and_prediction.py first."
        )
    return candidates[-1]


def _extract_run_id_from_path(path: Path) -> str:
    return path.stem.replace("predictions_", "")


def main() -> None:
    global _start_time, _last_time
    _start_time = None
    _last_time = None

    _log("run_evaluation.py is running")

    phase    = src.config.PHASE_SETTINGS["phase"]
    top_n    = src.config.EVAL_SETTINGS["top_n"]
    max_hold = src.config.EVAL_SETTINGS["max_holding_quarters"]

    # ── Load predictions ──────────────────────────────────────────────────────
    preds_path = _resolve_predictions_path(RUN_ID, phase)
    run_id = _extract_run_id_from_path(preds_path)
    _log(f"Loading predictions: {preds_path.name}  (run_id={run_id})")

    preds_df = pd.read_csv(preds_path)
    _log(f"Loaded {len(preds_df):,} rows  |  top_n={top_n}")

    # ── Compute evaluation metrics ────────────────────────────────────────────
    # best_iterations not available here; pass empty list (nbr stats already in model row)
    metrics = src.evaluation.compute_eval_metrics(
        preds_df,
        best_iterations=[],
        top_n=top_n,
        max_holding_quarters=max_hold,
    )
    _log(
        f"cum_return_strategy={metrics.get('cum_return_strategy')}  "
        f"ann={metrics.get('ann_return_strategy', 0)*100:.1f}%  "
        f"IC={metrics.get('spearman_ic_mean')}"
    )

    # ── Update or append protocol row ─────────────────────────────────────────
    eval_cols = {
        "top_n":                       top_n,
        "cum_return_strategy":         metrics.get("cum_return_strategy"),
        "cum_return_benchmark":        metrics.get("cum_return_benchmark"),
        "ann_return_strategy":         metrics.get("ann_return_strategy"),
        "ann_return_benchmark":        metrics.get("ann_return_benchmark"),
        "prob_beat_bench_1q":          metrics.get("prob_beat_bench_1q"),
        "min_q_90pct_beat_bench":      metrics.get("min_q_90pct_beat_bench"),
        "max_q_negative_worst_return": metrics.get("max_q_negative_worst_return"),
        "worst_1q_return":             metrics.get("worst_1q_return"),
        "spearman_ic_mean":            metrics.get("spearman_ic_mean"),
    }

    protocol_path = src.config.RESULT_FILES["protocol"]
    eval_id = src.evaluation.upsert_eval_in_protocol(
        protocol_path, run_id, eval_cols
    )
    _log(
        f"Protocol updated (eval_id={eval_id}) -> {protocol_path.name}\n"
        f"  To evaluate with a different strategy: change EVAL_SETTINGS and re-run."
    )

    _log("run_evaluation.py ran successfully")


if __name__ == "__main__":
    main()
