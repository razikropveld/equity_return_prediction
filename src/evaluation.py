
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

TOP_N = 15
ROLLING_SHARPE_WINDOW = 12

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def build_quarterly_portfolios(df):

    df['quarter'] = df['date'].dt.to_period("Q")

    portfolios = []

    for q, group in df.groupby('quarter'):

        group = group.sort_values('pred')

        short_leg = group.head(TOP_N)
        long_leg = group.tail(TOP_N)

        long_return = long_leg['label'].mean()
        short_return = short_leg['label'].mean()

        portfolios.append({
            "date": group['date'].max(),
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": long_return - short_return
        })

    port = pd.DataFrame(portfolios).sort_values("date")

    return port


def build_benchmark(df):

    bench = (
        df.groupby("date")["label"]
        .mean()
        .reset_index()
        .rename(columns={"label": "benchmark_return"})
    )

    return bench


def compute_ic(df):

    ic_series = []

    for date, group in df.groupby("date"):

        if group["pred"].nunique() < 2:
            continue

        ic, _ = spearmanr(group["pred"], group["label"])

        ic_series.append({
            "date": date,
            "IC": ic
        })

    ic_df = pd.DataFrame(ic_series).sort_values("date")

    return ic_df


def rolling_sharpe(returns, window):

    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()

    sharpe = np.sqrt(4) * mean / std

    return sharpe


def cumulative_wealth(returns):

    return (1 + returns).cumprod()


def plot_cumulative(port, benchmark):

    merged = pd.merge(port, benchmark, on="date", how="left")

    merged["wealth_strategy"] = cumulative_wealth(merged["long_return"])
    merged["wealth_benchmark"] = cumulative_wealth(merged["benchmark_return"])

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(
        merged["date"],
        merged["wealth_strategy"],
        label="Top 15 Strategy",
        linewidth=2
    )

    ax.plot(
        merged["date"],
        merged["wealth_benchmark"],
        label="Benchmark",
        linestyle="--",
        linewidth=2
    )

    ax.set_yscale("log")
    ax.set_ylabel("Cumulative Wealth (log scale)")
    ax.set_title("Cumulative Wealth")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_ic(ic_df):

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(ic_df["date"], ic_df["IC"], label="Information Coefficient")

    ax.axhline(0, color="black", linewidth=1)

    ax.set_title("IC Over Time")
    ax.set_ylabel("Spearman IC")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(port, benchmark):

    merged = pd.merge(port, benchmark, on="date", how="left")

    merged["sharpe_strategy"] = rolling_sharpe(
        merged["long_return"], ROLLING_SHARPE_WINDOW
    )

    merged["sharpe_benchmark"] = rolling_sharpe(
        merged["benchmark_return"], ROLLING_SHARPE_WINDOW
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        merged["date"],
        merged["sharpe_strategy"],
        label="Strategy Sharpe"
    )

    ax.plot(
        merged["date"],
        merged["sharpe_benchmark"],
        label="Benchmark Sharpe",
        linestyle="--"
    )

    ax.set_title("Rolling Sharpe Ratio")
    ax.set_ylabel("Sharpe")

    ax.legend()

    plt.tight_layout()
    plt.show()


def run_evaluation(path):

    df = load_data(path)

    portfolios = build_quarterly_portfolios(df)

    benchmark = build_benchmark(df)

    ic = compute_ic(df)

    plot_cumulative(portfolios, benchmark)
    plot_ic(ic)
    plot_rolling_sharpe(portfolios, benchmark)


def save_dataframe(df: pd.DataFrame, path) -> None:
    """Save a dataframe to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compute_eval_metrics(
    test_preds_df: pd.DataFrame,
    best_iterations: list,
    top_n: int,
    max_holding_quarters: int = 20,
) -> dict:
    """Compute protocol-table evaluation metrics from the TSCV results dataframe.

    The input dataframe must have columns: date, label, pred.
    'label' is the gross quarterly return (e.g. 1.05 = +5%).
    """
    df = test_preds_df.copy()
    df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})

    # Convert log-label back to gross return if it looks like log-returns
    # (evaluation.ipynb applies np.exp to label on load).
    df["label"] = np.exp(df["label"])

    df["quarter"] = df["date"].dt.to_period("Q")

    # Build quarterly long portfolio and benchmark returns
    quarterly_rows = []
    for q, group in df.groupby("quarter"):
        group = group.sort_values("pred")
        long_ret = group.tail(top_n)["label"].mean()
        bench_ret = group["label"].mean()
        quarterly_rows.append({
            "quarter": q,
            "date": group["date"].max(),
            "long_return": long_ret,
            "bench_return": bench_ret,
        })
    port = pd.DataFrame(quarterly_rows).sort_values("date")
    n = len(port)

    if n == 0:
        return {}

    long_r = port["long_return"].values
    bench_r = port["bench_return"].values
    rel_r = long_r / bench_r

    # Cumulative and annualised returns
    cum_strategy = float(np.prod(long_r))
    cum_benchmark = float(np.prod(bench_r))
    ann_strategy = float(cum_strategy ** (4 / n) - 1)
    ann_benchmark = float(cum_benchmark ** (4 / n) - 1)

    # Holding-period analysis (mirrors evaluation.ipynb graphs 1, 2)
    xs = list(range(1, min(max_holding_quarters, n - 1) + 1))
    ns_obs = [n - x + 1 for x in xs]

    prob_beat_bench_1q = float(np.mean(long_r > bench_r) * 100)

    min_q_90pct = None
    max_q_negative_worst = 0
    for x, n_obs in zip(xs, ns_obs):
        wins = sum(
            1 for t in range(n_obs)
            if np.prod(long_r[t:t+x]) > np.prod(bench_r[t:t+x])
        )
        prob = wins / n_obs * 100
        if min_q_90pct is None and prob >= 90:
            min_q_90pct = x

        worst = min((np.prod(long_r[t:t+x]) - 1) * 100 for t in range(n_obs))
        if worst < 0:
            max_q_negative_worst = x

    # Worst single-quarter return of the strategy (%)
    worst_1q_return = float((np.min(long_r) - 1) * 100)

    # Mean Spearman IC across quarters
    ic_values = []
    for _, group in df.groupby("quarter"):
        if group["pred"].nunique() >= 2:
            ic, _ = spearmanr(group["pred"], group["label"])
            if not np.isnan(ic):
                ic_values.append(ic)
    spearman_ic_mean = float(np.mean(ic_values)) if ic_values else float("nan")

    return {
        "cum_return_strategy":         round(cum_strategy,      4),
        "cum_return_benchmark":        round(cum_benchmark,     4),
        "ann_return_strategy":         round(ann_strategy,      4),
        "ann_return_benchmark":        round(ann_benchmark,     4),
        "prob_beat_bench_1q":          round(prob_beat_bench_1q, 2),
        "min_q_90pct_beat_bench":      min_q_90pct,
        "max_q_negative_worst_return": max_q_negative_worst,
        "worst_1q_return":             round(worst_1q_return,   2),
        "spearman_ic_mean":            round(spearman_ic_mean,  4) if not np.isnan(spearman_ic_mean) else None,
    }


def append_to_protocol(protocol_path, row_dict: dict) -> None:
    """Append one row to the protocol CSV. Creates file with header if it does not exist."""
    protocol_path = Path(protocol_path)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row_dict])
    if protocol_path.exists():
        existing = pd.read_csv(protocol_path)
        for col in row_df.columns:
            if col not in existing.columns:
                existing[col] = None
        for col in existing.columns:
            if col not in row_df.columns:
                row_df[col] = None
        row_df = row_df[existing.columns]
        pd.concat([existing, row_df], ignore_index=True).to_csv(protocol_path, index=False)
    else:
        row_df.to_csv(protocol_path, index=False)


def upsert_eval_in_protocol(protocol_path, run_id: str, eval_cols: dict) -> str:
    """Fill evaluation columns for a run_id in the protocol table.

    Behaviour:
    - If the run_id has a row where eval_id is null/empty → update that row in-place.
    - If all rows for the run_id already have an eval_id (a prior eval was saved) →
      append a new row copying the model columns and filling the eval columns.
      This lets you compare multiple evaluation strategies on the same predictions.

    Returns the eval_id that was assigned.
    """
    from datetime import datetime as _dt
    protocol_path = Path(protocol_path)
    eval_id = _dt.now().strftime("%Y%m%d_%H%M%S")

    if not protocol_path.exists():
        # Nothing to update — just write a row with eval cols only
        row = {"run_id": run_id, "eval_id": eval_id, **eval_cols}
        pd.DataFrame([row]).to_csv(protocol_path, index=False)
        return eval_id

    df = pd.read_csv(protocol_path)

    run_rows = df[df["run_id"].astype(str) == str(run_id)]
    if run_rows.empty:
        row = {"run_id": run_id, "eval_id": eval_id, **eval_cols}
        append_to_protocol(protocol_path, row)
        return eval_id

    # Find a row where eval_id is missing (the model-only partial row)
    unevaluated_mask = (
        run_rows["eval_id"].isna() |
        (run_rows["eval_id"].astype(str).str.strip() == "") |
        (run_rows["eval_id"].astype(str) == "None")
    ) if "eval_id" in run_rows.columns else pd.Series([True] * len(run_rows), index=run_rows.index)

    if unevaluated_mask.any():
        # Update the first unevaluated row
        target_idx = run_rows[unevaluated_mask].index[0]
        df.loc[target_idx, "eval_id"] = eval_id
        from datetime import datetime as _dt2
        df.loc[target_idx, "datetime_eval"] = _dt2.now().isoformat(timespec="seconds")
        for col, val in eval_cols.items():
            df.loc[target_idx, col] = val
        df.to_csv(protocol_path, index=False)
    else:
        # All rows already have eval_id — append a new complete row
        source_row = run_rows.iloc[0].to_dict()
        source_row.update({"eval_id": eval_id, **eval_cols})
        source_row["datetime_eval"] = _dt.now().isoformat(timespec="seconds")
        append_to_protocol(protocol_path, source_row)

    return eval_id


if __name__ == "__main__":

    run_evaluation("train_and_test_results.csv")
