"""
Equity Return Prediction

What this file does
-------------------
This file creates the core datasets that connect daily prices to reported
fundamentals.

Step-by-step logic
------------------
1. Daily prices:
   - read ticker, date, and close,
   - keep only tickers that exist in the fundamentals source,
   - remove very short price histories,
   - create forward prices for 20, 30, and 60 trading days,
   - create the trailing 60-day return.
2. Fundamentals snapshot:
   - read Sharadar SF1,
   - keep quarterly ARQ observations as the report snapshot,
   - keep TTM ART observations for selected flow variables,
   - require datekey >= calendardate for the quarterly rows,
   - merge in ticker metadata,
   - enforce one row per ticker and report date so the later daily merge stays
     memory-safe and does not create a many-to-many explosion.
3. Daily panel:
   - merge report-day fundamentals into daily prices,
   - carry the latest report forward within each ticker,
   - compute days_from_report,
   - compute bdfr = business-day position from the report date,
   - rescale market cap away from the report date using the price ratio,
   - keep only selected bdfr checkpoints.

The output of this file is the cleaned panel that later feeds the feature
engineering and model evaluation stages.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd


PRICE_COLUMNS = ["ticker", "date", "close"]
SF1_COLUMNS = [
    "ticker",
    "datekey",
    "netinc",
    "ncf",
    "fcf",
    "gp",
    "revenue",
    "ebitda",
    "equity",
    "debt",
    "marketcap",
    "ev",
    "currentratio",
    "calendardate",
    "dimension",
    "netinccmn",
]
SF1_TTM_COLUMNS = ["ticker", "datekey", "netinccmn", "fcf"]
TICKER_COLUMNS = ["ticker", "exchange", "isdelisted", "sector", "industry", "lastpricedate"]
FUNDAMENTALS_OUTPUT_COLUMNS = [
    "ticker",
    "datekey",
    "sector",
    "netinc",
    "ncf",
    "fcf",
    "gp",
    "revenue",
    "ebitda",
    "equity",
    "debt",
    "marketcap",
    "netinccmn_ttm",
    "fcf_ttm",
    "ev",
    "currentratio",
    "isdelisted",
    "lastpricedate",
    "exchange",
]


# This function adds a forward price label for a chosen trading-day horizon.
def add_forward_price_column(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Add a per-ticker future close column and forward-fill the tail."""
    prices = prices.copy()
    column_name = f"next_{horizon_days}_days_close"
    prices[column_name] = prices.groupby("ticker")["close"].shift(-horizon_days)
    prices[column_name] = prices.groupby("ticker")[column_name].ffill()
    return prices


# This function computes the trailing return from the close horizon_days ago to today.
def add_trailing_return(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Add the trailing return over a chosen trading-day horizon."""
    prices = prices.copy()
    previous_close = prices.groupby("ticker")["close"].shift(horizon_days)
    prices[f"current_{horizon_days}_d_r"] = prices["close"] / previous_close
    return prices


# This function keeps only one observation for each ticker and report date.
def keep_latest_report_per_date(df: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    """Resolve duplicate ticker-date rows before merges to avoid many-to-many joins."""
    available_sort_columns = [column for column in sort_columns if column in df.columns]
    if available_sort_columns:
        df = df.sort_values(["ticker", "datekey", *available_sort_columns])
    else:
        df = df.sort_values(["ticker", "datekey"])
    return df.drop_duplicates(["ticker", "datekey"], keep="last").copy()


# This function creates the cleaned daily prices dataset used throughout the project.
def build_daily_prices_dataset(
    prices_csv: str | Path,
    sf1_csv: str | Path,
    label_horizon_days: int,
    extra_forward_price_horizons: list[int],
    minimum_price_history_days: int,
    sample_fraction: float = 1.0,
    sample_random_state: int = 123,
) -> pd.DataFrame:
    """Build the cleaned daily prices table with forward price labels."""
    sf1_tickers = pd.read_csv(sf1_csv, usecols=["ticker"])
    all_unique_tickers = sf1_tickers["ticker"].dropna().unique()
    if sample_fraction < 1.0:
        rng = pd.Series(all_unique_tickers)
        all_unique_tickers = rng.sample(frac=sample_fraction, random_state=sample_random_state).values
    valid_tickers = set(all_unique_tickers)

    # Stream through the full prices CSV, keeping only rows for valid tickers.
    # Encode ticker as an integer ID and date as YYYYMMDD int so that pandas
    # reads the compact temp file with int32/float32 columns (avoiding the
    # ~600 MB object-array datetime-inference allocation on the full dataset).
    ticker_to_id: dict[str, int] = {}
    id_to_ticker: list[str] = []

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(prices_csv, newline="", encoding="utf-8") as src, \
             os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            header = next(reader)
            col_idx = {col: header.index(col) for col in PRICE_COLUMNS}
            ti, di, ci = col_idx["ticker"], col_idx["date"], col_idx["close"]
            writer.writerow(["ticker_id", "date_int", "close"])
            for row in reader:
                t = row[ti]
                if t in valid_tickers:
                    if t not in ticker_to_id:
                        ticker_to_id[t] = len(id_to_ticker)
                        id_to_ticker.append(t)
                    date_int = int(row[di].replace("-", ""))
                    writer.writerow([ticker_to_id[t], date_int, row[ci]])

        prices = pd.read_csv(tmp_path, dtype={"ticker_id": "int32", "date_int": "int32", "close": "float32"})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    prices["ticker"] = [id_to_ticker[i] for i in prices["ticker_id"]]
    prices["date"] = pd.to_datetime(prices["date_int"].astype(str), format="%Y%m%d")
    prices = prices.drop(columns=["ticker_id", "date_int"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_history_lengths = prices.groupby("ticker")["date"].size()
    keep_tickers = ticker_history_lengths[ticker_history_lengths > minimum_price_history_days].index
    prices = prices[prices["ticker"].isin(keep_tickers)].copy()

    for horizon_days in [label_horizon_days, *extra_forward_price_horizons]:
        prices = add_forward_price_column(prices, horizon_days)

    prices = add_trailing_return(prices, label_horizon_days)
    prices = prices.dropna(subset=[f"next_{label_horizon_days}_days_close"]).reset_index(drop=True)
    return prices


# This function builds the compact report-date fundamentals table used in the daily merge.
def build_fundamentals_snapshot(
    sf1_csv: str | Path,
    tickers_csv: str | Path,
    sample_fraction: float = 1.0,
    sample_random_state: int = 123,
) -> pd.DataFrame:
    """Build a report-date fundamentals snapshot table from SF1 and ticker metadata."""
    sf1 = pd.read_csv(sf1_csv, usecols=SF1_COLUMNS)
    sf1["datekey"] = pd.to_datetime(sf1["datekey"])
    sf1["calendardate"] = pd.to_datetime(sf1["calendardate"])
    if sample_fraction < 1.0:
        unique_tickers = sf1["ticker"].dropna().unique()
        sampled_tickers = pd.Series(unique_tickers).sample(frac=sample_fraction, random_state=sample_random_state).values
        sf1 = sf1[sf1["ticker"].isin(set(sampled_tickers))].copy()
    sf1 = sf1.dropna(subset=["ticker"]).copy()

    quarterly = sf1[sf1["dimension"] == "ARQ"].drop(columns=["dimension"])
    quarterly = quarterly[quarterly["datekey"] >= quarterly["calendardate"]].drop(columns=["calendardate"])
    quarterly = keep_latest_report_per_date(quarterly, sort_columns=["marketcap", "ev"])

    ttm = sf1[sf1["dimension"] == "ART"].drop(columns=["dimension", "calendardate"])
    ttm = ttm[SF1_TTM_COLUMNS].rename(columns={"netinccmn": "netinccmn_ttm", "fcf": "fcf_ttm"})
    ttm = keep_latest_report_per_date(ttm, sort_columns=["netinccmn_ttm", "fcf_ttm"])

    ticker_metadata = pd.read_csv(tickers_csv, usecols=TICKER_COLUMNS + ["table"])
    ticker_metadata["lastpricedate"] = pd.to_datetime(ticker_metadata["lastpricedate"])
    ticker_metadata = ticker_metadata[ticker_metadata["table"] == "SF1"][TICKER_COLUMNS].copy()
    ticker_metadata = ticker_metadata.sort_values(["ticker", "lastpricedate"]).drop_duplicates(["ticker"], keep="last")

    fundamentals = quarterly.merge(ticker_metadata, on="ticker", how="inner", validate="many_to_one")
    fundamentals = fundamentals.merge(ttm, on=["ticker", "datekey"], how="inner", validate="one_to_one")
    fundamentals = fundamentals.sort_values(["ticker", "datekey"]).drop_duplicates(["ticker", "datekey"], keep="last")
    return fundamentals[FUNDAMENTALS_OUTPUT_COLUMNS].copy()


# This function merges daily prices with the latest available report and carries that report forward.
def build_daily_price_fundamentals_panel(
    daily_prices: pd.DataFrame,
    fundamentals_snapshot: pd.DataFrame,
    panel_business_day_positions: list[int],
) -> pd.DataFrame:
    """Create the daily panel that carries each report snapshot forward until the next report."""
    fundamentals = fundamentals_snapshot.rename(columns={"datekey": "date"}).copy()
    fundamentals = fundamentals.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    fundamentals["report_date"] = fundamentals["date"]

    panel = daily_prices.merge(fundamentals, on=["ticker", "date"], how="left", validate="many_to_one")

    report_rows = panel["report_date"].notna()
    report_value_columns = [
        column for column in fundamentals.columns if column not in {"ticker", "date", "report_date"}
    ]
    panel.loc[report_rows, report_value_columns] = panel.loc[report_rows, report_value_columns].fillna(0)

    panel = panel.sort_values(["ticker", "date"])
    tickers = panel["ticker"].copy()
    panel = panel.groupby("ticker", group_keys=False).ffill()
    panel["ticker"] = tickers.values
    panel = panel.dropna().reset_index(drop=True)

    if panel.empty:
        raise ValueError("build_daily_price_fundamentals_panel produced empty panel; check sample_fraction or input files")

    panel["days_from_report"] = (panel["date"] - panel["report_date"]).dt.days
    panel["bdfr"] = panel.groupby(["ticker", "report_date"])["date"].rank(method="first") - 1

    report_close = (
        panel.loc[panel["date"] == panel["report_date"], ["ticker", "report_date", "close"]]
        .rename(columns={"close": "close_on_report_date"})
        .drop_duplicates(["ticker", "report_date"])
    )

    panel = panel.merge(report_close, on=["ticker", "report_date"], how="inner", validate="many_to_one")
    panel = panel.rename(columns={"marketcap": "marketcap_on_report_date"})
    panel["marketcap"] = panel["marketcap_on_report_date"] * (panel["close"] / panel["close_on_report_date"])
    panel = panel.drop(columns=["marketcap_on_report_date", "close_on_report_date"])

    panel["bdfr"] = panel["bdfr"].astype(int)
    panel = panel[panel["bdfr"].isin(panel_business_day_positions)].copy()
    return panel.reset_index(drop=True)


# This function saves one dataframe to disk and creates the parent folder if needed.
def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a dataframe as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
