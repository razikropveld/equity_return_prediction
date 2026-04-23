"""
Equity Return Prediction

What this file does
-------------------
This file converts the daily price-fundamentals panel into the modeling dataset
used by the prediction stage.

Step-by-step logic
------------------
1. Create the prediction label:
   - label = next_60_days_close / close.
2. Remove the final unlabeled tail of the sample.
3. Keep rows with positive revenue, positive net income, and positive market cap.
4. Apply transformations:
   - log(label) and log(netinc), dropping invalid rows,
   - log-transform marketcap, revenue, fcf, and current_60_d_r, replacing invalid values with 0,
   - encode sector as integers.
5. Build cross-sectional valuation references:
   - total-market ratios of net income and FCF relative to total market cap,
   - sector-level ratios of net income and FCF relative to sector market cap.
6. Build firm-history features:
   - previous net income and previous FCF at the retained panel checkpoints,
   - log-transform those lagged features and their change ratios.
7. Scale selected firm-level quantities by market cap.
8. Build the final spread-style signals:
   - earnings yield deviation from the market,
   - earnings yield deviation from sector,
   - FCF yield deviation from the market,
   - FCF yield deviation from sector,
   - a combined earnings-and-FCF signal adjusted by sector spread.

Meaning of the main engineered features
--------------------------------------
- netinc_ratio_total / fcf_ratio_total:
  the market-wide earnings or FCF amount per unit of market cap on that date.
- netinc_ratio_sector / fcf_ratio_sector:
  the same quantity, but only within the firm's sector.
- ep_diff_from_total / fcfp_diff_from_total:
  how cheap or expensive the company looks relative to the full market.
- ep_diff_from_sector / fcfp_diff_from_sector:
  how cheap or expensive the company looks relative to its own sector.
- ep_fcfp_add_sub_sector:
  a combined signal that rewards cheap earnings and FCF versus the market,
  while subtracting the sector-relative component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_COLUMNS = [
    "ticker",
    "label",
    "days_from_report",
    "bdfr",
    "fcf",
    "netinc",
    "revenue",
    "sector",
    "marketcap",
    "netinc_ratio_total",
    "fcf_ratio_total",
    "netinc_ratio_sector",
    "fcf_ratio_sector",
    "netinc_prev_1",
    "fcf_prev_1",
    "ep_diff_from_total",
    "ep_sector_diff_total",
    "fcfp_diff_from_total",
    "fcfp_sector_diff_total",
    "ep_fcfp_add_sub_sector",
    "current_60_d_r",
]


# This function applies a log transform and replaces invalid values with zero.
def log_with_zero_fill(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Log-transform selected columns and replace invalid values with 0."""
    df = df.copy()
    df[columns] = np.log(df[columns])
    df[columns] = df[columns].replace([-np.inf, np.inf, np.nan], 0)
    return df


# This function applies a log transform and drops rows where the result is invalid.
def log_and_drop_invalid_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Log-transform selected columns and drop rows with invalid transformed values."""
    df = df.copy()
    df[columns] = np.log(df[columns])
    df[columns] = df[columns].replace([-np.inf, np.inf], np.nan)
    return df.dropna(subset=columns)


# This function converts categorical columns to integer labels.
def encode_categorical_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encode categorical columns as integer ids."""
    df = df.copy()
    for column in columns:
        df[column] = df[column].fillna("none")
        mapping = {value: index for index, value in enumerate(df[column].unique())}
        df[column] = df[column].map(mapping).astype(int)
    return df


# This function adds market-wide net income and FCF ratios for each date.
def add_market_reference_ratios(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Add date-level market reference ratios for selected flow variables."""
    by_date = df.groupby("date")[feature_columns + ["marketcap"]].sum()
    ratios = by_date[feature_columns].div(by_date["marketcap"], axis=0)
    ratios.columns = [f"{column}_ratio_total" for column in ratios.columns]
    return df.merge(ratios, on="date", how="left")


# This function adds sector-level net income and FCF ratios for each date and sector.
def add_sector_reference_ratios(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Add date-and-sector reference ratios for selected flow variables."""
    by_date_sector = df.groupby(["date", "sector"])[feature_columns + ["marketcap"]].sum()
    ratios = by_date_sector[feature_columns].div(by_date_sector["marketcap"], axis=0)
    ratios.columns = [f"{column}_ratio_sector" for column in ratios.columns]
    ratios = ratios.reset_index()
    return df.merge(ratios, on=["date", "sector"], how="left")


# This function adds a previous value and change ratio for a feature 
# within each ticker history.
def add_lagged_feature(df: pd.DataFrame, column: str, lag_steps: int) -> pd.DataFrame:
    """Add a lagged feature and a ratio versus that lagged value for each ticker."""
    df = df.sort_values(["ticker", "date"]).copy()
    previous_column = f"{column}_prev_{lag_steps}"
    change_column = f"{column}_change_from_previous_{lag_steps}"
    df[previous_column] = df.groupby("ticker")[column].shift(lag_steps)
    df[change_column] = df[column] / df[previous_column]
    return df


# This function divides selected columns by market cap.
def scale_columns_by_marketcap(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scale selected firm-level quantities by market cap."""
    df = df.copy()
    for column in columns:
        if column != "marketcap":
            df[column] = df[column] / df["marketcap"]
    return df


# This function creates monthly reference returns used later in the evaluation stage.
def add_monthly_reference_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add return-to-end-of-month and end-of-month-to-horizon returns."""
    prices = prices.copy()
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp("M")

    close_at_month_end = (
        prices.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"])["close"]
        .last()
        .rename("close_eom")
        .reset_index()
    )

    prices = prices.merge(close_at_month_end, on=["ticker", "month"], how="left")
    prices["return_to_eom"] = prices["close_eom"] / prices["close"]
    prices["return_eom_to_60"] = prices["next_60_days_close"] / prices["close_eom"]
    return prices


# This function applies the full feature-engineering pipeline and returns the final modeling table.
def build_model_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the final modeling dataset from the daily price-fundamentals panel."""
    df = panel.copy()
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("Input panel is empty. Try increasing sample_fraction or check earlier pipeline steps.")
    # Check for 'date' column
    if "date" not in df.columns:
        raise ValueError("Input panel is missing 'date' column. Check sample_fraction or earlier pipeline steps.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).set_index("date")

    all_dates = sorted(df.index.unique())
    if len(all_dates) < 61:
        raise ValueError(f"Not enough dates in panel for modeling ({len(all_dates)} found, need at least 61). Try increasing sample_fraction.")
    last_labeled_date = all_dates[-60]

    df["label"] = df["next_60_days_close"] / df["close"]
    df = df.loc[df.index < last_labeled_date].copy()
    df = df[(df["revenue"] > 0) & (df["netinc"] > 0) & (df["marketcap"] > 0)].copy()

    df = log_and_drop_invalid_rows(df, ["label", "netinc"])
    df = log_with_zero_fill(df, ["marketcap", "revenue", "fcf", "current_60_d_r"])
    df = encode_categorical_columns(df, ["sector"])

    df = add_market_reference_ratios(df, ["netinc", "fcf"])
    df = add_sector_reference_ratios(df, ["netinc", "fcf"])

    df = add_lagged_feature(df.reset_index(), "netinc", 1)
    df = add_lagged_feature(df, "fcf", 1)
    lagged_columns = [
        "netinc_prev_1",
        "netinc_change_from_previous_1",
        "fcf_prev_1",
        "fcf_change_from_previous_1",
    ]
    df[lagged_columns] = np.log(df[lagged_columns]).replace([-np.inf, np.inf, np.nan], 0)

    df = scale_columns_by_marketcap(df, ["marketcap", "netinc", "fcf", "netinc_prev_1", "fcf_prev_1"])

    df["ep_diff_from_total"] = df["netinc"] - df["netinc_ratio_total"]
    df["ep_diff_from_sector"] = df["netinc"] - df["netinc_ratio_sector"]
    df["fcfp_diff_from_total"] = df["fcf"] - df["fcf_ratio_total"]
    df["fcfp_diff_from_sector"] = df["fcf"] - df["fcf_ratio_sector"]

    df["ep_sector_diff_total"] = df["ep_diff_from_sector"] - df["ep_diff_from_total"]
    df["fcfp_sector_diff_total"] = df["fcfp_diff_from_sector"] - df["fcfp_diff_from_total"]
    df["ep_fcfp_add_sub_sector"] = (
        df["ep_diff_from_total"]
        + df["fcfp_diff_from_total"]
        - (df["ep_sector_diff_total"] + df["fcfp_sector_diff_total"])
    )

    df = df.sort_values(["date", "ticker"])
    df = df.set_index("date")[MODEL_COLUMNS].copy()
    return df
