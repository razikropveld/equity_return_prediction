"""
Equity Return Prediction

What this file does
-------------------
This file trains the XGBoost model and runs the custom walk-forward evaluation.

Meaning of the custom train-validation-test design
--------------------------------------------------
For each monthly test batch:
1. Test block:
   - one forward month of observations.
2. Train block:
   - a historical window ending 61 trading days before the test period.
3. Earlier validation block:
   - a block placed before the train window,
   - restricted to the same report-timing subset used for testing,
   - used for early stopping and effective num_boost_round selection.
4. Final prediction:
   - either reuse the model selected on the earlier validation block,
   - or refit using the chosen number of boosting rounds and then score the test month.

This creates two selection layers:
- local tree-count selection on the earlier validation block,
- true forward performance measurement on the later monthly test blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import zscore
from joblib import Parallel, delayed


# This function gives smaller weight to older samples and larger weight to newer ones.
def compute_time_decay_weights(indexed_features: pd.DataFrame, time_weights_lambda: float) -> pd.Series:
    """Compute inverse-square-root time-decay weights."""
    dates = indexed_features.reset_index()["date"]
    age_in_days = (dates.max() - dates).dt.days + 1
    return 1 / np.sqrt(age_in_days + time_weights_lambda)


# This function creates one test batch per calendar month.
def generate_monthly_test_windows(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create month-start to month-end test windows covering the sample."""
    first_month = df.index.min().to_period("M").to_timestamp()
    month_starts = pd.date_range(first_month, df.index.max(), freq="MS")
    month_ends = pd.date_range(month_starts.min(), periods=len(month_starts), freq="M")
    return list(zip(month_starts, month_ends))


# This function optionally removes validation tickers from the training set.
def remove_validation_tickers_from_training(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Remove validation tickers from the training fold."""
    validation_tickers = set(validation["ticker"].unique())
    return train[~train["ticker"].isin(validation_tickers)].copy()


# This function separates predictors from the target while keeping date and ticker identifiers.
def split_features_and_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the modeling dataframe into X and y."""
    indexed = df.reset_index().set_index(["date", "ticker"])
    X = indexed.drop(columns=["label", "bdfr"])
    y = indexed["label"]
    return X, y


# This function converts one data split into an XGBoost DMatrix.
def build_dmatrix(
    df: pd.DataFrame,
    apply_weights: bool,
    zscore_inputs_and_label: bool,
    time_weights_lambda: float,
) -> xgb.DMatrix:
    """Convert one dataframe split into an XGBoost DMatrix."""
    X, y = split_features_and_label(df)

    if zscore_inputs_and_label:
        float_columns = [column for column in X.columns if column not in ["days_from_report", "sector"]]
        if float_columns:
            X.loc[:, float_columns] = zscore(X[float_columns], nan_policy="omit")
        y = pd.Series(zscore(y, nan_policy="omit"), index=y.index)

    X = X.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    y = y.replace([np.inf, -np.inf], 0).fillna(0)

    if apply_weights:
        weights = compute_time_decay_weights(X, time_weights_lambda).values * np.abs(y.values)
        return xgb.DMatrix(X, label=y.values, weight=weights)

    return xgb.DMatrix(X, label=y.values)


# This function trains on the train block and uses the earlier validation block for early stopping.
def train_with_early_stopping(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    zscore_inputs_and_label: bool,
    time_weights_lambda: float,
) -> tuple[xgb.Booster, int | None, float | None, np.ndarray]:
    """Train the model and choose the effective number of trees using validation."""
    train_dmatrix = build_dmatrix(train, True, zscore_inputs_and_label, time_weights_lambda)
    validation_dmatrix = build_dmatrix(validation, True, zscore_inputs_and_label, time_weights_lambda)

    model = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=num_boost_round,
        evals=[(train_dmatrix, "train"), (validation_dmatrix, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    validation_predictions = model.predict(validation_dmatrix)
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return model, best_iteration, best_score, validation_predictions


# This function scores the forward test month either with the validated model or with a refitted model.
def predict_test_window(
    validated_model: xgb.Booster,
    train_for_test: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
    prediction_model_source: str,
    zscore_inputs_and_label: bool,
    time_weights_lambda: float,
) -> tuple[np.ndarray, xgb.Booster]:
    """Generate predictions for the forward test month."""
    train_dmatrix = build_dmatrix(train_for_test, True, zscore_inputs_and_label, time_weights_lambda)
    test_dmatrix = build_dmatrix(test, True, zscore_inputs_and_label, time_weights_lambda)

    if prediction_model_source == "test":
        best_iteration = max(1, getattr(validated_model, "best_iteration", 1))
        refit_model = xgb.train(params, train_dmatrix, num_boost_round=best_iteration)
        return refit_model.predict(test_dmatrix), refit_model

    return validated_model.predict(test_dmatrix), validated_model


# This function creates the train, earlier-validation, and forward-test splits for one month and runs the model.
def run_one_walk_forward_window(
    available_history: pd.DataFrame,
    evaluation_subset: pd.DataFrame,
    all_dates: pd.Series,
    test_window: tuple[pd.Timestamp, pd.Timestamp],
    settings: dict,
) -> tuple[pd.DataFrame, xgb.Booster, int | None, float | None, pd.DataFrame]:
    """Run one complete walk-forward train/validation/test cycle."""
    test_start, test_end = test_window
    test = evaluation_subset.loc[(evaluation_subset.index >= test_start) & (evaluation_subset.index <= test_end)].copy()

    latest_date_before_test = all_dates[all_dates < test_start].max()
    latest_date_before_test_position = pd.Index(all_dates).get_loc(latest_date_before_test)

    if settings["validation_before_train"]:
        train_end_date = all_dates.iloc[latest_date_before_test_position - 61]
        train_start_date = train_end_date - pd.Timedelta(settings["train_max_days"], "days")
        train = available_history.loc[
            (available_history.index <= train_end_date) & (available_history.index >= train_start_date)
        ].iloc[-settings["train_max_size"] :]

        train_actual_start = train.index.min()
        train_actual_start_position = pd.Index(all_dates).get_loc(train_actual_start)
        validation_end_date = all_dates.iloc[train_actual_start_position - 61]
        validation = evaluation_subset.loc[evaluation_subset.index <= validation_end_date].iloc[
            -settings["validation_size"] :
        ].copy()
    else:
        validation_end_date = all_dates.iloc[latest_date_before_test_position - 61]
        validation = evaluation_subset.loc[evaluation_subset.index <= validation_end_date].iloc[
            -settings["validation_size"] :
        ].copy()
        validation_actual_start = validation.index.min()
        validation_actual_start_position = pd.Index(all_dates).get_loc(validation_actual_start)
        train_end_date = all_dates.iloc[validation_actual_start_position - 61]
        train_start_date = train_end_date - pd.Timedelta(settings["train_max_days"], "days")
        train = available_history.loc[
            (available_history.index <= train_end_date) & (available_history.index >= train_start_date)
        ].iloc[-settings["train_max_size"] :]

    if settings["remove_validation_tickers_from_train"]:
        train = remove_validation_tickers_from_training(train, validation)

    validated_model, best_iteration, validation_score, validation_predictions = train_with_early_stopping(
        train=train,
        validation=validation,
        params=settings["xgb_params"],
        num_boost_round=settings["num_boost_round"],
        early_stopping_rounds=settings["early_stopping_rounds"],
        zscore_inputs_and_label=settings["zscore_inputs_and_label"],
        time_weights_lambda=settings["time_weights_lambda"],
    )

    validation = validation.copy()
    validation["pred"] = validation_predictions

    test_predictions, fitted_model = predict_test_window(
        validated_model=validated_model,
        train_for_test=train,
        test=test,
        params=settings["xgb_params"],
        prediction_model_source=settings["prediction_model_source"],
        zscore_inputs_and_label=settings["zscore_inputs_and_label"],
        time_weights_lambda=settings["time_weights_lambda"],
    )

    test = test.copy()
    test["pred"] = test_predictions
    return test, fitted_model, best_iteration, validation_score, validation


def _run_one_window_parallel(
    test_window: tuple,
    model_dataset: pd.DataFrame,
    evaluation_subset: pd.DataFrame,
    all_dates: pd.Series,
    settings: dict,
) -> tuple:
    """Wrapper for a single TSCV window, suitable for parallel execution."""
    history_until_test_end = model_dataset.loc[model_dataset.index <= test_window[1]]
    history_window = history_until_test_end.loc[
        history_until_test_end.index >= (test_window[0] - pd.Timedelta(settings["train_max_days"] + 360, "days"))
    ]
    test_predictions, fitted_model, best_iteration, validation_score, validation_predictions = run_one_walk_forward_window(
        available_history=history_window,
        evaluation_subset=evaluation_subset,
        all_dates=all_dates,
        test_window=test_window,
        settings=settings,
    )
    validation_predictions = validation_predictions.copy()
    validation_predictions["test_window_start"] = test_window[0]
    importance = fitted_model.get_score(importance_type="gain")
    return test_window[0], test_predictions, fitted_model, best_iteration, validation_score, validation_predictions, importance


# This function runs the full monthly walk-forward loop across the sample.
def run_walk_forward_evaluation(
    model_dataset: pd.DataFrame,
    settings: dict,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, list[xgb.Booster], list[int | None], list[float | None], pd.DataFrame, pd.DataFrame]:
    """Run the full monthly walk-forward evaluation across the sample.

    n_jobs controls parallelism: -1 = use all available CPU cores.
    Set n_jobs=1 to disable parallelism (useful for debugging).
    """
    model_dataset = model_dataset.sort_index().copy()
    all_dates = pd.Series(sorted(model_dataset.index.unique()))

    evaluation_subset = model_dataset[
        model_dataset["bdfr"].isin(settings["validation_and_test_bdfr"])
    ].copy()
    evaluation_subset = evaluation_subset[
        evaluation_subset["days_from_report"].isin(settings["validation_and_test_days_from_report"])
    ].copy()

    active_windows = generate_monthly_test_windows(model_dataset)[
        settings["period_start"] : settings["period_end"]
    ]

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_one_window_parallel)(
            test_window, model_dataset, evaluation_subset, all_dates, settings
        )
        for test_window in active_windows
    )

    # Results arrive in order (joblib preserves order by default)
    all_test_predictions = []
    all_validation_predictions = []
    fitted_models = []
    best_iterations = []
    validation_scores = []
    feature_importance_records = []

    for period_date, test_preds, fitted_model, best_iter, val_score, val_preds, importance in results:
        all_test_predictions.append(test_preds)
        all_validation_predictions.append(val_preds)
        fitted_models.append(fitted_model)
        best_iterations.append(best_iter)
        validation_scores.append(val_score)
        for feature, score in importance.items():
            feature_importance_records.append({
                "period_date": period_date,
                "feature": feature,
                "importance": score,
            })

    non_empty_test = [p for p in all_test_predictions if not p.empty]
    test_predictions_df = pd.concat(non_empty_test) if non_empty_test else pd.DataFrame()
    non_empty_val = [p for p in all_validation_predictions if not p.empty]
    validation_predictions_df = pd.concat(non_empty_val) if non_empty_val else pd.DataFrame()
    feature_importance_df = (
        pd.DataFrame(feature_importance_records)
        if feature_importance_records
        else pd.DataFrame(columns=["period_date", "feature", "importance"])
    )
    return test_predictions_df, fitted_models, best_iterations, validation_scores, validation_predictions_df, feature_importance_df
