import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.dynamic_features import prepare_dynamic_features
from src.dynamic_split import split_dynamic_data
from src.train import build_models
from src.predict import recursive_forecast
from src.baseline import seasonal_naive_forecast, evaluate_forecast


def train_dynamic_model(processed_data, validation_days=30):
    if not isinstance(validation_days, int) or isinstance(validation_days, bool):
        raise ValueError("Validation days must be an integer")

    if not 1 <= validation_days <= 30:
        raise ValueError("Validation days must be between 1 and 30")

    features, _, readiness = prepare_dynamic_features(
        processed_data
    )

    train, train_clean, validation = split_dynamic_data(
        features,
        validation_days=validation_days,
    )

    history = train[["date", "sales"]].copy()

    baseline_predictions = seasonal_naive_forecast(
        history,
        horizon=validation_days,
    )

    baseline_metrics = evaluate_forecast(
        validation[TARGET_COLUMN],
        baseline_predictions,
    )

    model = build_models()["xgboost"]

    model.fit(
        train_clean[FEATURE_COLUMNS],
        train_clean[TARGET_COLUMN],
    )

    forecast = recursive_forecast(
        model,
        history,
        horizon=validation_days,
    )

    expected_dates = validation["date"].reset_index(drop=True)
    actual_dates = forecast["date"].reset_index(drop=True)

    if not actual_dates.equals(expected_dates):
        raise ValueError("Validation forecast dates do not match")

    predicted = forecast["predicted_sales"].to_numpy()

    if not np.isfinite(predicted).all():
        raise ValueError("Invalid model predictions")

    xgboost_metrics = evaluate_forecast(
        validation[TARGET_COLUMN],
        predicted,
    )

    comparison = pd.DataFrame([
        {"model": "seasonal_naive", **baseline_metrics},
        {"model": "xgboost", **xgboost_metrics},
    ]).sort_values("MAE").reset_index(drop=True)

    predictions = validation[["date", "sales"]].copy()
    predictions = predictions.rename(
        columns={"sales": "actual_sales"}
    )
    predictions["seasonal_naive"] = baseline_predictions
    predictions["xgboost"] = predicted

    report = {
        **readiness,
        "training_end": str(train["date"].max().date()),
        "validation_start": str(validation["date"].min().date()),
        "validation_end": str(validation["date"].max().date()),
        "validation_days": validation_days,
        "training_rows": len(train_clean),
        "best_validation_model": str(comparison.iloc[0]["model"]),
    }

    return model, comparison, predictions, report
