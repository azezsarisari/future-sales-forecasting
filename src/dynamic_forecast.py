import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.dynamic_features import prepare_dynamic_features
from src.dynamic_train import train_dynamic_model
from src.train import build_models
from src.predict import recursive_forecast
from src.baseline import seasonal_naive_forecast


SUPPORTED_HORIZONS = (7, 14, 30)


def generate_dynamic_forecast(
    processed_data,
    horizon=30,
    validation_days=30,
):
    if type(horizon) is not int or horizon not in SUPPORTED_HORIZONS:
        raise ValueError("Forecast horizon must be 7, 14, or 30")

    if type(validation_days) is not int or not 1 <= validation_days <= 30:
        raise ValueError("Validation days must be between 1 and 30")

    features, training_rows, readiness = prepare_dynamic_features(
        processed_data
    )

    _, comparison, validation_predictions, evaluation = (
        train_dynamic_model(
            processed_data,
            validation_days=validation_days,
        )
    )

    selected_model = evaluation["best_validation_model"]

    history = features[["date", "sales"]].copy()
    last_date = history["date"].max()

    if selected_model == "xgboost":
        model = build_models()["xgboost"]
        model.fit(
            training_rows[FEATURE_COLUMNS],
            training_rows[TARGET_COLUMN],
        )

        forecast = recursive_forecast(
            model,
            history,
            horizon=horizon,
        )

    elif selected_model == "seasonal_naive":
        predictions = seasonal_naive_forecast(
            history,
            horizon=horizon,
        )

        forecast = pd.DataFrame({
            "date": pd.date_range(
                last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq="D",
            ),
            "predicted_sales": predictions,
        })

    else:
        raise ValueError("Unsupported selected model")

    expected_dates = pd.Series(
        pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
    )

    if not forecast["date"].reset_index(drop=True).equals(
        expected_dates
    ):
        raise ValueError("Forecast dates do not match expected dates")

    values = forecast["predicted_sales"].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Forecast contains invalid values")

    if (values < 0).any():
        raise ValueError("Forecast contains negative sales")

    report = {
        **readiness,
        "selected_model": selected_model,
        "selection_metric": "MAE",
        "validation_days": validation_days,
        "forecast_horizon": horizon,
        "forecast_start": str(forecast["date"].min().date()),
        "forecast_end": str(forecast["date"].max().date()),
        "total_predicted_sales": round(float(values.sum()), 2),
        "average_predicted_sales": round(float(values.mean()), 2),
    }

    return forecast, comparison, validation_predictions, report
