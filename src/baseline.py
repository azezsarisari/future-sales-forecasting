import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON
from src.train import load_training_data, split_data


def seasonal_naive_forecast(history, horizon=30):
    sales = pd.to_numeric(history["sales"], errors="coerce")

    if len(sales) < 7:
        raise ValueError("At least 7 historical days are required")

    last_week = sales.iloc[-7:].to_numpy(dtype=float)

    if not np.isfinite(last_week).all():
        raise ValueError("Last 7 historical sales must be valid")

    predictions = np.resize(last_week, horizon)

    return np.maximum(predictions, 0.0)


def evaluate_forecast(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ")

    if not np.isfinite(actual).all():
        raise ValueError("Actual sales contain missing values")

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    total_actual = np.sum(np.abs(actual))

    wape = (
        100 * np.sum(np.abs(errors)) / total_actual
        if total_actual > 0
        else np.nan
    )

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "WAPE": round(float(wape), 2),
    }


def main():
    df = load_training_data()
    train, validation, test = split_data(df)

    predictions = seasonal_naive_forecast(
        train,
        horizon=FORECAST_HORIZON
    )

    metrics = evaluate_forecast(
        validation["sales"],
        predictions
    )

    print("=" * 55)
    print("SEASONAL NAIVE BASELINE")
    print("=" * 55)

    print("Forecast horizon:", FORECAST_HORIZON)
    print("Training end:", train["date"].max().date())
    print("Validation start:", validation["date"].min().date())
    print("Validation end:", validation["date"].max().date())

    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nBASELINE EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
