import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DAILY_SALES_PATH,
    MODELS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FORECAST_HORIZON,
)

from src.predict import recursive_forecast


def main():
    print("=" * 55)
    print("FUTURE SALES FORECASTING")
    print("=" * 55)

    model_path = MODELS_DIR / "xgboost_production.joblib"
    metadata_path = METRICS_DIR / "production_model_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["model"] != "xgboost":
        raise ValueError("Unexpected production model")

    history = pd.read_csv(
        DAILY_SALES_PATH,
        parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)

    last_date = history["date"].max()

    if str(last_date.date()) != metadata["training_end"]:
        raise ValueError(
            "Historical data does not match model training end"
        )

    model = joblib.load(model_path)

    forecast = recursive_forecast(
        model,
        history[["date", "sales"]],
        horizon=FORECAST_HORIZON
    )

    if len(forecast) != FORECAST_HORIZON:
        raise ValueError("Invalid forecast length")

    if not np.isfinite(
        forecast["predicted_sales"].to_numpy()
    ).all():
        raise ValueError("Invalid forecast values")

    if (forecast["predicted_sales"] < 0).any():
        raise ValueError("Negative forecast values")

    expected_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D"
    )

    if not forecast["date"].reset_index(drop=True).equals(
        pd.Series(expected_dates)
    ):
        raise ValueError("Forecast dates are incorrect")

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PREDICTIONS_DIR / "future_30_days.csv"

    forecast.to_csv(output_path, index=False)

    print("Historical end:", last_date.date())
    print("Forecast start:", forecast["date"].min().date())
    print("Forecast end:", forecast["date"].max().date())
    print("Forecast days:", len(forecast))

    print(
        "Total predicted sales:",
        round(forecast["predicted_sales"].sum(), 2)
    )

    print(
        "Average daily predicted sales:",
        round(forecast["predicted_sales"].mean(), 2)
    )

    print("\nFirst 5 predictions:")
    print(forecast.head().to_string(index=False))

    print("\nLast 5 predictions:")
    print(forecast.tail().to_string(index=False))

    print("\nOutput:", output_path)
    print("FUTURE FORECASTING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
