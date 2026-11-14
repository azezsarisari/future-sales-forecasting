import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, FORECAST_HORIZON
from src.feature_engineering import create_features


def recursive_forecast(model, history, horizon=FORECAST_HORIZON):
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")

    required = {"date", "sales"}
    if not required.issubset(history.columns):
        raise ValueError("History must contain date and sales")

    working = history[["date", "sales"]].copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values("date").reset_index(drop=True)

    if working.empty:
        raise ValueError("History cannot be empty")

    if working["date"].duplicated().any():
        raise ValueError("Duplicate historical dates")

    if not working["date"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError("Historical calendar must be continuous")

    if len(working) < 28:
        raise ValueError(
            "Last 28 historical sales must be valid"
        )

    if not np.isfinite(
        working["sales"].tail(28).to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Last 28 historical sales must be valid"
        )

    working["is_missing_date"] = (
        working["sales"].isna().astype(int)
    )

    predictions = []

    for _ in range(horizon):
        next_date = working["date"].iloc[-1] + pd.Timedelta(days=1)

        future_row = pd.DataFrame({
            "date": [next_date],
            "sales": [np.nan],
            "is_missing_date": [1],
        })

        working = pd.concat(
            [working, future_row],
            ignore_index=True
        )

        features = create_features(working)

        X_next = features.loc[
            [features.index[-1]], FEATURE_COLUMNS
        ]

        if X_next.isna().any().any():
            raise ValueError(
                "Missing features in forecast step"
            )

        prediction = float(model.predict(X_next)[0])

        if not np.isfinite(prediction):
            raise ValueError("Model returned invalid prediction")

        prediction = max(prediction, 0.0)

        working.loc[working.index[-1], "sales"] = prediction
        working.loc[working.index[-1], "is_missing_date"] = 0

        predictions.append({
            "date": next_date,
            "predicted_sales": prediction,
        })

    return pd.DataFrame(predictions)

