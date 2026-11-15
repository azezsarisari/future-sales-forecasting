import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.dynamic_features import prepare_dynamic_features
from src.train import build_models
from src.predict import recursive_forecast
from src.baseline import seasonal_naive_forecast, evaluate_forecast


def run_dynamic_backtest(
    processed_data,
    windows=4,
    horizon=30,
):
    if type(windows) is not int or not 1 <= windows <= 10:
        raise ValueError("Windows must be between 1 and 10")

    if type(horizon) is not int or not 1 <= horizon <= 30:
        raise ValueError("Horizon must be between 1 and 30")

    features, _, _ = prepare_dynamic_features(processed_data)

    required_history = 28 + 60 + windows * horizon

    if len(features) < required_history:
        raise ValueError(
            f"At least {required_history} calendar days "
            "are required for this backtest"
        )

    results = []

    for window in range(windows):
        remaining = (windows - window) * horizon
        split_index = len(features) - remaining

        train = features.iloc[:split_index].copy()
        test = features.iloc[
            split_index:split_index + horizon
        ].copy()

        train_clean = train.dropna(
            subset=FEATURE_COLUMNS + [TARGET_COLUMN]
        )

        if len(train_clean) < 60:
            raise ValueError("Insufficient backtest training rows")

        if train["sales"].tail(28).isna().any():
            raise ValueError(
                "Missing sales in last 28 training days"
            )

        if test["sales"].isna().any():
            raise ValueError(
                "Missing actual sales in backtest window"
            )

        history = train[["date", "sales"]].copy()

        baseline = seasonal_naive_forecast(
            history,
            horizon=horizon,
        )

        model = build_models()["xgboost"]
        model.fit(
            train_clean[FEATURE_COLUMNS],
            train_clean[TARGET_COLUMN],
        )

        forecast = recursive_forecast(
            model,
            history,
            horizon=horizon,
        )

        if not forecast["date"].reset_index(drop=True).equals(
            test["date"].reset_index(drop=True)
        ):
            raise ValueError("Backtest forecast dates mismatch")

        xgb_predictions = forecast[
            "predicted_sales"
        ].to_numpy()

        if not np.isfinite(xgb_predictions).all():
            raise ValueError("Invalid backtest predictions")

        for model_name, predictions in [
            ("seasonal_naive", baseline),
            ("xgboost", xgb_predictions),
        ]:
            metrics = evaluate_forecast(
                test[TARGET_COLUMN],
                predictions,
            )

            results.append({
                "window": window + 1,
                "model": model_name,
                "training_end": str(
                    train["date"].max().date()
                ),
                "test_start": str(
                    test["date"].min().date()
                ),
                "test_end": str(
                    test["date"].max().date()
                ),
                **metrics,
            })

    details = pd.DataFrame(results)

    summary = (
        details.groupby("model", as_index=False)[
            ["MAE", "RMSE", "WAPE"]
        ]
        .mean()
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    return details, summary
