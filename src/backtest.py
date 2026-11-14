import pandas as pd
import numpy as np

from src.config import (
    FEATURES_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FORECAST_HORIZON,
    METRICS_DIR,
)

from src.train import (
    build_models,
    prepare_training_rows,
)

from src.baseline import (
    seasonal_naive_forecast,
    evaluate_forecast,
)

from src.predict import recursive_forecast


BACKTEST_END_DATES = [
    "2017-03-16",
    "2017-04-16",
    "2017-05-16",
    "2017-06-16",
]


def main():
    df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)

    results = []

    for train_end_str in BACKTEST_END_DATES:
        train_end = pd.Timestamp(train_end_str)
        forecast_end = train_end + pd.Timedelta(
            days=FORECAST_HORIZON
        )

        train = df[df["date"] <= train_end].copy()

        validation = df[
            (df["date"] > train_end)
            & (df["date"] <= forecast_end)
        ].copy()

        if len(validation) != FORECAST_HORIZON:
            raise ValueError("Invalid backtest period")

        if validation["sales"].isna().any():
            raise ValueError(
                "Missing actual sales in backtest period"
            )

        print(
            f"\nBacktest: {train_end.date()} "
            f"to {forecast_end.date()}"
        )

        baseline_pred = seasonal_naive_forecast(
            train,
            FORECAST_HORIZON
        )

        baseline_metrics = evaluate_forecast(
            validation["sales"],
            baseline_pred
        )

        results.append({
            "train_end": str(train_end.date()),
            "model": "seasonal_naive",
            **baseline_metrics,
        })

        train_clean = prepare_training_rows(train)

        X_train = train_clean[FEATURE_COLUMNS]
        y_train = train_clean[TARGET_COLUMN]

        models = build_models()

        for name, model in models.items():
            model.fit(X_train, y_train)

            forecast = recursive_forecast(
                model,
                train[["date", "sales"]],
                FORECAST_HORIZON
            )

            if not forecast["date"].reset_index(drop=True).equals(
                validation["date"].reset_index(drop=True)
            ):
                raise ValueError("Forecast date mismatch")

            metrics = evaluate_forecast(
                validation["sales"],
                forecast["predicted_sales"]
            )

            results.append({
                "train_end": str(train_end.date()),
                "model": name,
                **metrics,
            })

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby("model")[
            ["MAE", "RMSE", "WAPE"]
        ]
        .mean()
        .round(2)
        .sort_values("MAE")
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        METRICS_DIR / "backtest_results.csv",
        index=False
    )

    summary.to_csv(
        METRICS_DIR / "backtest_summary.csv"
    )

    print("\nBACKTEST SUMMARY:")
    print(summary)

    print("\nBACKTESTING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
