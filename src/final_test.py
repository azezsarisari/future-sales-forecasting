import json

import joblib
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURES_PATH,
    MODELS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FORECAST_HORIZON,
    TEST_START,
    TEST_END,
)

from src.train import build_models, prepare_training_rows
from src.predict import recursive_forecast
from src.baseline import evaluate_forecast


def main():
    print("=" * 55)
    print("FINAL HOLDOUT TEST EVALUATION")
    print("=" * 55)

    selection_path = METRICS_DIR / "model_selection.json"

    with open(selection_path, "r", encoding="utf-8-sig") as file:
        selection = json.load(file)

    if selection["selected_model"] != "xgboost":
        raise ValueError("Unexpected selected model")

    if selection["selection_status"] != "locked_before_final_test":
        raise ValueError("Model selection is not locked")

    if selection["test_evaluated"]:
        raise ValueError("Final test already evaluated")

    df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    train = df[df["date"] < test_start].copy()

    test = df[
        (df["date"] >= test_start)
        & (df["date"] <= test_end)
    ].copy()

    if len(test) != FORECAST_HORIZON:
        raise ValueError("Invalid final test horizon")

    if test["sales"].isna().any():
        raise ValueError("Missing actual sales in final test")

    train_clean = prepare_training_rows(train)

    X_train = train_clean[FEATURE_COLUMNS]
    y_train = train_clean[TARGET_COLUMN]

    print("Training rows:", len(train_clean))
    print("Training end:", train["date"].max().date())
    print("Test start:", test_start.date())
    print("Test end:", test_end.date())

    model = build_models()["xgboost"]

    print("\nTraining final test model...")
    model.fit(X_train, y_train)

    forecast = recursive_forecast(
        model,
        train[["date", "sales"]],
        horizon=FORECAST_HORIZON
    )

    if not forecast["date"].reset_index(drop=True).equals(
        test["date"].reset_index(drop=True)
    ):
        raise ValueError("Forecast dates do not match final test")

    metrics = evaluate_forecast(
        test["sales"],
        forecast["predicted_sales"]
    )

    print("\nFINAL TEST METRICS:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "xgboost_final_test.joblib"
    )

    output = test[["date", "sales"]].copy()
    output = output.rename(columns={"sales": "actual_sales"})
    output["predicted_sales"] = forecast["predicted_sales"].to_numpy()

    output.to_csv(
        PREDICTIONS_DIR / "final_test_predictions.csv",
        index=False
    )

    with open(
        METRICS_DIR / "final_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "model": "xgboost",
                "forecast_horizon": FORECAST_HORIZON,
                "training_end": str(train["date"].max().date()),
                "test_start": TEST_START,
                "test_end": TEST_END,
                **metrics,
            },
            file,
            indent=4
        )

    selection["test_evaluated"] = True
    selection["selection_status"] = "final_test_completed"

    with open(selection_path, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=4)

    print("\nFINAL TEST EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
