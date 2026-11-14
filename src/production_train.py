import json

import joblib
import pandas as pd

from src.config import (
    FEATURES_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODELS_DIR,
    METRICS_DIR,
)

from src.train import build_models, prepare_training_rows


def main():
    print("=" * 55)
    print("FULL HISTORICAL MODEL TRAINING")
    print("=" * 55)

    selection_path = METRICS_DIR / "model_selection.json"

    with open(selection_path, "r", encoding="utf-8-sig") as file:
        selection = json.load(file)

    if selection["selected_model"] != "xgboost":
        raise ValueError("Unexpected selected model")

    if not selection["test_evaluated"]:
        raise ValueError("Final holdout test not completed")

    df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)

    train_clean = prepare_training_rows(df)

    X_train = train_clean[FEATURE_COLUMNS]
    y_train = train_clean[TARGET_COLUMN]

    print("Historical records:", len(df))
    print("Usable training records:", len(train_clean))
    print("Features:", len(FEATURE_COLUMNS))
    print("Last historical date:", df["date"].max().date())

    model = build_models()["xgboost"]

    print("\nTraining production model...")
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgboost_production.joblib"

    joblib.dump(model, model_path)

    metadata = {
        "model": "xgboost",
        "training_end": str(df["date"].max().date()),
        "training_rows": len(train_clean),
        "features": FEATURE_COLUMNS,
        "forecast_horizon": 30,
        "status": "trained_on_full_historical_data"
    }

    with open(
        METRICS_DIR / "production_model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nModel saved:", model_path)
    print("FULL HISTORICAL TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
