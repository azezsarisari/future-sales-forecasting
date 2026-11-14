from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import (
    FEATURES_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    MODELS_DIR,
    RANDOM_STATE,
)


def load_training_data():
    df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["date"]
    )

    return df.sort_values("date").reset_index(drop=True)


def split_data(df):
    train = df[df["date"] <= TRAIN_END].copy()

    validation = df[
        (df["date"] >= VALIDATION_START)
        & (df["date"] <= VALIDATION_END)
    ].copy()

    test = df[
        (df["date"] >= TEST_START)
        & (df["date"] <= TEST_END)
    ].copy()

    return train, validation, test


def prepare_training_rows(df):
    return df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()


def build_models():
    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=100.0)),
        ]),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
    }

    return models


def main():
    print("=" * 55)
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 55)

    df = load_training_data()

    train, validation, test = split_data(df)

    train_clean = prepare_training_rows(train)

    assert len(validation) == 30
    assert len(test) == 30

    X_train = train_clean[FEATURE_COLUMNS]
    y_train = train_clean[TARGET_COLUMN]

    print("Training records:", len(train_clean))
    print("Features:", len(FEATURE_COLUMNS))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = build_models()

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        output_path = MODELS_DIR / f"{name}_validation.joblib"

        joblib.dump(model, output_path)

        print("Model saved:", output_path)

    print("\nMODEL TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
