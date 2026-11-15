import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN

VALIDATION_DAYS = 30
MIN_TRAINING_ROWS = 60


def split_dynamic_data(features, validation_days=VALIDATION_DAYS):
    if validation_days < 1:
        raise ValueError("Validation days must be positive")

    if features.empty:
        raise ValueError("Feature dataset is empty")

    data = features.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    if not data["date"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError("Calendar must be continuous")

    if len(data) <= validation_days:
        raise ValueError("Insufficient data for validation")

    train = data.iloc[:-validation_days].copy()
    validation = data.iloc[-validation_days:].copy()

    train_clean = train.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    if len(train_clean) < MIN_TRAINING_ROWS:
        raise ValueError(
            "Insufficient usable training rows before validation"
        )

    if validation[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Validation period contains missing sales"
        )

    if train[TARGET_COLUMN].tail(28).isna().any():
        raise ValueError(
            "Last 28 training days must contain valid sales"
        )

    if train["date"].max() >= validation["date"].min():
        raise ValueError("Chronological split is invalid")

    return train, train_clean, validation
