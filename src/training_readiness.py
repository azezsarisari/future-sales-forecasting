import numpy as np
import pandas as pd


MIN_CALENDAR_DAYS = 120
MIN_OBSERVED_DAYS = 90
MAX_MISSING_RATIO = 0.10
REQUIRED_RECENT_DAYS = 28


def validate_training_readiness(df):
    required = {"date", "sales", "is_missing_date"}

    if not required.issubset(df.columns):
        raise ValueError("Missing required training columns")

    if df.empty:
        raise ValueError("Training data is empty")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    if data["date"].isna().any():
        raise ValueError("Invalid training dates")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate training dates")

    data = data.sort_values("date").reset_index(drop=True)

    expected = pd.date_range(
        data["date"].min(),
        data["date"].max(),
        freq="D"
    )

    if not data["date"].reset_index(drop=True).equals(
        pd.Series(expected)
    ):
        raise ValueError("Training calendar is not continuous")

    calendar_days = len(data)
    observed_days = int(data["sales"].notna().sum())
    missing_days = calendar_days - observed_days
    missing_ratio = missing_days / calendar_days

    if calendar_days < MIN_CALENDAR_DAYS:
        raise ValueError("Insufficient calendar history")

    if observed_days < MIN_OBSERVED_DAYS:
        raise ValueError("Insufficient observed sales history")

    if missing_ratio > MAX_MISSING_RATIO:
        raise ValueError("Too many missing sales days")

    sales = pd.to_numeric(data["sales"], errors="coerce")
    observed_sales = sales.dropna()

    if not np.isfinite(observed_sales.to_numpy()).all():
        raise ValueError("Non-finite sales values")

    if (observed_sales < 0).any():
        raise ValueError("Negative sales values")

    if len(data) < REQUIRED_RECENT_DAYS:
        raise ValueError("Insufficient recent history")

    recent = sales.tail(REQUIRED_RECENT_DAYS)

    if recent.isna().any():
        raise ValueError("Last 28 days must contain valid sales")

    expected_flags = sales.isna().astype(int)

    if not data["is_missing_date"].equals(expected_flags):
        raise ValueError("Missing-date flags are inconsistent")

    return {
        "ready": True,
        "calendar_days": calendar_days,
        "observed_days": observed_days,
        "missing_days": missing_days,
        "missing_ratio": round(missing_ratio, 4),
        "last_date": str(data["date"].max().date()),
    }
