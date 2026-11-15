import numpy as np
import pandas as pd
import pytest

from src.training_readiness import validate_training_readiness


def make_data(days=150):
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days, freq="D"),
        "sales": [100.0] * days,
        "is_missing_date": [0] * days,
    })


def test_valid_data():
    result = validate_training_readiness(make_data())
    assert result["ready"] is True
    assert result["observed_days"] == 150


def test_missing_columns():
    df = make_data().drop(columns=["sales"])
    with pytest.raises(ValueError, match="Missing required"):
        validate_training_readiness(df)


def test_empty_data():
    with pytest.raises(ValueError, match="empty"):
        validate_training_readiness(
            pd.DataFrame(columns=["date", "sales", "is_missing_date"])
        )


def test_insufficient_calendar_history():
    with pytest.raises(ValueError, match="Insufficient calendar"):
        validate_training_readiness(make_data(60))


def test_insufficient_observed_history():
    df = make_data()
    df.loc[:65, "sales"] = np.nan
    df["is_missing_date"] = df["sales"].isna().astype(int)

    with pytest.raises(ValueError, match="Insufficient observed"):
        validate_training_readiness(df)


def test_excessive_missing_ratio():
    df = make_data()
    df.loc[:19, "sales"] = np.nan
    df["is_missing_date"] = df["sales"].isna().astype(int)

    with pytest.raises(ValueError, match="Too many missing"):
        validate_training_readiness(df)


def test_missing_recent_sales():
    df = make_data()
    df.loc[149, "sales"] = np.nan
    df["is_missing_date"] = df["sales"].isna().astype(int)

    with pytest.raises(ValueError, match="Last 28 days"):
        validate_training_readiness(df)


def test_negative_sales():
    df = make_data()
    df.loc[10, "sales"] = -50.0

    with pytest.raises(ValueError, match="Negative"):
        validate_training_readiness(df)


def test_infinite_sales():
    df = make_data()
    df.loc[10, "sales"] = np.inf

    with pytest.raises(ValueError, match="Non-finite"):
        validate_training_readiness(df)


def test_duplicate_dates():
    df = make_data()
    df.loc[1, "date"] = df.loc[0, "date"]

    with pytest.raises(ValueError, match="Duplicate"):
        validate_training_readiness(df)


def test_missing_calendar_day():
    df = make_data().drop(index=10)

    with pytest.raises(ValueError, match="not continuous"):
        validate_training_readiness(df)


def test_inconsistent_missing_flags():
    df = make_data()
    df.loc[10, "is_missing_date"] = 1

    with pytest.raises(ValueError, match="inconsistent"):
        validate_training_readiness(df)


def test_valid_data_with_missing_history():
    df = make_data()
    df.loc[10, "sales"] = np.nan
    df["is_missing_date"] = df["sales"].isna().astype(int)

    result = validate_training_readiness(df)

    assert result["ready"] is True
    assert result["missing_days"] == 1
    assert result["observed_days"] == 149
