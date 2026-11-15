import numpy as np
import pandas as pd
import pytest

from src.upload_validation import validate_sales_data


def make_data(days=150):
    return pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01",
            periods=days,
            freq="D"
        ),
        "sales": [100.0] * days,
    })


def test_valid_dataset():
    result = validate_sales_data(make_data())

    assert len(result) == 150
    assert result["sales"].sum() == 15000.0
    assert result["is_missing_date"].sum() == 0


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_sales_data(pd.DataFrame())


def test_missing_columns_rejected():
    df = make_data().drop(columns=["sales"])

    with pytest.raises(ValueError, match="Missing required"):
        validate_sales_data(df)


def test_negative_sales_rejected():
    df = make_data()
    df.loc[5, "sales"] = -100.0

    with pytest.raises(ValueError, match="Negative"):
        validate_sales_data(df)


def test_invalid_dates_rejected():
    df = make_data()
    df["date"] = df["date"].astype(object)
    df.loc[5, "date"] = "not-a-date"

    with pytest.raises(ValueError, match="Invalid dates"):
        validate_sales_data(df)


def test_missing_sales_rejected():
    df = make_data()
    df.loc[5, "sales"] = np.nan

    with pytest.raises(ValueError, match="Invalid or missing"):
        validate_sales_data(df)


def test_infinite_sales_rejected():
    df = make_data()
    df.loc[5, "sales"] = np.inf

    with pytest.raises(ValueError, match="Non-finite"):
        validate_sales_data(df)


def test_short_history_rejected():
    with pytest.raises(ValueError, match="At least 120"):
        validate_sales_data(make_data(days=30))


def test_duplicate_dates_aggregated():
    df = make_data()

    duplicate = df.iloc[[0]].copy()
    duplicate["sales"] = 50.0

    df = pd.concat([df, duplicate], ignore_index=True)

    result = validate_sales_data(df)

    assert len(result) == 150
    assert result.iloc[0]["sales"] == 150.0


def test_missing_calendar_dates_preserved():
    df = make_data()
    df = df.drop(index=10)

    result = validate_sales_data(df)

    assert len(result) == 150
    assert result["is_missing_date"].sum() == 1
    assert pd.isna(result.loc[10, "sales"])


def test_custom_column_mapping():
    df = make_data().rename(
        columns={
            "date": "transaction_date",
            "sales": "revenue",
        }
    )

    result = validate_sales_data(
        df,
        date_column="transaction_date",
        sales_column="revenue",
    )

    assert len(result) == 150
    assert result["sales"].sum() == 15000.0


def test_same_columns_rejected():
    df = make_data()

    with pytest.raises(ValueError, match="different"):
        validate_sales_data(
            df,
            date_column="date",
            sales_column="date",
        )


def test_time_components_rejected():
    df = make_data()
    df.loc[0, "date"] = "2026-01-01 12:30:00"

    with pytest.raises(ValueError, match="time components"):
        validate_sales_data(df)
