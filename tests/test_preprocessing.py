import pandas as pd
import pytest

from src.data_preprocessing import (
    aggregate_daily_sales,
    complete_calendar,
    validate_processed_data,
)


def test_daily_aggregation():
    raw = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
            "2024-01-02",
        ]),
        "sales": [100.0, 200.0, 150.0],
    })

    daily = aggregate_daily_sales(raw)

    assert len(daily) == 2
    assert daily["sales"].tolist() == [300.0, 150.0]
    assert daily["sales"].sum() == pytest.approx(raw["sales"].sum())


def test_missing_calendar_dates():
    daily = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-03",
        ]),
        "sales": [100.0, 300.0],
    })

    processed = complete_calendar(daily)

    assert len(processed) == 3
    assert processed["is_missing_date"].sum() == 1

    missing_row = processed.loc[
        processed["date"] == pd.Timestamp("2024-01-02")
    ]

    assert missing_row["sales"].isna().all()
    assert missing_row["is_missing_date"].iloc[0] == 1


def test_processed_data_validation():
    daily = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
        ]),
        "sales": [100.0, 200.0],
    })

    processed = complete_calendar(daily)

    validate_processed_data(processed)


def test_duplicate_dates_rejected():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
        ]),
        "sales": [100.0, 200.0],
        "is_missing_date": [0, 0],
    })

    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_processed_data(df)


def test_negative_sales_rejected():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
        ]),
        "sales": [100.0, -50.0],
        "is_missing_date": [0, 0],
    })

    with pytest.raises(ValueError, match="Negative sales"):
        validate_processed_data(df)
