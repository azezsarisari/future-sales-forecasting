import pandas as pd
import pytest

from src.feature_engineering import create_features


def make_sample_data():
    return pd.DataFrame({
        "date": pd.date_range(
            "2024-01-01",
            periods=40,
            freq="D"
        ),
        "sales": [float(i) for i in range(1, 41)],
        "is_missing_date": [0] * 40,
    })


def test_feature_columns():
    df = create_features(make_sample_data())

    expected = [
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
    ]

    for column in expected:
        assert column in df.columns


def test_lag_features():
    df = create_features(make_sample_data())

    assert df.loc[7, "lag_1"] == 7.0
    assert df.loc[7, "lag_7"] == 1.0
    assert df.loc[28, "lag_28"] == 1.0


def test_rolling_features():
    df = create_features(make_sample_data())

    assert df.loc[7, "rolling_mean_7"] == pytest.approx(4.0)
    assert df.loc[14, "rolling_mean_14"] == pytest.approx(7.5)
    assert df.loc[28, "rolling_mean_28"] == pytest.approx(14.5)


def test_no_future_leakage():
    original = make_sample_data()
    modified = original.copy()

    modified.loc[20:, "sales"] = 999999.0

    original_features = create_features(original)
    modified_features = create_features(modified)

    feature_columns = [
        col for col in original_features.columns
        if col.startswith("lag_")
        or col.startswith("rolling_mean_")
    ]

    pd.testing.assert_frame_equal(
        original_features.loc[:20, feature_columns],
        modified_features.loc[:20, feature_columns],
    )


def test_missing_sales_preserved():
    df = make_sample_data()
    df.loc[10, "sales"] = float("nan")
    df.loc[10, "is_missing_date"] = 1

    result = create_features(df)

    assert pd.isna(result.loc[10, "sales"])
    assert pd.isna(result.loc[11, "lag_1"])
    assert pd.isna(result.loc[17, "lag_7"])


def test_duplicate_dates_rejected():
    df = make_sample_data()
    df.loc[1, "date"] = df.loc[0, "date"]

    with pytest.raises(ValueError, match="Duplicate dates"):
        create_features(df)


def test_missing_calendar_day_rejected():
    df = make_sample_data()
    df = df.drop(index=10)

    with pytest.raises(ValueError, match="Calendar must be continuous"):
        create_features(df)
