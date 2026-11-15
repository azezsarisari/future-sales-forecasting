import numpy as np
import pandas as pd
import pytest

from src.dynamic_features import prepare_dynamic_features
from src.dynamic_split import split_dynamic_data


def make_features(days=180):
    df = pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01", periods=days, freq="D"
        ),
        "sales": np.arange(1000, 1000 + days, dtype=float),
        "is_missing_date": [0] * days,
    })
    return prepare_dynamic_features(df)[0]


def test_valid_split():
    train, clean, validation = split_dynamic_data(
        make_features()
    )

    assert len(train) == 150
    assert len(clean) == 122
    assert len(validation) == 30


def test_chronological_order():
    train, _, validation = split_dynamic_data(
        make_features()
    )

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].is_monotonic_increasing


def test_validation_period():
    _, _, validation = split_dynamic_data(
        make_features()
    )

    assert validation["date"].min() == pd.Timestamp(
        "2026-05-31"
    )
    assert validation["date"].max() == pd.Timestamp(
        "2026-06-29"
    )


def test_custom_validation_days():
    train, _, validation = split_dynamic_data(
        make_features(), validation_days=14
    )

    assert len(train) == 166
    assert len(validation) == 14


def test_invalid_validation_days():
    with pytest.raises(ValueError, match="positive"):
        split_dynamic_data(
            make_features(), validation_days=0
        )


def test_empty_features():
    with pytest.raises(ValueError, match="empty"):
        split_dynamic_data(pd.DataFrame())


def test_duplicate_dates():
    df = make_features()
    df.loc[1, "date"] = df.loc[0, "date"]

    with pytest.raises(ValueError, match="Duplicate"):
        split_dynamic_data(df)


def test_missing_calendar_date():
    df = make_features().drop(index=20)

    with pytest.raises(ValueError, match="continuous"):
        split_dynamic_data(df)


def test_insufficient_training_rows():
    from src.feature_engineering import create_features

    raw = pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01", periods=100, freq="D"
        ),
        "sales": np.arange(1000, 1100, dtype=float),
        "is_missing_date": [0] * 100,
    })

    features = create_features(raw)

    with pytest.raises(ValueError, match="Insufficient usable"):
        split_dynamic_data(features)


def test_missing_validation_sales():
    df = make_features()
    df.loc[df.index[-1], "sales"] = np.nan

    with pytest.raises(ValueError, match="Validation period"):
        split_dynamic_data(df)


def test_missing_recent_training_sales():
    df = make_features()
    df.loc[149, "sales"] = np.nan

    with pytest.raises(ValueError, match="Last 28 training"):
        split_dynamic_data(df)


def test_input_not_modified():
    df = make_features()
    original = df.copy(deep=True)

    split_dynamic_data(df)

    pd.testing.assert_frame_equal(df, original)
