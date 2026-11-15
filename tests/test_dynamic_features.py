import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.dynamic_features import prepare_dynamic_features


def make_data(days=150):
    return pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01", periods=days, freq="D"
        ),
        "sales": np.arange(1000, 1000 + days, dtype=float),
        "is_missing_date": [0] * days,
    })


def test_feature_generation():
    features, training, report = prepare_dynamic_features(
        make_data()
    )

    assert len(features) == 150
    assert len(training) == 122
    assert report["feature_count"] == 12


def test_training_features_complete():
    _, training, _ = prepare_dynamic_features(make_data())

    assert training[FEATURE_COLUMNS].notna().all().all()
    assert training["sales"].notna().all()


def test_first_28_days_excluded():
    _, training, _ = prepare_dynamic_features(make_data())

    assert training["date"].min() == pd.Timestamp(
        "2026-01-29"
    )


def test_lag_values():
    features, _, _ = prepare_dynamic_features(make_data())

    assert features.loc[28, "lag_1"] == 1027.0
    assert features.loc[28, "lag_7"] == 1021.0
    assert features.loc[28, "lag_28"] == 1000.0


def test_rolling_average():
    features, _, _ = prepare_dynamic_features(make_data())

    assert features.loc[28, "rolling_mean_7"] == pytest.approx(
        1024.0
    )
    assert features.loc[28, "rolling_mean_28"] == pytest.approx(
        1013.5
    )


def test_no_future_leakage():
    original = make_data()
    modified = original.copy()

    modified.loc[100:, "sales"] = 999999.0

    original_features, _, _ = prepare_dynamic_features(
        original
    )
    modified_features, _, _ = prepare_dynamic_features(
        modified
    )

    pd.testing.assert_frame_equal(
        original_features.loc[:100, FEATURE_COLUMNS],
        modified_features.loc[:100, FEATURE_COLUMNS],
    )


def test_missing_historical_sales():
    df = make_data()
    df.loc[40, "sales"] = np.nan
    df["is_missing_date"] = df["sales"].isna().astype(int)

    features, training, report = prepare_dynamic_features(df)

    assert pd.isna(features.loc[40, "sales"])
    assert report["missing_days"] == 1
    assert len(training) < 122


def test_insufficient_calendar_history():
    with pytest.raises(ValueError, match="Insufficient calendar"):
        prepare_dynamic_features(make_data(60))


def test_missing_recent_sales_rejected():
    df = make_data()
    df.loc[149, "sales"] = np.nan
    df["is_missing_date"] = df["sales"].isna().astype(int)

    with pytest.raises(ValueError, match="Last 28 days"):
        prepare_dynamic_features(df)


def test_input_not_modified():
    df = make_data()
    original = df.copy(deep=True)

    prepare_dynamic_features(df)

    pd.testing.assert_frame_equal(df, original)
