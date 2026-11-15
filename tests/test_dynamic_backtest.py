import numpy as np
import pandas as pd
import pytest

from src.dynamic_backtest import run_dynamic_backtest


@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(42)
    days = np.arange(365)

    sales = (
        5000
        + days * 4
        + 800 * np.sin(2 * np.pi * days / 7)
        + rng.normal(0, 150, 365)
    )

    return pd.DataFrame({
        "date": pd.date_range(
            "2025-01-01", periods=365, freq="D"
        ),
        "sales": np.maximum(sales, 0).round(2),
        "is_missing_date": [0] * 365,
    })


@pytest.fixture(scope="module")
def backtest_result(sample_data):
    return run_dynamic_backtest(
        sample_data, windows=4, horizon=30
    )


def test_result_structure(backtest_result):
    details, summary = backtest_result

    assert len(details) == 8
    assert len(summary) == 2
    assert details["window"].nunique() == 4
    assert set(details["model"]) == {
        "xgboost", "seasonal_naive"
    }


def test_metrics_are_valid(backtest_result):
    details, summary = backtest_result

    for frame in (details, summary):
        values = frame[["MAE", "RMSE", "WAPE"]].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0).all()


def test_chronological_windows(backtest_result):
    details, _ = backtest_result

    for window in range(1, 5):
        rows = details[details["window"] == window]

        training_end = pd.Timestamp(rows.iloc[0]["training_end"])
        test_start = pd.Timestamp(rows.iloc[0]["test_start"])
        test_end = pd.Timestamp(rows.iloc[0]["test_end"])

        assert test_start == training_end + pd.Timedelta(days=1)
        assert (test_end - test_start).days == 29


def test_nonoverlapping_windows(backtest_result):
    details, _ = backtest_result

    periods = (
        details.drop_duplicates("window")
        .sort_values("window")
    )

    for i in range(1, len(periods)):
        previous_end = pd.Timestamp(
            periods.iloc[i - 1]["test_end"]
        )
        current_start = pd.Timestamp(
            periods.iloc[i]["test_start"]
        )

        assert current_start == previous_end + pd.Timedelta(days=1)


def test_summary_matches_details(backtest_result):
    details, summary = backtest_result

    expected = details.groupby("model")[
        ["MAE", "RMSE", "WAPE"]
    ].mean()

    for _, row in summary.iterrows():
        model = row["model"]

        for metric in ("MAE", "RMSE", "WAPE"):
            assert row[metric] == pytest.approx(
                expected.loc[model, metric]
            )


@pytest.mark.parametrize("windows", [0, -1, 11, True, "4"])
def test_invalid_windows(sample_data, windows):
    with pytest.raises(ValueError, match="Windows"):
        run_dynamic_backtest(sample_data, windows=windows)


@pytest.mark.parametrize("horizon", [0, -1, 31, True, "30"])
def test_invalid_horizons(sample_data, horizon):
    with pytest.raises(ValueError, match="Horizon"):
        run_dynamic_backtest(sample_data, horizon=horizon)


def test_insufficient_history(sample_data):
    short = sample_data.head(150).copy()

    with pytest.raises(ValueError, match="At least"):
        run_dynamic_backtest(
            short, windows=4, horizon=30
        )


def test_missing_backtest_actuals(sample_data):
    modified = sample_data.copy()
    modified.loc[300, "sales"] = np.nan
    modified["is_missing_date"] = (
        modified["sales"].isna().astype(int)
    )

    with pytest.raises(ValueError, match="Missing actual sales"):
        run_dynamic_backtest(
            modified, windows=4, horizon=30
        )


def test_input_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    run_dynamic_backtest(
        sample_data, windows=2, horizon=30
    )

    pd.testing.assert_frame_equal(sample_data, original)


def test_future_actuals_do_not_change_earlier_results(sample_data):
    modified = sample_data.copy(deep=True)
    modified.loc[335:, "sales"] *= 3

    original_details, _ = run_dynamic_backtest(
        sample_data, windows=4, horizon=30
    )

    modified_details, _ = run_dynamic_backtest(
        modified, windows=4, horizon=30
    )

    original_early = original_details[
        original_details["window"] < 4
    ].reset_index(drop=True)

    modified_early = modified_details[
        modified_details["window"] < 4
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_early,
        modified_early,
    )
