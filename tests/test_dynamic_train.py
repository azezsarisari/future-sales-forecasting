import numpy as np
import pandas as pd
import pytest

from src.dynamic_train import train_dynamic_model


@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(42)
    days = np.arange(180)

    sales = (
        1000
        + days * 3
        + 120 * np.sin(2 * np.pi * days / 7)
        + rng.normal(0, 20, 180)
    )

    return pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01",
            periods=180,
            freq="D"
        ),
        "sales": np.maximum(sales, 0),
        "is_missing_date": [0] * 180,
    })


@pytest.fixture(scope="module")
def training_result(sample_data):
    return train_dynamic_model(
        sample_data,
        validation_days=30,
    )


def test_model_training(training_result):
    model, _, _, _ = training_result

    assert hasattr(model, "predict")
    assert hasattr(model, "feature_importances_")


def test_model_comparison(training_result):
    _, comparison, _, _ = training_result

    assert set(comparison["model"]) == {
        "xgboost",
        "seasonal_naive",
    }

    assert len(comparison) == 2
    assert comparison["MAE"].notna().all()
    assert comparison["RMSE"].notna().all()
    assert comparison["WAPE"].notna().all()


def test_validation_predictions(training_result):
    _, _, predictions, _ = training_result

    assert len(predictions) == 30
    assert predictions["date"].is_unique
    assert predictions["xgboost"].notna().all()
    assert predictions["seasonal_naive"].notna().all()


def test_nonnegative_predictions(training_result):
    _, _, predictions, _ = training_result

    assert predictions["xgboost"].ge(0).all()
    assert predictions["seasonal_naive"].ge(0).all()


def test_chronological_boundaries(training_result):
    _, _, predictions, report = training_result

    assert report["training_end"] == "2026-05-30"
    assert report["validation_start"] == "2026-05-31"
    assert report["validation_end"] == "2026-06-29"

    assert predictions["date"].min() == pd.Timestamp(
        "2026-05-31"
    )


def test_training_rows(training_result):
    _, _, _, report = training_result

    assert report["training_rows"] == 122
    assert report["validation_days"] == 30


@pytest.mark.parametrize("days", [0, -1, 31, 100])
def test_invalid_validation_horizon(sample_data, days):
    with pytest.raises(ValueError, match="between 1 and 30"):
        train_dynamic_model(
            sample_data,
            validation_days=days,
        )


@pytest.mark.parametrize("days", [1.5, "30", True, None])
def test_invalid_validation_type(sample_data, days):
    with pytest.raises(ValueError, match="integer"):
        train_dynamic_model(
            sample_data,
            validation_days=days,
        )


def test_input_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    train_dynamic_model(sample_data, validation_days=7)

    pd.testing.assert_frame_equal(sample_data, original)


def test_validation_actuals_not_used_for_training(sample_data):
    modified = sample_data.copy(deep=True)

    modified.loc[150:, "sales"] *= 5

    _, _, original_predictions, _ = train_dynamic_model(
        sample_data,
        validation_days=30,
    )

    _, _, modified_predictions, _ = train_dynamic_model(
        modified,
        validation_days=30,
    )

    np.testing.assert_allclose(
        original_predictions["xgboost"],
        modified_predictions["xgboost"],
        rtol=0,
        atol=1e-6,
    )
