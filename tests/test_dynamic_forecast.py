import numpy as np
import pandas as pd
import pytest

from src.dynamic_forecast import generate_dynamic_forecast


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
            "2026-01-01", periods=180, freq="D"
        ),
        "sales": np.maximum(sales, 0),
        "is_missing_date": [0] * 180,
    })


@pytest.fixture(scope="module")
def forecast_result(sample_data):
    return generate_dynamic_forecast(
        sample_data, horizon=30, validation_days=30
    )


@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_supported_horizons(sample_data, horizon):
    forecast, _, _, report = generate_dynamic_forecast(
        sample_data, horizon=horizon
    )
    assert len(forecast) == horizon
    assert report["forecast_horizon"] == horizon


def test_forecast_dates(forecast_result):
    forecast, _, _, report = forecast_result

    assert forecast["date"].is_unique
    assert forecast["date"].min() == pd.Timestamp("2026-06-30")
    assert forecast["date"].max() == pd.Timestamp("2026-07-29")
    assert report["forecast_start"] == "2026-06-30"


def test_forecast_values(forecast_result):
    forecast, _, _, _ = forecast_result

    values = forecast["predicted_sales"].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_model_selection(forecast_result):
    _, comparison, _, report = forecast_result

    assert report["selected_model"] in {
        "xgboost", "seasonal_naive"
    }
    assert report["selected_model"] == comparison.iloc[0]["model"]


def test_validation_predictions(forecast_result):
    _, _, validation, _ = forecast_result

    assert len(validation) == 30
    assert validation["actual_sales"].notna().all()
    assert validation["xgboost"].notna().all()
    assert validation["seasonal_naive"].notna().all()


def test_forecast_summary(forecast_result):
    forecast, _, _, report = forecast_result

    assert report["total_predicted_sales"] == pytest.approx(
        forecast["predicted_sales"].sum(), abs=0.01
    )


@pytest.mark.parametrize("horizon", [0, 1, 8, 31, -7, True, "30"])
def test_invalid_horizons(sample_data, horizon):
    with pytest.raises(ValueError, match="7, 14, or 30"):
        generate_dynamic_forecast(sample_data, horizon=horizon)


@pytest.mark.parametrize("days", [0, -1, 31, True, "30"])
def test_invalid_validation_days(sample_data, days):
    with pytest.raises(ValueError, match="between 1 and 30"):
        generate_dynamic_forecast(
            sample_data, validation_days=days
        )


def test_input_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    generate_dynamic_forecast(sample_data, horizon=7)

    pd.testing.assert_frame_equal(sample_data, original)

def test_seasonal_naive_selected_when_best(sample_data, monkeypatch):
    import src.dynamic_forecast as module

    class FakeModel:
        def fit(self, X, y):
            return self

    def fake_train(data, validation_days=30):
        comparison = pd.DataFrame([
            {"model": "seasonal_naive", "MAE": 10.0},
            {"model": "xgboost", "MAE": 20.0},
        ])

        report = {"best_validation_model": "seasonal_naive"}

        return FakeModel(), comparison, pd.DataFrame(), report

    def forbid_training():
        raise AssertionError(
            "XGBoost must not be trained when baseline wins"
        )

    monkeypatch.setattr(module, "train_dynamic_model", fake_train)
    monkeypatch.setattr(module, "build_models", forbid_training)

    forecast, comparison, _, report = (
        module.generate_dynamic_forecast(
            sample_data,
            horizon=7,
            validation_days=30,
        )
    )

    assert report["selected_model"] == "seasonal_naive"
    assert len(forecast) == 7
    assert forecast["predicted_sales"].ge(0).all()

    expected = sample_data["sales"].tail(7).to_numpy()

    np.testing.assert_allclose(
        forecast["predicted_sales"].to_numpy(),
        expected,
    )
