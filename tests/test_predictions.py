import numpy as np
import pandas as pd
import pytest

from src.predict import recursive_forecast


class DummyModel:
    def predict(self, X):
        return np.full(len(X), 100.0)


class NegativeModel:
    def predict(self, X):
        return np.full(len(X), -50.0)


class TrackingModel:
    def __init__(self):
        self.inputs = []

    def predict(self, X):
        self.inputs.append(X.copy())
        return np.full(len(X), 100.0)


def make_history():
    return pd.DataFrame({
        "date": pd.date_range(
            "2024-01-01",
            periods=60,
            freq="D"
        ),
        "sales": np.arange(1, 61, dtype=float),
    })


def test_forecast_horizon():
    result = recursive_forecast(
        DummyModel(), make_history(), horizon=30
    )

    assert len(result) == 30
    assert result["date"].is_unique


def test_forecast_dates():
    result = recursive_forecast(
        DummyModel(), make_history(), horizon=30
    )

    assert result["date"].min() == pd.Timestamp("2024-03-01")
    assert result["date"].max() == pd.Timestamp("2024-03-30")


def test_forecast_values():
    result = recursive_forecast(
        DummyModel(), make_history(), horizon=30
    )

    assert result["predicted_sales"].eq(100.0).all()


def test_negative_predictions_clipped():
    result = recursive_forecast(
        NegativeModel(), make_history(), horizon=7
    )

    assert result["predicted_sales"].eq(0.0).all()


def test_recursive_predictions_used():
    model = TrackingModel()

    recursive_forecast(
        model, make_history(), horizon=2
    )

    assert len(model.inputs) == 2
    assert model.inputs[0]["lag_1"].iloc[0] == 60.0
    assert model.inputs[1]["lag_1"].iloc[0] == 100.0


def test_invalid_horizon():
    with pytest.raises(ValueError, match="positive"):
        recursive_forecast(
            DummyModel(), make_history(), horizon=0
        )


def test_insufficient_history():
    history = make_history().head(10)

    with pytest.raises(ValueError, match="Last 28"):
        recursive_forecast(
            DummyModel(), history, horizon=30
        )
