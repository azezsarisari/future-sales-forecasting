import pandas as pd
import pytest

import src.backtest_service as service


def test_successful_backtest(monkeypatch):
    expected_details = pd.DataFrame({
        "window": [1, 1],
        "model": ["xgboost", "seasonal_naive"],
    })

    expected_summary = pd.DataFrame({
        "model": ["xgboost", "seasonal_naive"],
        "MAE": [100.0, 120.0],
    })

    def fake_backtest(data, windows, horizon):
        assert windows == 4
        assert horizon == 30
        return expected_details, expected_summary

    monkeypatch.setattr(
        service, "run_dynamic_backtest", fake_backtest
    )

    result = service.evaluate_backtest_availability(
        pd.DataFrame(),
        windows=4,
        horizon=30,
    )

    assert result["available"] is True
    assert result["reason"] is None
    pd.testing.assert_frame_equal(
        result["details"], expected_details
    )
    pd.testing.assert_frame_equal(
        result["summary"], expected_summary
    )


def test_missing_history_handled(monkeypatch):
    def fake_backtest(data, windows, horizon):
        raise ValueError("Insufficient historical data")

    monkeypatch.setattr(
        service, "run_dynamic_backtest", fake_backtest
    )

    result = service.evaluate_backtest_availability(
        pd.DataFrame()
    )

    assert result["available"] is False
    assert "Insufficient historical" in result["reason"]
    assert result["details"] is None
    assert result["summary"] is None


def test_missing_training_days_handled(monkeypatch):
    def fake_backtest(data, windows, horizon):
        raise ValueError(
            "Missing sales in last 28 training days"
        )

    monkeypatch.setattr(
        service, "run_dynamic_backtest", fake_backtest
    )

    result = service.evaluate_backtest_availability(
        pd.DataFrame()
    )

    assert result["available"] is False
    assert "last 28 training days" in result["reason"]


def test_unexpected_error_not_hidden(monkeypatch):
    def fake_backtest(data, windows, horizon):
        raise RuntimeError("Unexpected model failure")

    monkeypatch.setattr(
        service, "run_dynamic_backtest", fake_backtest
    )

    with pytest.raises(RuntimeError, match="Unexpected"):
        service.evaluate_backtest_availability(
            pd.DataFrame()
        )


def test_backtest_arguments_forwarded(monkeypatch):
    received = {}

    def fake_backtest(data, windows, horizon):
        received["windows"] = windows
        received["horizon"] = horizon
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        service, "run_dynamic_backtest", fake_backtest
    )

    service.evaluate_backtest_availability(
        pd.DataFrame(),
        windows=2,
        horizon=14,
    )

    assert received == {
        "windows": 2,
        "horizon": 14,
    }
