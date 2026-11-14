import json
import subprocess
import sys

import pandas as pd

from src.config import (
    PROJECT_ROOT,
    METRICS_DIR,
    PREDICTIONS_DIR,
)

REQUIRED_ARTIFACTS = [
    METRICS_DIR / "model_selection.json",
    METRICS_DIR / "final_test_metrics.json",
    METRICS_DIR / "validation_metrics.csv",
    METRICS_DIR / "backtest_summary.csv",
    PREDICTIONS_DIR / "final_test_predictions.csv",
]


def run_module(module):
    print(f"\nRunning {module}...", flush=True)

    subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        check=True,
    )


def lock_model_selection():
    summary = pd.read_csv(
        METRICS_DIR / "backtest_summary.csv"
    )

    validation = pd.read_csv(
        METRICS_DIR / "validation_metrics.csv"
    )

    best = summary.sort_values("MAE").iloc[0]
    selected = str(best["model"])

    if selected != "xgboost":
        raise RuntimeError(
            f"Selected model is {selected}. "
            "Final test currently supports xgboost only."
        )

    validation_row = validation.loc[
        validation["model"] == selected
    ]

    if len(validation_row) != 1:
        raise ValueError("Invalid validation metrics")

    selection = {
        "selected_model": selected,
        "selection_metric": "MAE",
        "forecast_horizon": 30,
        "validation_mae": float(validation_row.iloc[0]["MAE"]),
        "backtest_mean_mae": float(best["MAE"]),
        "selection_status": "locked_before_final_test",
        "test_evaluated": False,
    }

    with open(
        METRICS_DIR / "model_selection.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(selection, file, indent=4)

    print("Model selection locked:", selected)


def main():
    existing = [
        str(path)
        for path in REQUIRED_ARTIFACTS
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "Bootstrap requires a fresh workspace. "
            "Existing evaluation artifacts were not modified:\n"
            + "\n".join(existing)
        )

    run_module("src.data_preprocessing")
    run_module("src.feature_engineering")
    run_module("src.train")
    run_module("src.evaluate")
    run_module("src.backtest")

    lock_model_selection()

    run_module("src.final_test")
    run_module("src.production_train")
    run_module("src.forecast")
    run_module("src.visualize")

    print("\nBOOTSTRAP COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
