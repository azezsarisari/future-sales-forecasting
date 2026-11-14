import subprocess
import sys
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    METRICS_DIR,
    PREDICTIONS_DIR,
)

STAGES = [
    ("Data Preprocessing", "src.data_preprocessing"),
    ("Feature Engineering", "src.feature_engineering"),
    ("Production Training", "src.production_train"),
    ("30-Day Forecasting", "src.forecast"),
    ("Visualizations", "src.visualize"),
]

REQUIRED_ARTIFACTS = [
    METRICS_DIR / "model_selection.json",
    METRICS_DIR / "final_test_metrics.json",
    METRICS_DIR / "validation_metrics.csv",
    METRICS_DIR / "backtest_summary.csv",
    PREDICTIONS_DIR / "final_test_predictions.csv",
]


def check_prerequisites():
    missing = [
        str(path)
        for path in REQUIRED_ARTIFACTS
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required evaluation artifacts are missing:\n"
            + "\n".join(missing)
        )


def run_stage(name, module):
    print(f"\n{'=' * 60}", flush=True)
    print(f"RUNNING: {name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f"COMPLETED: {name}", flush=True)


def main():
    print("FUTURE SALES FORECASTING PIPELINE", flush=True)

    check_prerequisites()

    for name, module in STAGES:
        run_stage(name, module)

    print("\nRunning automated tests...", flush=True)

    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("\nPIPELINE COMPLETED SUCCESSFULLY", flush=True)


if __name__ == "__main__":
    main()
