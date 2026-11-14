from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DAILY_SALES_PATH,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(
        DAILY_SALES_PATH,
        parse_dates=["date"]
    ).sort_values("date")

    forecast = pd.read_csv(
        PREDICTIONS_DIR / "future_30_days.csv",
        parse_dates=["date"]
    ).sort_values("date")

    test = pd.read_csv(
        PREDICTIONS_DIR / "final_test_predictions.csv",
        parse_dates=["date"]
    ).sort_values("date")

    # Figure 1: Historical daily sales
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(history["date"], history["sales"], linewidth=0.8)
    ax.set_title("Historical Daily Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "historical_sales.png", dpi=160)
    plt.close(fig)

    # Figure 2: Average sales by weekday
    weekday = history.copy()
    weekday["day_of_week"] = weekday["date"].dt.day_of_week

    weekday_avg = (
        weekday.groupby("day_of_week")["sales"]
        .mean()
        .reindex(range(7))
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        weekday_avg
    )
    ax.set_title("Average Sales by Day of Week")
    ax.set_ylabel("Average Sales")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "weekday_sales.png", dpi=160)
    plt.close(fig)

    # Figure 3: Final holdout actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        test["date"], test["actual_sales"],
        label="Actual", linewidth=2
    )
    ax.plot(
        test["date"], test["predicted_sales"],
        label="Predicted", linewidth=2
    )
    ax.set_title("Final Holdout: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_comparison.png", dpi=160)
    plt.close(fig)

    # Figure 4: Historical sales and future forecast
    recent = history.tail(90)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        recent["date"], recent["sales"],
        label="Historical Sales", linewidth=1.5
    )
    ax.plot(
        forecast["date"], forecast["predicted_sales"],
        label="30-Day Forecast", linewidth=2
    )
    ax.axvline(
        history["date"].max(),
        linestyle="--", alpha=0.6
    )
    ax.set_title("Future Sales Forecast - 30 Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "future_forecast.png", dpi=160)
    plt.close(fig)

    print("VISUALIZATION COMPLETED SUCCESSFULLY")
    print("\nGenerated figures:")

    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(path.name, "-", path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
