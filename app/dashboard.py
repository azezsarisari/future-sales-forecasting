from pathlib import Path
import json

import pandas as pd
import streamlit as st

from app.upload_page import render_upload_page

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

st.set_page_config(
    page_title="Future Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    history = pd.read_csv(
        DATA_DIR / "processed" / "daily_sales.csv",
        parse_dates=["date"]
    )

    forecast = pd.read_csv(
        DATA_DIR / "predictions" / "future_30_days.csv",
        parse_dates=["date"]
    )

    test = pd.read_csv(
        DATA_DIR / "predictions" / "final_test_predictions.csv",
        parse_dates=["date"]
    )

    comparison = pd.read_csv(
        METRICS_DIR / "validation_metrics.csv"
    )

    backtest = pd.read_csv(
        METRICS_DIR / "backtest_summary.csv"
    )

    with open(
        METRICS_DIR / "final_test_metrics.json",
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    return history, forecast, test, comparison, backtest, metrics


try:
    history, forecast, test, comparison, backtest, metrics = load_data()
except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
    st.error(f"Unable to load project data: {exc}")
    st.stop()


st.sidebar.title("Sales Forecasting")
st.sidebar.caption("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Historical Analysis",
        "Model Performance",
        "30-Day Forecast",
        "Upload & Forecast",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "Dataset: Kaggle Store Sales\n\n"
    "Model: XGBoost\n\n"
    "Historical data ends: 2017-08-15\n\n"
    "Forecast horizon: 30 days"
)


if page == "Overview":
    st.title("Future Sales Forecasting")
    st.caption("Historical sales analysis and machine learning predictions")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Historical Days",
        f"{len(history):,}"
    )

    col2.metric(
        "Average Daily Sales",
        f"{history['sales'].mean():,.0f}"
    )

    col3.metric(
        "30-Day Forecast Total",
        f"{forecast['predicted_sales'].sum():,.0f}"
    )

    col4.metric(
        "Final Test WAPE",
        f"{metrics['WAPE']:.2f}%"
    )

    st.subheader("Historical Sales")

    st.line_chart(
        history.set_index("date")["sales"]
    )

    st.subheader("Forecast Summary")

    st.write(
        "The XGBoost model forecasts daily sales "
        "for the 30 days following August 15, 2017."
    )

    st.info(
        "This is a historical forecasting demonstration, "
        "not a live 2026 sales forecast."
    )


elif page == "Historical Analysis":
    st.title("Historical Sales Analysis")

    st.subheader("Daily Sales Trend")

    st.line_chart(
        history.set_index("date")["sales"]
    )

    st.subheader("Weekly Seasonality")

    weekly = history.copy()
    weekly["weekday"] = weekly["date"].dt.day_of_week

    weekday_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekly_avg = (
        weekly.groupby("weekday")["sales"]
        .mean()
        .reindex(range(7))
    )

    weekly_avg.index = weekday_names

    st.bar_chart(weekly_avg)

    st.subheader("Missing Historical Dates")

    missing = history[
        history["is_missing_date"] == 1
    ]

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )


elif page == "Model Performance":
    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Final Test MAE", f"{metrics['MAE']:,.2f}")
    col2.metric("Final Test RMSE", f"{metrics['RMSE']:,.2f}")
    col3.metric("Final Test WAPE", f"{metrics['WAPE']:.2f}%")

    st.subheader("Final Holdout: Actual vs Predicted")

    chart_data = test.set_index("date")[
        ["actual_sales", "predicted_sales"]
    ]

    st.line_chart(chart_data)

    st.subheader("Validation Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Walk-forward Backtesting")

    st.dataframe(
        backtest,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The final holdout test covers July 17 to August 15, 2017."
    )


elif page == "30-Day Forecast":
    st.title("30-Day Sales Forecast")

    st.caption(
        "Forecast period: August 16 to September 14, 2017"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Forecast",
        f"{forecast['predicted_sales'].sum():,.2f}"
    )

    col2.metric(
        "Daily Average",
        f"{forecast['predicted_sales'].mean():,.2f}"
    )

    col3.metric(
        "Forecast Days",
        len(forecast)
    )

    st.subheader("Historical vs Forecast")

    recent = history.tail(90)[["date", "sales"]].copy()
    recent = recent.rename(
        columns={"sales": "Historical Sales"}
    )
    recent["Forecast Sales"] = float("nan")

    future = forecast.rename(
        columns={"predicted_sales": "Forecast Sales"}
    ).copy()
    future["Historical Sales"] = float("nan")

    combined = pd.concat(
        [recent, future],
        ignore_index=True
    ).set_index("date")

    st.line_chart(
        combined[["Historical Sales", "Forecast Sales"]]
    )

    st.subheader("Daily Predictions")

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="future_30_days.csv",
        mime="text/csv",
    )

    st.info(
        "Predictions are generated from historical data ending "
        "August 15, 2017. They are not live forecasts."
    )


elif page == "Upload & Forecast":
    render_upload_page()
