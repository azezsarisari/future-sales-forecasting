import streamlit as st
import pandas as pd

from src.upload_pipeline import (
    inspect_uploaded_file,
    process_uploaded_file,
)
from src.dynamic_forecast import generate_dynamic_forecast
from src.backtest_service import evaluate_backtest_availability


def render_upload_page():
    st.title("Upload & Forecast")
    st.caption(
        "Upload your sales dataset, evaluate forecasting models, "
        "and generate predictions for the next 7, 14, or 30 days."
    )

    st.info(
        "Supported formats: CSV and XLSX. Maximum file size: 20 MB. "
        "At least 120 calendar days are required, with sufficient "
        "observed sales and a complete recent history."
    )

    uploaded_file = st.file_uploader(
        "Upload Sales Dataset",
        type=["csv", "xlsx"],
        key="dynamic_sales_upload",
    )

    if uploaded_file is None:
        st.stop()

    content = uploaded_file.getvalue()

    try:
        inspection = inspect_uploaded_file(
            uploaded_file.name,
            content,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(
        inspection["preview"],
        use_container_width=True,
        hide_index=True,
    )

    columns = inspection["columns"]

    date_index = columns.index("date") if "date" in columns else 0
    sales_index = columns.index("sales") if "sales" in columns else 0

    col1, col2 = st.columns(2)

    with col1:
        date_column = st.selectbox(
            "Date Column",
            columns,
            index=date_index,
        )

    with col2:
        sales_column = st.selectbox(
            "Sales Column",
            columns,
            index=sales_index,
        )

    horizon = st.selectbox(
        "Forecast Horizon",
        [7, 14, 30],
        index=2,
    )

    if not st.button(
        "Train Models & Generate Forecast",
        type="primary",
    ):
        return

    try:
        with st.spinner("Validating data and training models..."):
            processed, quality_report = process_uploaded_file(
                uploaded_file.name,
                content,
                date_column,
                sales_column,
            )

            forecast, comparison, validation, report = (
                generate_dynamic_forecast(
                    processed,
                    horizon=horizon,
                    validation_days=30,
                )
            )

    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Forecasting failed: {exc}")
        return

    st.success("Forecast generated successfully.")

    st.subheader("Data Quality Report")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Raw Records", quality_report["raw_rows"])
    c2.metric("Calendar Days", quality_report["calendar_days"])
    c3.metric("Observed Days", quality_report["observed_days"])
    c4.metric("Missing Days", quality_report["missing_days"])

    st.subheader("Model Comparison")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.metric(
        "Selected Model",
        report["selected_model"].replace("_", " ").title(),
    )

    st.subheader("Validation: Actual vs Predicted")

    st.line_chart(
        validation.set_index("date")[
            ["actual_sales", "seasonal_naive", "xgboost"]
        ]
    )

    st.subheader("Walk-Forward Backtesting")

    with st.spinner("Running historical backtesting..."):
        backtest_result = evaluate_backtest_availability(
            processed,
            windows=4,
            horizon=30,
        )

    if backtest_result["available"]:
        st.success("Four backtesting windows completed.")

        st.subheader("Average Backtesting Performance")

        st.dataframe(
            backtest_result["summary"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Individual Backtesting Windows")

        st.dataframe(
            backtest_result["details"],
            use_container_width=True,
            hide_index=True,
        )

        best_model = backtest_result["summary"].iloc[0]["model"]

        st.metric(
            "Best Average Backtesting Model",
            best_model.replace("_", " ").title(),
        )

    else:
        st.warning(
            "Backtesting is unavailable for this dataset: "
            + backtest_result["reason"]
        )

    st.subheader("Future Sales Forecast")

    st.line_chart(
        forecast.set_index("date")["predicted_sales"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Predicted Sales",
        f'{report["total_predicted_sales"]:,.2f}',
    )

    c2.metric(
        "Average Daily Sales",
        f'{report["average_predicted_sales"]:,.2f}',
    )

    c3.metric("Forecast Days", horizon)

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="dynamic_sales_forecast.csv",
        mime="text/csv",
    )

    st.caption(
        "Forecast accuracy depends on the quality and history "
        "of the uploaded dataset. Validation results do not "
        "guarantee future performance."
    )
