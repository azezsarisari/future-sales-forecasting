import pandas as pd


MIN_HISTORY_DAYS = 120


def validate_sales_data(
    df,
    date_column="date",
    sales_column="sales",
):
    if df.empty:
        raise ValueError("Uploaded dataset is empty")

    if date_column == sales_column:
        raise ValueError("Date and sales columns must be different")

    required = [date_column, sales_column]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = df[required].copy()

    data.columns = ["date", "sales"]

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    if data["date"].isna().any():
        raise ValueError("Invalid dates found")

    if not data["date"].dt.normalize().eq(
        data["date"]
    ).all():
        raise ValueError(
            "Dates must not contain time components"
        )

    data["sales"] = pd.to_numeric(
        data["sales"],
        errors="coerce"
    )

    if data["sales"].isna().any():
        raise ValueError(
            "Invalid or missing sales values"
        )

    if not data["sales"].map(
        lambda value: float("-inf") < value < float("inf")
    ).all():
        raise ValueError("Non-finite sales values found")

    if (data["sales"] < 0).any():
        raise ValueError("Negative sales values found")

    daily = (
        data.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    calendar = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D"
    )

    if len(calendar) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"At least {MIN_HISTORY_DAYS} calendar days are required"
        )

    processed = (
        daily.set_index("date")
        .reindex(calendar)
        .rename_axis("date")
        .reset_index()
    )

    processed["is_missing_date"] = (
        processed["sales"].isna().astype(int)
    )

    return processed
