from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/train.csv")
OUTPUT_PATH = Path("data/processed/daily_sales.csv")


def load_sales_data():
    df = pd.read_csv(
        RAW_PATH,
        usecols=["date", "sales"],
        parse_dates=["date"]
    )

    if df.empty:
        raise ValueError("Training dataset is empty")

    if df["date"].isna().any():
        raise ValueError("Invalid dates found")

    if df["sales"].isna().any():
        raise ValueError("Missing sales values found")

    if (df["sales"] < 0).any():
        raise ValueError("Negative sales values found")

    return df


def aggregate_daily_sales(df):
    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    return daily


def complete_calendar(daily):
    full_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D"
    )

    complete = (
        daily.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )

    complete["is_missing_date"] = (
        complete["sales"].isna().astype(int)
    )

    return complete


def validate_processed_data(df):
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted")

    if (df["sales"].dropna() < 0).any():
        raise ValueError("Negative sales found")

    expected_dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D"
    )

    if len(df) != len(expected_dates):
        raise ValueError("Calendar is incomplete")

    if df["is_missing_date"].sum() != df["sales"].isna().sum():
        raise ValueError("Missing date flags are inconsistent")


def main():
    print("=" * 55)
    print("SALES DATA PREPROCESSING")
    print("=" * 55)

    raw = load_sales_data()
    print("Raw records:", len(raw))

    daily = aggregate_daily_sales(raw)
    print("Observed days:", len(daily))

    processed = complete_calendar(daily)
    validate_processed_data(processed)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Calendar days:", len(processed))
    print(
        "Missing dates:",
        processed["is_missing_date"].sum()
    )

    print("Output:", OUTPUT_PATH)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
