from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/daily_sales.csv")
OUTPUT_PATH = Path("data/processed/features.csv")

LAGS = [1, 7, 14, 28]
WINDOWS = [7, 14, 28]


def create_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    if not df["date"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError("Calendar must be continuous")

    # Calendar features
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Lag features
    for lag in LAGS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # Rolling features: historical observations only
    shifted_sales = df["sales"].shift(1)

    for window in WINDOWS:
        df[f"rolling_mean_{window}"] = (
            shifted_sales
            .rolling(window=window, min_periods=window)
            .mean()
        )

    return df


def main():
    print("=" * 55)
    print("FEATURE ENGINEERING")
    print("=" * 55)

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"]
    )

    features = create_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Input rows:", len(df))
    print("Output rows:", len(features))
    print("Total columns:", len(features.columns))
    print("Feature columns:")

    for column in features.columns:
        print(" -", column)

    print("\nMissing values:")
    print(features.isna().sum().to_string())

    print("\nOutput:", OUTPUT_PATH)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
