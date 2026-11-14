from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

DAILY_SALES_PATH = PROCESSED_DATA_DIR / "daily_sales.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"

FORECAST_HORIZON = 30
RANDOM_STATE = 42

LAG_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
]

ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
]

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
]

FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)

TARGET_COLUMN = "sales"
DATE_COLUMN = "date"

# Chronological split boundaries
TRAIN_END = "2017-06-16"
VALIDATION_START = "2017-06-17"
VALIDATION_END = "2017-07-16"
TEST_START = "2017-07-17"
TEST_END = "2017-08-15"
