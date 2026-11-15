from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.feature_engineering import create_features
from src.training_readiness import validate_training_readiness


def prepare_dynamic_features(processed_data):
    readiness = validate_training_readiness(processed_data)

    features = create_features(processed_data)

    training_rows = features.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    if training_rows.empty:
        raise ValueError("No usable training rows")

    if len(training_rows) < 60:
        raise ValueError(
            "At least 60 usable training rows are required"
        )

    report = {
        **readiness,
        "total_feature_rows": len(features),
        "usable_training_rows": len(training_rows),
        "excluded_training_rows": len(features) - len(training_rows),
        "feature_count": len(FEATURE_COLUMNS),
    }

    return features, training_rows, report
