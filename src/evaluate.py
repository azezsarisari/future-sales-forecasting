import json

import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FORECAST_HORIZON,
)

from src.train import load_training_data, split_data
from src.baseline import (
    seasonal_naive_forecast,
    evaluate_forecast,
)
from src.predict import recursive_forecast


def main():
    print("=" * 55)
    print("MODEL VALIDATION EVALUATION")
    print("=" * 55)

    df = load_training_data()
    train, validation, test = split_data(df)

    assert len(validation) == FORECAST_HORIZON
    assert train["date"].max() < validation["date"].min()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    predictions_df = validation[["date", "sales"]].copy()
    predictions_df = predictions_df.rename(
        columns={"sales": "actual_sales"}
    )

    # Seasonal Naive baseline
    baseline_predictions = seasonal_naive_forecast(
        train,
        horizon=FORECAST_HORIZON
    )

    baseline_metrics = evaluate_forecast(
        validation["sales"],
        baseline_predictions
    )

    results.append({
        "model": "seasonal_naive",
        **baseline_metrics
    })

    predictions_df["seasonal_naive"] = baseline_predictions

    # Machine learning models
    for name in ["ridge", "xgboost"]:
        print(f"\nEvaluating {name}...")

        model_path = MODELS_DIR / f"{name}_validation.joblib"

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        model = joblib.load(model_path)

        forecast = recursive_forecast(
            model,
            train[["date", "sales"]],
            horizon=FORECAST_HORIZON
        )

        if not forecast["date"].reset_index(drop=True).equals(
            validation["date"].reset_index(drop=True)
        ):
            raise ValueError("Forecast dates do not match validation")

        predicted = forecast["predicted_sales"].to_numpy()

        metrics = evaluate_forecast(
            validation["sales"],
            predicted
        )

        results.append({
            "model": name,
            **metrics
        })

        predictions_df[name] = predicted

    # Compare models
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "MAE"
    ).reset_index(drop=True)

    print("\nMODEL COMPARISON:")
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]["model"]

    print("\nBest validation model:", best_model)

    # Save evaluation results
    results_df.to_csv(
        METRICS_DIR / "validation_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_DIR / "validation_predictions.csv",
        index=False
    )

    with open(
        METRICS_DIR / "best_validation_model.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "best_model": best_model,
                "selection_metric": "MAE",
                "forecast_horizon": FORECAST_HORIZON
            },
            file,
            indent=4
        )

    print("\nValidation results saved.")
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
