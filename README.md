# Future Sales Forecasting

An end-to-end machine learning project for forecasting retail sales using Python, XGBoost, and time-series analysis.

## Project Overview

This project predicts aggregate daily retail sales for a 30-day horizon using historical sales data from the Kaggle Store Sales competition.

## Technologies

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Pytest

## Dataset

Source: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

- 3,000,888 training records
- 54 stores
- 33 product families
- Historical period: 2013-01-01 to 2017-08-15

## Model Performance

| Model | Validation MAE | Validation WAPE |
|---|---:|---:|
| XGBoost | 40,017.30 | 4.64% |
| Seasonal Naive | 67,195.81 | 7.78% |
| Ridge Regression | 72,977.61 | 8.45% |

### Final Holdout Test

- MAE: 48,175.14
- RMSE: 69,347.60
- WAPE: 5.78%

## 30-Day Forecast

- Forecast period: 2017-08-16 to 2017-09-14
- Total predicted sales: 25,735,076.56
- Average daily predicted sales: 857,835.89

These are historical demonstration forecasts, not live forecasts.

## Installation

```powershell
git clone https://github.com/azezsarisari/future-sales-forecasting.git
cd future-sales-forecasting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

## Dataset Setup

Download the dataset from:
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Place the CSV files inside data/raw/.

The original dataset is not included in this repository.

## Reproduce the Project

For a fresh workspace, run:

    python -m src.bootstrap

This runs preprocessing, feature engineering, training, validation, backtesting, final evaluation, forecasting, and visualization.

For an initialized workspace, run:

    python -m src.pipeline

## Streamlit Dashboard

Start the dashboard:

    python -m streamlit run app/dashboard.py --server.address 127.0.0.1

Open http://localhost:8501

Dashboard pages:
- Overview
- Historical Analysis
- Model Performance
- 30-Day Forecast

## Automated Tests

Run:

    python -m pytest tests/ -q

All 19 automated tests passed during development.

## Reproducibility

The project was tested in a separate workspace and a fresh Python environment.

## Limitations

- Historical data ends in August 2017.
- Forecasts represent aggregate daily sales.
- No live data ingestion is implemented.
- The automated final-test workflow currently supports XGBoost.
- Forecast accuracy on new data is not guaranteed.

## License

MIT License.

