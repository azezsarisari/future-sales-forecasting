from pathlib import Path
import subprocess

root = Path.cwd()

required = [
    "src/config.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/baseline.py",
    "src/train.py",
    "src/predict.py",
    "src/evaluate.py",
    "src/backtest.py",
    "src/final_test.py",
    "src/production_train.py",
    "src/forecast.py",
    "src/visualize.py",
    "src/pipeline.py",
    "app/dashboard.py",
    "notebooks/01_exploratory_analysis.ipynb",
    "requirements.txt",
    ".gitignore",
    "README.md",
    "LICENSE",
    "data/raw/train.csv",
    "reports/metrics/model_selection.json",
    "reports/metrics/final_test_metrics.json",
    "data/predictions/future_30_days.csv",
]

print("=" * 55)
print("PROJECT QUALITY ASSURANCE")
print("=" * 55)

missing = []

for item in required:
    path = root / item
    exists = path.is_file()

    print(f"{'PASS' if exists else 'FAIL'}: {item}")

    if not exists:
        missing.append(item)

print("\nMissing files:", len(missing))

print("\nGIT STATUS:")
subprocess.run(["git", "status", "--short"], check=True)

print("\nTRACKED FILES:")
subprocess.run(["git", "ls-files"], check=True)

if missing:
    raise SystemExit("QA FAILED: Missing required files")

print("\nINITIAL QA CHECK PASSED")
