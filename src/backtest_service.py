from src.dynamic_backtest import run_dynamic_backtest


def evaluate_backtest_availability(
    processed_data,
    windows=4,
    horizon=30,
):
    try:
        details, summary = run_dynamic_backtest(
            processed_data,
            windows=windows,
            horizon=horizon,
        )

    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "details": None,
            "summary": None,
        }

    return {
        "available": True,
        "reason": None,
        "details": details,
        "summary": summary,
    }
