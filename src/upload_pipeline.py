from src.file_reader import read_uploaded_file
from src.upload_validation import validate_sales_data


def inspect_uploaded_file(filename, content):
    """Read a file and return its columns for user mapping."""
    df = read_uploaded_file(filename, content)

    return {
        "columns": df.columns.tolist(),
        "rows": len(df),
        "preview": df.head(10).copy(),
    }


def process_uploaded_file(
    filename,
    content,
    date_column,
    sales_column,
):
    """Validate and aggregate uploaded sales without saving files."""
    raw = read_uploaded_file(filename, content)

    processed = validate_sales_data(
        raw,
        date_column=date_column,
        sales_column=sales_column,
    )

    missing = int(processed["is_missing_date"].sum())
    observed = int(processed["sales"].notna().sum())

    report = {
        "source_file": filename,
        "raw_rows": len(raw),
        "calendar_days": len(processed),
        "observed_days": observed,
        "missing_days": missing,
        "start_date": str(processed["date"].min().date()),
        "end_date": str(processed["date"].max().date()),
        "total_sales": float(processed["sales"].sum()),
        "duplicate_date_rows": int(
            raw.duplicated(subset=[date_column]).sum()
        ),
    }

    return processed, report
