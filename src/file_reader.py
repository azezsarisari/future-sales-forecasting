from io import BytesIO
from pathlib import Path

import pandas as pd


MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def read_uploaded_file(filename, content):
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file format. Use CSV or XLSX."
        )

    if not isinstance(content, bytes):
        raise ValueError("File content must be bytes")

    if len(content) == 0:
        raise ValueError("Uploaded file is empty")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File exceeds {MAX_FILE_SIZE_MB} MB limit"
        )

    buffer = BytesIO(content)

    try:
        if extension == ".csv":
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(
                buffer,
                engine="openpyxl"
            )
    except Exception as exc:
        raise ValueError(
            "Unable to read uploaded file"
        ) from exc

    if df.empty:
        raise ValueError("Uploaded dataset is empty")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found")

    return df
