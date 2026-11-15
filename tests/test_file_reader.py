from io import BytesIO

import pandas as pd
import pytest

from src.file_reader import (
    read_uploaded_file,
    MAX_FILE_SIZE_BYTES,
)


def test_valid_csv():
    content = b"date,sales\n2026-01-01,100\n2026-01-02,200\n"

    result = read_uploaded_file("sales.csv", content)

    assert len(result) == 2
    assert result["sales"].sum() == 300


def test_valid_xlsx():
    source = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "sales": [100, 200],
    })

    buffer = BytesIO()
    source.to_excel(buffer, index=False, engine="openpyxl")

    result = read_uploaded_file("sales.xlsx", buffer.getvalue())

    assert len(result) == 2
    assert result["sales"].sum() == 300


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        read_uploaded_file("sales.txt", b"date,sales\n")


def test_empty_file():
    with pytest.raises(ValueError, match="empty"):
        read_uploaded_file("sales.csv", b"")


def test_invalid_content_type():
    with pytest.raises(ValueError, match="bytes"):
        read_uploaded_file("sales.csv", "not bytes")


def test_corrupted_csv():
    with pytest.raises(ValueError, match="Unable to read"):
        read_uploaded_file("sales.csv", b"\xff\xfe\xfa")


def test_corrupted_xlsx():
    with pytest.raises(ValueError, match="Unable to read"):
        read_uploaded_file("sales.xlsx", b"not an excel file")


def test_file_size_limit():
    oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)

    with pytest.raises(ValueError, match="exceeds"):
        read_uploaded_file("sales.csv", oversized)


def test_header_only_csv():
    with pytest.raises(ValueError, match="empty"):
        read_uploaded_file("sales.csv", b"date,sales\n")


def test_uppercase_extension():
    content = b"date,sales\n2026-01-01,100\n"

    result = read_uploaded_file("SALES.CSV", content)

    assert len(result) == 1
