import os
import tempfile
import pandas as pd


def _read_csv_smart(path: str, **kwargs) -> pd.DataFrame:
    """
    Try multiple encodings in order.
    Most CSVs are utf-8, but older datasets (spam, emails, 
    european data) are often latin-1 or cp1252.
    """
    encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, Exception):
            continue
    # last resort — ignore undecodable bytes
    return pd.read_csv(path, encoding="utf-8", errors="ignore", **kwargs)


def get_columns(path: str) -> list[str]:
    if path.lower().endswith(".csv"):
        return list(_read_csv_smart(path, nrows=1).columns)
    if path.lower().endswith((".xlsx", ".xls")):
        return list(pd.read_excel(path, nrows=1).columns)
    raise ValueError("Only CSV and Excel files are supported.")


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def validate_dataset(path: str, target_column: str):
    """
    Returns (is_valid, message, preview_df).
    """
    try:
        if path.lower().endswith(".csv"):
            df = _read_csv_smart(path)
        else:
            df = pd.read_excel(path)

        if target_column not in df.columns:
            return (
                False,
                f"Target column '{target_column}' not found in dataset.",
                None,
            )

        if df.shape[0] < 10:
            return (
                False,
                f"Dataset too small: only {df.shape[0]} rows.",
                None,
            )

        return (
            True,
            f"Valid dataset: {df.shape[0]} rows, {df.shape[1]} columns.",
            df,
        )

    except Exception as exc:
        return False, f"Could not read file: {exc}", None