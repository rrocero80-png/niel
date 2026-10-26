"""Module 1 helpers: CSV loading, schema validation, null/duplicate summary."""
from __future__ import annotations

import pandas as pd

from .config import (CATEGORY_VALUES, FEATURES, LEAKAGE_COLUMNS, NUMERIC,
                     NUMERIC_BOUNDS, TARGET)


def load_csv(file) -> tuple[pd.DataFrame | None, str | None]:
    """Read an uploaded CSV. Returns (df, error_message)."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return None, "The file is empty or has no header row."
    except pd.errors.ParserError as e:
        return None, f"The file could not be parsed as CSV: {e}"
    except UnicodeDecodeError:
        return None, "The file is not valid UTF-8 text. Re-export it as CSV (UTF-8)."
    except Exception as e:  # noqa: BLE001 - show a friendly message to the user
        return None, f"Could not read the file: {e}"
    if df.empty:
        return None, "The file has a header but no data rows."
    df.columns = [str(c).strip() for c in df.columns]
    return df, None


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> dict:
    """Check required columns, types, categories, and ranges.

    Returns dict with: errors (blocking), warnings, leakage_found.
    """
    errors, warnings = [], []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if require_target and TARGET not in df.columns:
        errors.append(f"Missing target column '{TARGET}' (needed for class distribution / metrics).")

    for col in NUMERIC:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric.")
        elif col in df.columns:
            lo, hi = NUMERIC_BOUNDS[col]
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                warnings.append(f"'{col}' has {bad} value(s) outside the trained range {lo}-{hi}.")

    for col, allowed in CATEGORY_VALUES.items():
        if col in df.columns:
            unseen = sorted(set(df[col].dropna().astype(str)) - set(allowed))
            if unseen:
                warnings.append(f"'{col}' has categories unseen in training: {unseen[:5]} "
                                "(model will ignore them).")

    leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leakage:
        warnings.append(f"Post-outcome (leakage) columns found and will be dropped: {leakage}")

    if TARGET in df.columns and not df[TARGET].dropna().isin([0, 1]).all():
        errors.append(f"Target '{TARGET}' must contain only 0/1.")

    return {"errors": errors, "warnings": warnings, "leakage_found": leakage}


def clean_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the training features, in training order; drops leakage and the target."""
    return df[FEATURES].copy()


def quality_summary(df: pd.DataFrame) -> dict:
    nulls = df.isna().sum()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "null_table": nulls[nulls > 0].rename("missing").to_frame().assign(
            pct=lambda t: (t["missing"] / max(len(df), 1) * 100).round(2)),
        "total_nulls": int(nulls.sum()),
    }
