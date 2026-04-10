"""
Data-loading helpers and validation utilities.

Handles:
    - NASA C-MAPSS text / CSV / Excel ingestion
    - Missing-value imputation (forward-fill then backward-fill)
    - Duplicate / out-of-order timestamp detection
    - Flat-sensor removal
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import PipelineConfig

# ── C-MAPSS column names (26 raw columns) ────────────────────────────
_CMAPSS_COLS = (
    ["unit_id", "cycle", "op_set_1", "op_set_2", "op_set_3"]
    + [f"s{i}" for i in range(1, 22)]
)


def _read_cmapss_txt(path: Union[str, Path]) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS .txt file."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == len(_CMAPSS_COLS):
        df.columns = _CMAPSS_COLS
    else:
        raise ValueError(
            f"{path}: expected {len(_CMAPSS_COLS)} cols, got {df.shape[1]}"
        )
    return df


def load_cmapss_train(
    path: Union[str, Path],
    fmt: str = "auto",
) -> pd.DataFrame:
    """Load a C-MAPSS *training* file (txt, csv, or xlsx).

    Parameters
    ----------
    path : path-like
        File location.
    fmt : {"auto", "txt", "csv", "xlsx"}
        File format.  ``"auto"`` infers from the extension.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised column names.
    """
    path = Path(path)
    if fmt == "auto":
        fmt = path.suffix.lstrip(".")
    if fmt == "txt":
        return _read_cmapss_txt(path)
    if fmt in ("csv",):
        return pd.read_csv(path)
    if fmt in ("xlsx", "xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {fmt}")


def load_cmapss_excel_all_sheets(
    path: Union[str, Path],
    sheet_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load multiple training sheets from a C-MAPSS Excel file and merge them.

    Each sheet contains engines with ``unit_id`` starting from 1.
    This function offsets the IDs so every engine across all sheets
    gets a globally unique ``unit_id``.

    Parameters
    ----------
    path : path-like
        Path to the Excel file (e.g., ``All_train_data.xlsx``).
    sheet_names : list[str] or None
        Specific sheet names to load.  If None, all sheets are loaded.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with unique ``unit_id`` across all sheets.
    """
    path = Path(path)
    all_sheets = pd.read_excel(path, sheet_name=sheet_names)

    frames = []
    uid_offset = 0
    for sheet_name, df in all_sheets.items():
        # Standardise column name: 'unit id' -> 'unit_id'
        if "unit id" in df.columns and "unit_id" not in df.columns:
            df = df.rename(columns={"unit id": "unit_id"})

        df = df.copy()
        df["unit_id"] = df["unit_id"] + uid_offset
        df["dataset"] = sheet_name
        uid_offset = df["unit_id"].max()

        frames.append(df)
        print(f"  Loaded sheet '{sheet_name}': {df['unit_id'].nunique()} engines, {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {combined['unit_id'].nunique()} engines, {len(combined)} rows")
    return combined


def load_cmapss_test(
    test_path: Union[str, Path],
    rul_path: Union[str, Path],
    fmt: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a C-MAPSS test file *and* its companion RUL ground-truth file.

    Returns (test_df, rul_df) where rul_df has columns
    ``["unit_id", "remaining_rul"]``.
    """
    test_df = load_cmapss_train(test_path, fmt=fmt)
    rul_path = Path(rul_path)
    ext = rul_path.suffix.lstrip(".") if fmt == "auto" else fmt
    if ext == "txt":
        rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
        rul_df = rul_df.iloc[:, 0].to_frame("remaining_rul")
    elif ext == "csv":
        rul_df = pd.read_csv(rul_path)
    elif ext in ("xlsx", "xls"):
        rul_df = pd.read_excel(rul_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    rul_df["unit_id"] = range(1, len(rul_df) + 1)
    return test_df, rul_df


# ── Validation & cleaning ────────────────────────────────────────────


def validate_dataframe(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    impute: bool = True,
    drop_flat_sensors: bool = True,
) -> pd.DataFrame:
    """Run a battery of sanity checks and light cleaning.

    1. Verify required columns exist.
    2. Sort by (unit_id, cycle) so temporal ordering is guaranteed.
    3. Detect and drop exact-duplicate rows.
    4. Forward-fill then backward-fill missing values within each engine.
    5. Optionally drop sensors whose std < threshold across the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : PipelineConfig
    impute : bool
        If True, fill missing values with forward-fill + back-fill per unit.
    drop_flat_sensors : bool
        If True, remove sensor columns with near-zero variance.

    Returns
    -------
    pd.DataFrame  (copy — original is never mutated)
    """
    df = df.copy()

    required = [cfg.unit_col, cfg.cycle_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df.sort_values([cfg.unit_col, cfg.cycle_col], inplace=True)
    df.drop_duplicates(inplace=True)

    if impute and df.isnull().any().any():
        df = df.groupby(cfg.unit_col, group_keys=False).apply(
            lambda g: g.ffill().bfill()
        )

    if drop_flat_sensors:
        present_sensors = [c for c in cfg.sensor_cols if c in df.columns]
        stds = df[present_sensors].std()
        flat = stds[stds < cfg.flat_sensor_std_threshold].index.tolist()
        if flat:
            df.drop(columns=flat, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def get_active_sensor_cols(df: pd.DataFrame, cfg: PipelineConfig) -> List[str]:
    """Return the sensor columns actually present after cleaning."""
    return [c for c in cfg.sensor_cols if c in df.columns]


def get_active_feature_cols(df: pd.DataFrame, cfg: PipelineConfig) -> List[str]:
    """Return op-setting + remaining sensor columns."""
    ops = [c for c in cfg.op_setting_cols if c in df.columns]
    sensors = get_active_sensor_cols(df, cfg)
    return ops + sensors
