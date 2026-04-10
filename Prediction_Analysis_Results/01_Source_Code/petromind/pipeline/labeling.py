"""
Labeling logic for predictive maintenance.

Two label types are computed **per timestep** (i.e. per row in the training
DataFrame), *before* windowing.  This keeps the label semantics clean and
makes it trivial to align labels with the last timestep of each window later.

Label 1 — RUL (Remaining Useful Life)
--------------------------------------
    RUL_t = max_cycle(engine) − cycle_t

    Optionally capped at ``rul_clip`` (piece-wise linear) so the model
    focuses on the degradation phase rather than predicting large
    uninformative numbers for healthy engines.

Label 2 — Binary Classification
---------------------------------
    label_t = 1  if  RUL_t <= prediction_horizon
              0  otherwise

    "Will this engine fail within the next N cycles?"

Both computations are **strictly causal**: they use only the *engine's own
run-to-failure history* that has already been observed (max_cycle is known
because the training set records the complete life of each engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def compute_rul(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Add a ``rul`` column: remaining useful life per row.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``cfg.unit_col`` and ``cfg.cycle_col``.
    cfg : PipelineConfig

    Returns
    -------
    pd.DataFrame  (copy with new ``rul`` column)
    """
    df = df.copy()
    max_cycles = (
        df.groupby(cfg.unit_col)[cfg.cycle_col]
        .max()
        .rename("_max_cycle")
    )
    df = df.merge(max_cycles, on=cfg.unit_col, how="left")
    df["rul"] = df["_max_cycle"] - df[cfg.cycle_col]
    df.drop(columns=["_max_cycle"], inplace=True)

    if cfg.rul_clip is not None:
        df["rul"] = df["rul"].clip(upper=cfg.rul_clip)
    return df


def compute_classification_label(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Add a ``label`` column: binary failure-within-horizon flag.

    Requires the ``rul`` column to already exist (call ``compute_rul`` first).

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain a ``rul`` column.
    cfg : PipelineConfig

    Returns
    -------
    pd.DataFrame  (copy with new ``label`` column)
    """
    if "rul" not in df.columns:
        raise KeyError("Column 'rul' not found — call compute_rul() first.")
    df = df.copy()
    df["label"] = (df["rul"] <= cfg.prediction_horizon).astype(np.int64)
    return df


def compute_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Compute per-row RUL for a C-MAPSS *test* set.

    In the test set each engine is truncated before failure.  The companion
    ``rul_df`` gives the remaining cycles *after the last observed cycle*.

    RUL_t = remaining_rul(engine) + (max_observed_cycle − cycle_t)

    Parameters
    ----------
    test_df : pd.DataFrame
    rul_df : pd.DataFrame
        Must have ``unit_id`` and ``remaining_rul`` columns.
    cfg : PipelineConfig

    Returns
    -------
    pd.DataFrame
    """
    test_df = test_df.copy()
    max_obs = (
        test_df.groupby(cfg.unit_col)[cfg.cycle_col]
        .max()
        .rename("_max_obs_cycle")
    )
    test_df = test_df.merge(max_obs, on=cfg.unit_col, how="left")
    test_df = test_df.merge(
        rul_df[[cfg.unit_col, "remaining_rul"]], on=cfg.unit_col, how="left"
    )
    test_df["rul"] = (
        test_df["remaining_rul"]
        + test_df["_max_obs_cycle"]
        - test_df[cfg.cycle_col]
    )
    test_df.drop(columns=["_max_obs_cycle", "remaining_rul"], inplace=True)

    if cfg.rul_clip is not None:
        test_df["rul"] = test_df["rul"].clip(upper=cfg.rul_clip)
    return test_df
