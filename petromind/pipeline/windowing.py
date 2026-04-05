"""
Sliding-window construction over per-engine time-series.

Design decisions that prevent data leakage
-------------------------------------------
1. Windows are built **per engine**: data from different engines is never
   mixed inside a single window.
2. Each window is composed of **exactly ``window_size`` consecutive past
   cycles** ending at cycle *t*.  No future cycles are included.
3. The label for a window is taken from the **last timestep** in the window
   (cycle *t*), so the prediction target is always "what happens *after*
   the data the model has seen".
4. Engines with fewer cycles than ``window_size`` are silently skipped —
   no zero-padding is done, which would introduce synthetic signal.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import get_active_feature_cols


def build_sliding_windows(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a labelled DataFrame into sliding-window arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``cfg.unit_col``, ``cfg.cycle_col``, ``rul``, ``label``,
        and every column listed in *feature_cols*.
    cfg : PipelineConfig
    feature_cols : list[str] or None
        Columns to extract as features.  If None, all active op-setting +
        sensor columns are used.

    Returns
    -------
    X : np.ndarray, shape (N, window_size, n_features)
        Feature windows.  Axis 1 is time-ordered (oldest → newest).
    y_cls : np.ndarray, shape (N,)
        Binary classification labels (1 = failure within horizon).
    y_rul : np.ndarray, shape (N,)
        RUL values aligned with the *last* timestep of each window.
    engine_ids : np.ndarray, shape (N,)
        Engine (unit) id for each window — useful for grouped evaluation.
    """
    if feature_cols is None:
        feature_cols = get_active_feature_cols(df, cfg)

    for required in ("rul", "label"):
        if required not in df.columns:
            raise KeyError(
                f"Column '{required}' missing. Run labeling functions first."
            )

    windows: List[np.ndarray] = []
    labels_cls: List[int] = []
    labels_rul: List[float] = []
    ids: List[int] = []

    grouped = df.groupby(cfg.unit_col, sort=True)

    for uid, grp in grouped:
        grp = grp.sort_values(cfg.cycle_col)
        feat_matrix = grp[feature_cols].values      # (T, F)
        rul_arr = grp["rul"].values                  # (T,)
        cls_arr = grp["label"].values                # (T,)
        n_steps = len(grp)

        if n_steps < cfg.window_size:
            continue

        # Slide with stride; each window uses indices [i : i+window_size].
        # The label comes from the *last* timestep in the window.
        for start in range(0, n_steps - cfg.window_size + 1, cfg.stride):
            end = start + cfg.window_size
            windows.append(feat_matrix[start:end])
            labels_cls.append(cls_arr[end - 1])
            labels_rul.append(rul_arr[end - 1])
            ids.append(uid)

    if len(windows) == 0:
        n_feat = len(feature_cols)
        return (
            np.empty((0, cfg.window_size, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.stack(windows, axis=0).astype(np.float32)           # (N, W, F)
    y_cls = np.array(labels_cls, dtype=np.int64)                # (N,)
    y_rul = np.array(labels_rul, dtype=np.float32)              # (N,)
    engine_ids = np.array(ids)                                  # (N,)
    return X, y_cls, y_rul, engine_ids
