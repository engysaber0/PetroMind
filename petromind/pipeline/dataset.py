"""
PyTorch Dataset and DataLoader construction with time-based splits.

Split strategy (no leakage)
----------------------------
We split **by engine** (unit_id) rather than by individual windows.  Engines
are sorted by their maximum observed cycle, and the earliest ``1 − val_ratio``
fraction becomes the training set while the latest ``val_ratio`` fraction
becomes the validation set.

This mirrors a real deployment scenario where the model is trained on
historically completed run-to-failure records and evaluated on more recent
ones.  No windows from the same engine appear in both sets.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import PipelineConfig


class SensorNormalizer:
    """Per-sensor z-score normalization with train/val separation.

    Computes mean/std from training data only, applies to both train and val.
    Operates on windowed data of shape (N, W, F) or engineered features (N, F_eng).
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "SensorNormalizer":
        """Compute mean/std from training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, W, F) or (N, F_eng)
        """
        if X_train.ndim == 3:
            # Windowed data: compute per-feature stats across all timesteps
            self.mean = X_train.mean(axis=(0, 1))  # (F,)
            self.std = X_train.std(axis=(0, 1)) + 1e-8  # (F,)
        else:
            # Engineered features: (N, F_eng)
            self.mean = X_train.mean(axis=0)  # (F_eng,)
            self.std = X_train.std(axis=0) + 1e-8  # (F_eng,)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization.

        Parameters
        ----------
        X : np.ndarray, shape (N, ...) matching fit shape

        Returns
        -------
        np.ndarray : Normalized data
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before transform()")
        return (X - self.mean) / self.std

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and transform it."""
        self.fit(X_train)
        return self.transform(X_train)


class PredMaintenanceDataset(Dataset):
    """PyTorch Dataset wrapping windowed maintenance data.

    Stores three tensors:
        - ``X``     : (N, W, F)  or  (N, F_eng)  float32  — features
        - ``y_cls`` : (N,)  int64   — binary classification label
        - ``y_rul`` : (N,)  float32 — RUL regression target

    Plus a plain ndarray ``engine_ids`` for grouped splitting / eval.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        y_rul: np.ndarray,
        engine_ids: np.ndarray,
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_rul = torch.as_tensor(y_rul, dtype=torch.float32)
        self.engine_ids = engine_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.X[idx],
            "label": self.y_cls[idx],
            "rul": self.y_rul[idx],
        }


def time_based_split(
    engine_ids: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic time-based split by engine id.

    Engines with *lower* ids are treated as historically earlier data
    (matching C-MAPSS convention where unit_id order correlates with
    recording order).

    Parameters
    ----------
    engine_ids : np.ndarray, shape (N,)
        Per-window engine identifiers.
    cfg : PipelineConfig

    Returns
    -------
    train_idx, val_idx : np.ndarray[int]
        Indices into the dataset.
    """
    unique_engines = np.sort(np.unique(engine_ids))
    n_val = max(1, int(len(unique_engines) * cfg.val_ratio))
    val_engines = set(unique_engines[-n_val:])
    train_mask = np.array([eid not in val_engines for eid in engine_ids])
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(~train_mask)[0]
    return train_idx, val_idx


def build_dataloaders(
    X: np.ndarray,
    y_cls: np.ndarray,
    y_rul: np.ndarray,
    engine_ids: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[DataLoader, DataLoader, PredMaintenanceDataset]:
    """One-call convenience: build dataset → split → DataLoaders.

    Parameters
    ----------
    X : (N, ...) features (raw windows or engineered)
    y_cls, y_rul : labels
    engine_ids : per-sample engine id
    cfg : PipelineConfig

    Returns
    -------
    train_loader, val_loader, full_dataset
    """
    ds = PredMaintenanceDataset(X, y_cls, y_rul, engine_ids)
    train_idx, val_idx = time_based_split(engine_ids, cfg)

    train_loader = DataLoader(
        Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, ds
