"""
Training loop for the LSTM RUL regression model.

Handles:
    - RUL regression training (MSE loss)
    - Per-epoch evaluation with RMSE and MAE
    - Early stopping on validation loss
    - Best-model checkpoint saving and loading
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .models import LSTMRULModel


def _compute_metrics(
    y_true_rul: np.ndarray,
    y_pred_rul: np.ndarray,
) -> Dict[str, float]:
    """Compute RUL regression metrics."""
    diff = y_true_rul - y_pred_rul
    return {
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "mae": float(np.abs(diff).mean()),
    }


def _fmt_metrics(m: Dict[str, float]) -> str:
    return f"RMSE={m['rmse']:.1f}  MAE={m['mae']:.1f}"


class Trainer:
    """Trains and evaluates an LSTMRULModel.

    Parameters
    ----------
    model : LSTMRULModel
    cfg : PipelineConfig
    device : str or torch.device
        ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        model: LSTMRULModel,
        cfg: PipelineConfig,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4,
        )

        self.criterion = nn.MSELoss()

        self._best_val_loss = float("inf")
        self._patience_counter = 0

    # ── public API ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """Full training loop with early stopping.

        Returns
        -------
        history : dict
            Keys: ``train_loss``, ``val_loss``, ``val_metrics`` (list of dicts).
        """
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        ckpt_dir = Path(self.cfg.model_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_metrics = self._evaluate(val_loader)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{self.cfg.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={lr:.1e}  {_fmt_metrics(val_metrics)}"
            )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self.save(ckpt_dir / "best_model.pt")
            else:
                self._patience_counter += 1

            if self._patience_counter >= self.cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (patience={self.cfg.early_stop_patience})")
                break

        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            self.load(best_path)
            print(f"  Loaded best model from {best_path}")

        return history

    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Public evaluation interface."""
        return self._evaluate(loader)

    def save(self, path) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path) -> None:
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    # ── internals ─────────────────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            X = batch["features"].to(self.device)
            y_rul = batch["rul"].to(self.device)

            rul_pred = self.model(X)
            loss = self.criterion(rul_pred, y_rul)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_rul_true, all_rul_pred = [], []

        for batch in loader:
            X = batch["features"].to(self.device)
            y_rul = batch["rul"].to(self.device)

            rul_pred = self.model(X)
            loss = self.criterion(rul_pred, y_rul)

            total_loss += loss.item()
            n_batches += 1

            all_rul_true.append(y_rul.cpu().numpy())
            all_rul_pred.append(rul_pred.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        metrics = _compute_metrics(
            np.concatenate(all_rul_true),
            np.concatenate(all_rul_pred),
        )
        return avg_loss, metrics
