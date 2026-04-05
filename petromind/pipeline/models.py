"""
PyTorch model for RUL (Remaining Useful Life) regression.

An LSTM encoder maps a sensor window (B, W, F) to a scalar RUL prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import PipelineConfig


class LSTMRULModel(nn.Module):
    """LSTM encoder with a single RUL regression head.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep (F in the (B, W, F) input).
    cfg : PipelineConfig
        Supplies hidden_dim, n_lstm_layers, dropout.
    """

    def __init__(self, input_dim: int, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)

        self.rul_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, W, F)

        Returns
        -------
        rul_pred : Tensor, shape (B,) — predicted RUL (non-negative via ReLU)
        """
        lstm_out, _ = self.lstm(x)          # (B, W, H)
        h_last = lstm_out[:, -1, :]         # (B, H)
        h_last = self.dropout(h_last)

        rul_pred = self.rul_head(h_last).squeeze(-1)  # (B,)
        rul_pred = torch.relu(rul_pred)                # RUL >= 0

        return rul_pred
