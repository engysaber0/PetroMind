"""
Centralised, immutable configuration for every stage of the pipeline.

All magic numbers live here so that experiments are reproducible and
hyper-parameter sweeps only need to touch one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PipelineConfig:
    # ── Column schema (NASA C-MAPSS default) ─────────────────────────
    unit_col: str = "unit_id"
    cycle_col: str = "cycle"
    op_setting_cols: List[str] = field(
        default_factory=lambda: ["op_set_1", "op_set_2", "op_set_3"]
    )
    sensor_cols: List[str] = field(
        default_factory=lambda: [f"s{i}" for i in range(1, 22)]
    )

    # ── Windowing ─────────────────────────────────────────────────────
    window_size: int = 30
    stride: int = 1

    # ── Labeling ──────────────────────────────────────────────────────
    prediction_horizon: int = 30       # classify as 1 if RUL <= this
    rul_clip: Optional[int] = 125      # piece-wise linear RUL cap (None = no cap)

    # ── Feature engineering ───────────────────────────────────────────
    fft_top_k: int = 5                 # top-k FFT magnitudes to keep
    rolling_health_window: int = 10    # rolling window for health-indicator trend

    # ── Train / validation split ──────────────────────────────────────
    val_ratio: float = 0.2             # fraction of *engines* used for validation

    # ── DataLoader ────────────────────────────────────────────────────
    batch_size: int = 256
    num_workers: int = 0

    # ── Flat-sensor removal ───────────────────────────────────────────
    flat_sensor_std_threshold: float = 0.01

    # ── Training ──────────────────────────────────────────────────────
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    n_lstm_layers: int = 2
    dropout: float = 0.3
    early_stop_patience: int = 8
    model_dir: str = "checkpoints"

    @property
    def feature_cols(self) -> List[str]:
        """Columns treated as raw input features (op-settings + sensors)."""
        return self.op_setting_cols + self.sensor_cols
