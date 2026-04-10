# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PetroMind is an end-to-end ML pipeline for industrial predictive maintenance using NASA C-MAPSS sensor time-series data. The pipeline performs LSTM-based RUL (Remaining Useful Life) regression.

## Commands

```bash
# Install dependencies (editable mode)
pip install -e ".[dev]"

# Run full pipeline (generates synthetic data if no data provided)
python run_pipeline.py

# Run with real C-MAPSS data
python run_pipeline.py --data data/csv/train_1.csv

# Data prep only (skip training)
python run_pipeline.py --no-train

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v
pytest tests/test_training.py -v

# Hyperparameter tuning
python -c "
from petromind.pipeline import *
# Grid search example
param_grid = {'hidden_dim': [32, 64], 'learning_rate': [1e-3, 1e-4]}
# See tuner.py for grid_search() and random_search() usage
"
```

## Architecture

The pipeline (`petromind/pipeline/`) follows a sequential data flow:

```
utils.py      → labeling.py    → windowing.py   → features.py   → dataset.py   → models.py → trainer.py
(load/clean)     (RUL + label)    (sliding win)    (stat/FFT/PCA)   (DataLoader)   (LSTM)      (train/eval)
```

**Key modules:**
- `config.py` — Centralized `PipelineConfig` dataclass (all hyperparameters)
- `utils.py` — Data loading (C-MAPSS txt/csv/xlsx), validation, cleaning
- `labeling.py` — RUL computation (`max_cycle - cycle`, capped at 125) and binary classification label
- `windowing.py` — Sliding window builder (avoids leakage via past-only windows)
- `features.py` — Statistical, signal (RMS/FFT), health indicators, sensor fusion (PCA)
- `dataset.py` — PyTorch Dataset + `SensorNormalizer` (per-sensor z-score) + time-based split
- `models.py` — `LSTMRULModel`: LSTM encoder with RUL regression head
- `trainer.py` — Training loop with early stopping, gradient clipping, LR scheduling; prediction export
- `tuner.py` — Grid search and random search for hyperparameter optimization

**Model architecture:**
```
Input (B, W, F) → LSTM Encoder → last hidden state (B, H) → RUL Head → Estimated RUL
```

## Configuration

All parameters in `PipelineConfig` (`petromind/pipeline/config.py`):
- `window_size=30`, `stride=1` — Sliding window dimensions
- `prediction_horizon=30` — Binary label threshold (RUL <= horizon → label=1)
- `rul_clip=125` — RUL cap for focusing on degradation phase
- `fft_top_k=5` — Top FFT magnitudes retained
- `val_ratio=0.2` — Fraction of engines for validation
- `normalize_sensors=True` — Per-sensor z-score normalization (fit on train, apply to both)
- `hidden_dim=64`, `n_lstm_layers=2`, `dropout=0.3` — Model architecture
- `epochs=50`, `lr=1e-3`, `early_stop_patience=8` — Training

## Key Design Decisions

- **Leakage prevention:** Windows contain only past cycles; labels from last timestep; train/val split by engine
- **Short engine handling:** Engines with fewer cycles than window_size are skipped (no zero-padding)
- **Time-based split:** Validation uses entire engines, not random samples
- **Feature groups:** Statistical (mean/std/skew/kurt), signal (RMS/FFT), health indicators (trend slope), sensor fusion (PCA)
- **Normalization:** Per-sensor z-score normalization (fit on train only, apply to both train/val)
- **Evaluation metrics:** RMSE, MAE, and NASA's asymmetric scoring function (penalizes late predictions more)

## Testing

- `tests/test_pipeline.py` — 15 data pipeline tests (loading, labeling, windowing, features)
- `tests/test_training.py` — 10 tests (model, training, metrics, prediction export, normalization)
