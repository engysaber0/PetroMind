# TODO.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PetroMind is an end-to-end ML pipeline for industrial predictive maintenance using NASA C-MAPSS sensor time-series data. The pipeline performs LSTM-based RUL (Remaining Useful Life) regression.

**Team Structure (6 Engineers):**
| Engineer | Data Task | Implementation Task | Status |
|----------|-----------|---------------------|--------|
| **1** | Load & clean raw CSVs → Parquet | Baseline RUL regression (XGBoost/Linear) | TODO |
| **2** | Windowing + labeling (RUL + classification) | Baseline RUL regression | ✅ Done |
| **3** | Feature engineering (mean, std, RMS, FFT) | LSTM/GRU classifier | TODO |
| **4** | RUL-focused features (health indicators, PCA) | LSTM RUL regression | ✅ Done |
| **5** | Work orders → embeddings (Chroma/Qdrant) | RAG retrieval — work orders | TODO |
| **6** | PDF ingestion, chunking | RAG retrieval — manuals + FastAPI | TODO |

---

## Commands

```bash
# Install dependencies (editable mode)
pip install -e ".[dev]"

# Run full pipeline (Engineer 4's LSTM RUL model)
python run_pipeline.py

# Run with real C-MAPSS data
python run_pipeline.py --data data/csv/train_1.csv

# Run with multi-sheet Excel (all 4 FD subsets)
python run_pipeline.py --excel All_train_data.xlsx --epochs 100

# Data prep only (skip training)
python run_pipeline.py --no-train

# Run tests
pytest tests/ -v
```

---

## File Structure (Proposed Reorganization)

```
petromind/
├── __init__.py
├── pipeline/                    # Core ML pipeline (Eng 2 + Eng 4)
│   ├── __init__.py
│   ├── config.py                # Centralized hyperparameters
│   ├── utils.py                 # Data loading, validation, cleaning
│   ├── labeling.py              # RUL + classification labels (Eng 2)
│   ├── windowing.py             # Sliding window builder (Eng 2)
│   ├── features.py              # Feature extraction (Eng 3 + Eng 4)
│   ├── dataset.py               # PyTorch Dataset + SensorNormalizer
│   ├── models.py                # LSTMRULModel (Eng 4)
│   ├── trainer.py               # Training loop + export (Eng 4)
│   └── tuner.py                 # Hyperparameter search (Eng 4)
│
├── baseline/                    # Engineer 1 - Baseline models [TODO]
│   ├── __init__.py
│   ├── xgboost_rul.py           # XGBoost RUL regressor
│   └── linear_rul.py            # LinearRegression baseline
│
├── classifier/                  # Engineer 3 - Classification models [TODO]
│   ├── __init__.py
│   ├── lstm_classifier.py       # LSTM/GRU binary classifier
│   └── gru_classifier.py        # GRU alternative
│
├── rag/                         # Engineers 5 + 6 - RAG system [TODO]
│   ├── __init__.py
│   ├── work_orders/             # Engineer 5
│   │   ├── embeddings.py
│   │   └── retrieval.py
│   └── manuals/                 # Engineer 6
│       ├── pdf_ingestion.py
│       └── api.py               # FastAPI wiring
│
└── data/                        # Data utilities [Eng 1]
    ├── __init__.py
    ├── loader.py                # Unified data loader (Parquet)
    └── cleaning.py              # Advanced cleaning pipelines

data/                            # Raw and processed data
├── raw/                         # Original C-MAPSS files
├── processed/                   # Cleaned Parquet files
└── external/                    # Work orders, manuals

tests/
├── test_pipeline.py             # Data pipeline tests
├── test_training.py             # Model + training tests
├── test_baseline/               # Engineer 1 tests [TODO]
├── test_classifier/             # Engineer 3 tests [TODO]
└── test_rag/                    # Engineers 5+6 tests [TODO]

checkpoints/                     # Saved model weights
├── best_model.pt                # Engineer 4's LSTM RUL model
├── baseline/                    # Engineer 1 models [TODO]
└── classifier/                  # Engineer 3 models [TODO]

run_pipeline.py                  # Main entry point (Eng 2+4 pipeline)
run_baseline.py                  # Engineer 1 entry point [TODO]
run_classifier.py                # Engineer 3 entry point [TODO]
run_rag.py                       # Engineers 5+6 entry point [TODO]
```

---

## Architecture

The pipeline (`petromind/pipeline/`) follows a sequential data flow:

```
utils.py      → labeling.py    → windowing.py   → features.py   → dataset.py   → models.py → trainer.py
(load/clean)     (RUL + label)    (sliding win)    (stat/FFT/PCA)   (DataLoader)   (LSTM)      (train/eval)
```

**Model architecture:**
```
Input (B, W, F) → LSTM Encoder → last hidden state (B, H) → RUL Head → Estimated RUL
```

---

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

---

## Integration Guide by Engineer

### Engineer 1 — Baseline Models + Data Loading

**Your responsibilities:**
1. Clean Parquet output from raw C-MAPSS data
2. Baseline RUL regression (XGBoost, LinearRegression) for comparison

**How to integrate:**

```python
# 1. Create petromind/baseline/xgboost_rul.py
from xgboost import XGBRegressor
from petromind.pipeline import (
    PipelineConfig, build_sliding_windows,
    compute_classification_label, compute_rul,
    validate_dataframe, time_based_split
)
from petromind.pipeline.utils import get_active_feature_cols

# Use Engineer 2's windowing + labeling
df = load_and_clean_data()  # Your Parquet loader
df = compute_rul(df, cfg)
df = compute_classification_label(df, cfg)
X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols)

# Flatten windows for XGBoost: (N, W, F) → (N, W*F)
X_flat = X.reshape(X.shape[0], -1)

# Train with same train/val split as LSTM
train_idx, val_idx = time_based_split(engine_ids, cfg)
model = XGBRegressor()
model.fit(X_flat[train_idx], y_rul[train_idx])
```

**Key considerations:**
- Use **same windowing/labeling** as Engineer 2 (no leakage)
- Flatten windows for XGBoost: `(N, W, F) → (N, W*F)`
- Compare RMSE/MAE/Score against Engineer 4's LSTM
- Output clean Parquet to `data/processed/` for all engineers to use

**Files to create:**
- `petromind/baseline/__init__.py`
- `petromind/baseline/xgboost_rul.py`
- `petromind/baseline/linear_rul.py`
- `petromind/data/loader.py` — unified Parquet loader
- `run_baseline.py` — entry point

---

### Engineer 3 — LSTM/GRU Classifier

**Your responsibilities:**
1. Binary classification model (failure within horizon vs. not)
2. Can share feature pipeline with Engineer 4

**How to integrate:**

```python
# 1. Create petromind/classifier/lstm_classifier.py
import torch.nn as nn
from petromind.pipeline import (
    PipelineConfig, build_dataloaders, build_sliding_windows,
    compute_classification_label, compute_rul, validate_dataframe,
    FeatureExtractor, SensorNormalizer
)

# Binary classification model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, cfg: PipelineConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_lstm_layers > 1 else 0.0,
        )
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 2),  # 2 classes
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]
        return self.cls_head(h_last)

# Use Engineer 2's labels (y_cls) and same data pipeline
df = load_data()
df = compute_rul(df, cfg)
df = compute_classification_label(df, cfg)  # This gives you y_cls
X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols)

# Optional: use Engineer 4's feature extraction
extractor = FeatureExtractor(cfg, n_pca_components=3)
X_eng = extractor.transform(X)  # (N, 325) instead of (N, W, F)

# Train with BCE loss
criterion = nn.CrossEntropyLoss()
```

**Key considerations:**
- Reuse **Engineer 2's labeling** (`y_cls` = 1 if RUL <= prediction_horizon)
- Can use **raw windows** (N, W, F) or **Engineer 4's features** (N, 325)
- Use **same SensorNormalizer** for fair comparison
- Compare accuracy, precision, recall, F1 against Engineer 4's RUL→classification

**Files to create:**
- `petromind/classifier/__init__.py`
- `petromind/classifier/lstm_classifier.py`
- `petromind/classifier/gru_classifier.py` (optional)
- `run_classifier.py` — entry point

---

### Engineers 5 + 6 — RAG System

**Your responsibilities:**
- Engineer 5: Work orders → embeddings → retrieval
- Engineer 6: PDF manuals → chunking → FastAPI

**How to integrate:**

```python
# 1. Create petromind/rag/work_orders/retrieval.py
from chromadb import Client
from petromind.data.loader import load_work_orders

# Load and embed work orders
work_orders = load_work_orders("data/external/work_orders.csv")
client = Client()
collection = client.create_collection("work_orders")
collection.add(
    documents=work_orders["text"],
    ids=[f"wo_{i}" for i in range(len(work_orders))],
    embeddings=embed(work_orders["text"]),  # Your embedding model
)

# 2. Wire into main pipeline for maintenance recommendations
# When LSTM predicts RUL < 30, query RAG for relevant work orders
if predicted_rul < 30:
    results = collection.query(
        query_embeddings=embed(failure_mode_description),
        n_results=5,
    )
    # Return maintenance actions from similar past cases
```

**Files to create:**
- `petromind/rag/__init__.py`
- `petromind/rag/work_orders/embeddings.py`
- `petromind/rag/work_orders/retrieval.py`
- `petromind/rag/manuals/pdf_ingestion.py`
- `petromind/rag/manuals/api.py`
- `run_rag.py` — FastAPI server entry point

---

## Key Design Decisions

- **Leakage prevention:** Windows contain only past cycles; labels from last timestep; train/val split by engine
- **Short engine handling:** Engines with fewer cycles than window_size are skipped (no zero-padding)
- **Time-based split:** Validation uses entire engines, not random samples
- **Feature groups:** Statistical (mean/std/skew/kurt), signal (RMS/FFT), health indicators (trend slope), sensor fusion (PCA)
- **Normalization:** Per-sensor z-score normalization (fit on train only, apply to both train/val)
- **Evaluation metrics:** RMSE, MAE, and NASA's asymmetric scoring function (penalizes late predictions more)

---

## Testing

- `tests/test_pipeline.py` — 15 data pipeline tests (loading, labeling, windowing, features)
- `tests/test_training.py` — 10 tests (model, training, metrics, prediction export, normalization)

**TODO tests by engineer:**
- Engineer 1: `tests/test_baseline/test_xgboost.py`, `tests/test_baseline/test_linear.py`
- Engineer 3: `tests/test_classifier/test_lstm_cls.py`
- Engineers 5+6: `tests/test_rag/test_retrieval.py`, `tests/test_rag/test_api.py`

---

## Current Model Performance (Engineer 4's LSTM)

Trained on all 4 FD subsets (709 engines):
- **RMSE:** 15.6 cycles
- **MAE:** 10.4 cycles
- **NASA Score:** 483,886
- **Best for:** RUL < 90 (near-failure prediction)

**Known limitations:**
- Struggles with healthy engines (RUL > 90)
- Combined FD datasets add noise (consider per-FD training)
