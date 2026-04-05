# PetroMind — Predictive Maintenance ML Pipeline

End-to-end pipeline for industrial predictive maintenance: data preparation, feature engineering, and model training for Remaining Useful Life (RUL) prediction and failure classification on sensor time-series data (NASA C-MAPSS format).

---

## Quick Start (Local Setup)

### Prerequisites

- Python 3.9 or later
- pip

### 1. Clone the repo

```bash
git clone https://github.com/karimzaki8/Depi-project-.git
cd Depi-project-
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install the package

```bash
pip install -e ".[dev]"
```

This installs the `petromind` package in editable mode along with all dependencies (numpy, pandas, scipy, torch, scikit-learn, openpyxl) and dev tools (pytest).

### 4. Run the full pipeline (data prep + training)

**With synthetic data (no files needed):**

```bash
python run_pipeline.py
```

This will:
1. Generate synthetic C-MAPSS-like sensor data (20 engines by default)
2. Validate, clean, and label the data
3. Build sliding windows
4. Extract engineered features
5. Train a dual-head LSTM (classification + RUL regression)
6. Evaluate on validation set and save the best model

**With real C-MAPSS data:**

```bash
python run_pipeline.py --data data/csv/train_1.csv
```

**Data prep only (skip training):**

```bash
python run_pipeline.py --no-train
```

**Custom parameters:**

```bash
python run_pipeline.py \
  --window-size 50 \
  --stride 5 \
  --prediction-horizon 20 \
  --epochs 100 \
  --lr 0.001 \
  --hidden-dim 128 \
  --batch-size 128
```

Run `python run_pipeline.py --help` for all available options.

### 5. Run the tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
.
├── run_pipeline.py                  # Entry point — data prep + training
├── setup.py                         # Package installer
├── requirements.txt                 # Pinned dependencies
├── checkpoints/                     # Saved model weights (created on training)
│   └── best_model.pt
├── petromind/
│   ├── __init__.py
│   └── pipeline/
│       ├── __init__.py
│       ├── config.py                # PipelineConfig — all hyperparameters
│       ├── utils.py                 # Data loading, validation, cleaning
│       ├── labeling.py              # RUL + classification label computation
│       ├── windowing.py             # Sliding window builder
│       ├── features.py              # Feature extraction (stat, FFT, health, PCA)
│       ├── dataset.py               # PyTorch Dataset & DataLoader + split
│       ├── models.py                # DualHeadLSTM (classification + RUL heads)
│       ├── trainer.py               # Training loop, metrics, early stopping
│       └── example_usage.py         # Standalone demo (data prep only)
├── tests/
│   ├── test_pipeline.py             # 15 data pipeline tests
│   └── test_training.py             # 5 model + training tests
├── data_cleaning.ipynb              # Original data-cleaning notebook
└── PetroMind_System_Analysis.md     # System requirements document
```

---

## Pipeline Steps

| Step | What happens |
|------|-------------|
| **1. Load data** | Read C-MAPSS txt/csv/xlsx or generate synthetic data |
| **2. Validate & clean** | Sort by engine+cycle, impute missing values, remove flat sensors |
| **3. Label** | Compute RUL per timestep (`max_cycle - cycle`, capped at 125) and binary classification label (`1` if RUL <= horizon) |
| **4. Window** | Build sliding windows of configurable size/stride per engine |
| **5. Engineer features** | Extract statistical, signal (RMS, FFT), health indicator, and sensor fusion (PCA) features per window |
| **6. Split & load** | Time-based train/val split by engine, wrap in PyTorch DataLoaders |
| **7. Train** | Dual-head LSTM with early stopping, gradient clipping, LR scheduling |
| **8. Evaluate** | Classification (Accuracy, F1, AUC) + Regression (RMSE, MAE) metrics |

---

## Model Architecture

**DualHeadLSTM** — a shared LSTM encoder with two task-specific heads:

```
Input (B, W, F)
    │
    ▼
┌──────────┐
│   LSTM   │  (multi-layer, with dropout)
│ Encoder  │
└────┬─────┘
     │ last hidden state (B, H)
     │
     ├──────────────────┐
     ▼                  ▼
┌──────────┐     ┌──────────┐
│   Cls    │     │   RUL    │
│   Head   │     │   Head   │
│ (Binary) │     │  (Regr)  │
└────┬─────┘     └────┬─────┘
     │                │
     ▼                ▼
  P(failure)     Estimated RUL
```

- **Classification head**: predicts P(failure within next N cycles) — BCE loss
- **RUL head**: predicts remaining useful life in cycles — MSE loss
- Both heads are trained jointly; the combined loss is `cls_weight * L_cls + rul_weight * L_rul`

### Training features
- Adam optimizer with weight decay
- ReduceLROnPlateau learning rate scheduler
- Gradient clipping (max norm = 1.0)
- Early stopping on validation loss
- Best-model checkpointing

---

## How Leakage Is Avoided

- Each window contains **only past cycles** `[t-W+1, ..., t]`
- Labels come from the **last timestep** of each window
- Train/val split is **by engine** — no engine appears in both sets
- Short engines (fewer cycles than window size) are skipped, not zero-padded

## How RUL Is Computed

```
RUL_t = max_cycle(engine) - cycle_t
```

Capped at `rul_clip` (default 125) so the model focuses on the degradation phase rather than predicting large uninformative numbers for healthy engines.

## How Features Improve Model Performance

| Feature group | What it captures |
|---------------|-----------------|
| **Statistical** (mean, std, min, max, skew, kurtosis) | Distribution shift as components degrade |
| **Signal** (RMS, top-k FFT magnitudes) | Vibration and frequency degradation patterns |
| **Health indicators** (trend slope, tail-mean ratio) | How fast degradation is accelerating |
| **Sensor fusion** (PCA eigenvalues) | Correlated multi-sensor responses to failure |

---

## Configuration

All parameters are centralized in `PipelineConfig` (`petromind/pipeline/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 30 | Number of timesteps per window |
| `stride` | 1 | Step between consecutive windows |
| `prediction_horizon` | 30 | Label = 1 if RUL <= this |
| `rul_clip` | 125 | Cap RUL at this value (None = no cap) |
| `val_ratio` | 0.2 | Fraction of engines for validation |
| `batch_size` | 256 | DataLoader batch size |
| `fft_top_k` | 5 | Number of top FFT magnitudes to keep |
| `flat_sensor_std_threshold` | 0.01 | Drop sensors with std below this |
| `epochs` | 50 | Max training epochs |
| `learning_rate` | 0.001 | Adam learning rate |
| `hidden_dim` | 64 | LSTM hidden dimension |
| `n_lstm_layers` | 2 | Number of LSTM layers |
| `dropout` | 0.3 | Dropout rate |
| `early_stop_patience` | 8 | Epochs without improvement before stopping |

---

## Using with Your Own Data

```python
from petromind.pipeline import (
    PipelineConfig, load_cmapss_train, validate_dataframe,
    compute_rul, compute_classification_label,
    build_sliding_windows, FeatureExtractor, build_dataloaders,
    DualHeadLSTM, Trainer,
)
from petromind.pipeline.utils import get_active_feature_cols

cfg = PipelineConfig(window_size=50, stride=1, prediction_horizon=30, epochs=100)

# Data prep
df = load_cmapss_train("data/csv/train_1.csv")
df = validate_dataframe(df, cfg)
df = compute_rul(df, cfg)
df = compute_classification_label(df, cfg)

feature_cols = get_active_feature_cols(df, cfg)
X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols)
train_loader, val_loader, dataset = build_dataloaders(X, y_cls, y_rul, engine_ids, cfg)

# Train
model = DualHeadLSTM(input_dim=X.shape[2], cfg=cfg)
trainer = Trainer(model=model, cfg=cfg)
history = trainer.fit(train_loader, val_loader)

# Evaluate
val_loss, metrics = trainer.evaluate(val_loader)
print(f"F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  RMSE={metrics['rmse']:.1f}")
```

## Loading a Saved Model

```python
import torch
from petromind.pipeline import PipelineConfig, DualHeadLSTM

cfg = PipelineConfig()
model = DualHeadLSTM(input_dim=22, cfg=cfg)
model.load_state_dict(torch.load("checkpoints/best_model.pt", weights_only=True))
model.eval()
```
