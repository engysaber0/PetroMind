# PetroMind — Predictive Maintenance ML Pipeline

Windowing, labeling, and feature engineering for Remaining Useful Life (RUL) prediction on industrial sensor time-series data (NASA C-MAPSS format).

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

### 4. Run the pipeline

**With synthetic data (no files needed):**

```bash
python run_pipeline.py
```

**With real C-MAPSS data:**

```bash
python run_pipeline.py --data data/csv/train_1.csv
```

**Customize parameters:**

```bash
python run_pipeline.py --window-size 50 --stride 5 --prediction-horizon 20 --rul-clip 125 --batch-size 128
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
├── run_pipeline.py                  # Entry point — run the full pipeline
├── setup.py                         # Package installer
├── requirements.txt                 # Pinned dependencies
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
│       └── example_usage.py         # Standalone demo (alternative to run_pipeline.py)
├── tests/
│   └── test_pipeline.py             # 15 unit tests
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

### How leakage is avoided

- Each window contains **only past cycles** `[t-W+1, ..., t]`
- Labels come from the **last timestep** of each window
- Train/val split is **by engine** — no engine appears in both sets
- Short engines (fewer cycles than window size) are skipped, not zero-padded

### How RUL is computed

```
RUL_t = max_cycle(engine) - cycle_t
```

Capped at `rul_clip` (default 125) so the model focuses on the degradation phase rather than predicting large uninformative numbers for healthy engines.

### How features improve model performance

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

---

## Using with Your Own Data

```python
from petromind.pipeline import (
    PipelineConfig, load_cmapss_train, validate_dataframe,
    compute_rul, compute_classification_label,
    build_sliding_windows, FeatureExtractor, build_dataloaders,
)
from petromind.pipeline.utils import get_active_feature_cols

cfg = PipelineConfig(window_size=50, stride=1, prediction_horizon=30)

df = load_cmapss_train("data/csv/train_1.csv")
df = validate_dataframe(df, cfg)
df = compute_rul(df, cfg)
df = compute_classification_label(df, cfg)

feature_cols = get_active_feature_cols(df, cfg)
X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols)

# Option A: raw windows for LSTM / 1D-CNN
train_loader, val_loader, dataset = build_dataloaders(X, y_cls, y_rul, engine_ids, cfg)

# Option B: engineered features for XGBoost / dense net
ext = FeatureExtractor(cfg)
X_eng = ext.transform(X)
train_loader, val_loader, dataset = build_dataloaders(X_eng, y_cls, y_rul, engine_ids, cfg)
```
