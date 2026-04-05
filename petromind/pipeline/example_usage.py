#!/usr/bin/env python3
"""
End-to-end example of the PetroMind predictive-maintenance pipeline.

This script demonstrates the full flow on **synthetic** C-MAPSS-like data so
it can run without any external data files.  Replace the synthetic generator
with ``load_cmapss_train`` / ``load_cmapss_test`` for real data.

Usage::

    python -m petromind.pipeline.example_usage
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from petromind.pipeline import (
    PipelineConfig,
    build_dataloaders,
    build_sliding_windows,
    compute_classification_label,
    compute_rul,
    validate_dataframe,
    FeatureExtractor,
)
from petromind.pipeline.utils import get_active_feature_cols


# =====================================================================
# 1.  Synthetic data that mimics NASA C-MAPSS structure
# =====================================================================

def make_synthetic_cmapss(
    n_engines: int = 20,
    min_life: int = 50,
    max_life: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a toy run-to-failure dataset.

    Each engine starts with healthy sensor readings and gradually degrades
    toward failure (modelled as a linearly increasing trend plus noise).
    """
    rng = np.random.RandomState(seed)
    frames = []
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    for uid in range(1, n_engines + 1):
        life = rng.randint(min_life, max_life + 1)
        cycles = np.arange(1, life + 1)
        n = len(cycles)
        degradation = np.linspace(0, 1, n)[:, None]  # (n, 1)

        # Op settings — roughly constant per engine with small noise
        ops = rng.uniform(0, 1, size=(1, 3)) + rng.normal(0, 0.01, (n, 3))

        # Sensors: baseline + degradation trend + noise
        baseline = rng.uniform(0.2, 0.8, size=(1, 21))
        sensors = baseline + degradation * rng.uniform(0.1, 0.5, size=(1, 21)) \
                  + rng.normal(0, 0.02, (n, 21))

        # A few "flat" sensors to exercise the flat-sensor removal logic
        sensors[:, 0] = 0.5  # s1 — constant
        sensors[:, 4] = 0.5  # s5 — constant

        row = np.column_stack([
            np.full(n, uid),
            cycles,
            ops,
            sensors,
        ])
        cols = ["unit_id", "cycle", "op_set_1", "op_set_2", "op_set_3"] + sensor_cols
        frames.append(pd.DataFrame(row, columns=cols))

    df = pd.concat(frames, ignore_index=True)
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def main() -> None:
    cfg = PipelineConfig(
        window_size=30,
        stride=1,
        prediction_horizon=30,
        rul_clip=125,
        val_ratio=0.2,
        batch_size=64,
    )

    # ── Step 1: Load / generate data ──────────────────────────────────
    print("=" * 60)
    print("Step 1 — Generate synthetic C-MAPSS-like data")
    print("=" * 60)
    raw_df = make_synthetic_cmapss(n_engines=20, seed=42)
    print(f"  Raw shape : {raw_df.shape}")
    print(f"  Engines   : {raw_df['unit_id'].nunique()}")
    print(f"  Columns   : {list(raw_df.columns[:8])} ... ({len(raw_df.columns)} total)")

    # ── Step 2: Validate & clean ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Validate & clean (impute, drop flat sensors)")
    print("=" * 60)
    clean_df = validate_dataframe(raw_df, cfg)
    removed = set(cfg.sensor_cols) - set(clean_df.columns)
    print(f"  Cleaned shape : {clean_df.shape}")
    print(f"  Removed flat  : {sorted(removed) if removed else 'none'}")

    # ── Step 3: Compute labels ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 — Compute RUL + classification labels")
    print("=" * 60)
    labeled_df = compute_rul(clean_df, cfg)
    labeled_df = compute_classification_label(labeled_df, cfg)
    print(f"  RUL range     : [{labeled_df['rul'].min()}, {labeled_df['rul'].max()}]")
    print(f"  Label balance : {labeled_df['label'].value_counts().to_dict()}")

    # ── Step 4: Build sliding windows ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4 — Build sliding windows (no future leakage)")
    print("=" * 60)
    feature_cols = get_active_feature_cols(labeled_df, cfg)
    X_raw, y_cls, y_rul, engine_ids = build_sliding_windows(
        labeled_df, cfg, feature_cols=feature_cols,
    )
    print(f"  X_raw shape   : {X_raw.shape}  (samples, window, features)")
    print(f"  y_cls shape   : {y_cls.shape}")
    print(f"  y_rul shape   : {y_rul.shape}")
    print(f"  Engines repr. : {len(np.unique(engine_ids))}")

    # ── Step 5: Feature engineering ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5 — Extract engineered features per window")
    print("=" * 60)
    extractor = FeatureExtractor(cfg, n_pca_components=3)
    X_eng = extractor.transform(X_raw)
    names = extractor.feature_names(feature_cols)
    print(f"  Engineered shape : {X_eng.shape}  (samples, features)")
    print(f"  Feature names    : {names[:6]} ... ({len(names)} total)")

    # ── Step 6: Build PyTorch DataLoaders ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6 — Time-based split → PyTorch DataLoaders")
    print("=" * 60)

    # Option A: feed raw windows (for LSTM / 1D-CNN)
    train_raw, val_raw, ds_raw = build_dataloaders(
        X_raw, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  [Raw windows]  train batches={len(train_raw)}, val batches={len(val_raw)}")

    # Option B: feed engineered features (for XGBoost / dense net)
    train_eng, val_eng, ds_eng = build_dataloaders(
        X_eng, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  [Engineered]   train batches={len(train_eng)}, val batches={len(val_eng)}")

    # ── Step 7: Peek at a batch ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7 — Sample batch from raw-window DataLoader")
    print("=" * 60)
    batch = next(iter(train_raw))
    print(f"  features : {batch['features'].shape}  dtype={batch['features'].dtype}")
    print(f"  label    : {batch['label'].shape}    dtype={batch['label'].dtype}")
    print(f"  rul      : {batch['rul'].shape}    dtype={batch['rul'].dtype}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline summary")
    print("=" * 60)
    print(f"  Window size       : {cfg.window_size}")
    print(f"  Stride            : {cfg.stride}")
    print(f"  Prediction horizon: {cfg.prediction_horizon}")
    print(f"  RUL clip          : {cfg.rul_clip}")
    print(f"  Val ratio         : {cfg.val_ratio}")
    print(f"  Total samples     : {len(ds_raw)}")
    print(f"  Raw feature dim   : {X_raw.shape[1:]}")
    print(f"  Eng. feature dim  : {X_eng.shape[1:]}")
    print()
    print("How leakage is avoided:")
    print("  • Windows use ONLY past data (cycles [t-W+1 … t]).")
    print("  • Labels come from the LAST timestep in each window.")
    print("  • Train/val split is by ENGINE, not by random sample.")
    print("  • No engine appears in both train and val sets.")
    print()
    print("How RUL is computed:")
    print("  • RUL_t = max_cycle(engine) − cycle_t")
    print(f"  • Capped at {cfg.rul_clip} (piece-wise linear) so the model")
    print("    focuses on degradation, not large uninformative values.")
    print()
    print("How features improve model performance:")
    print("  • Statistical features (mean, std, …) summarise window distribution.")
    print("  • Signal features (RMS, FFT) capture vibration / frequency patterns.")
    print("  • Health indicators (trend slope, tail ratio) detect degradation onset.")
    print("  • Sensor fusion (PCA eigenvalues) finds correlated multi-sensor decay.")
    print("  • Together they compress a (W, F) window into a rich fixed-size vector")
    print("    suitable for both tree-based and neural models.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
