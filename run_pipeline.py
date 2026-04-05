#!/usr/bin/env python3
"""
Convenience entry-point to run the full PetroMind pipeline demo.

Usage:
    python run_pipeline.py                           # synthetic data (no files needed)
    python run_pipeline.py --data data/csv/train_1.csv   # real C-MAPSS CSV

All parameters are configurable via command-line flags.
Run ``python run_pipeline.py --help`` for the full list.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from petromind.pipeline.utils import get_active_feature_cols, load_cmapss_train


def make_synthetic_cmapss(
    n_engines: int = 20,
    min_life: int = 50,
    max_life: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a toy run-to-failure dataset (no external files needed)."""
    rng = np.random.RandomState(seed)
    frames = []
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    for uid in range(1, n_engines + 1):
        life = rng.randint(min_life, max_life + 1)
        cycles = np.arange(1, life + 1)
        n = len(cycles)
        degradation = np.linspace(0, 1, n)[:, None]
        ops = rng.uniform(0, 1, size=(1, 3)) + rng.normal(0, 0.01, (n, 3))
        baseline = rng.uniform(0.2, 0.8, size=(1, 21))
        sensors = baseline + degradation * rng.uniform(0.1, 0.5, size=(1, 21)) \
                  + rng.normal(0, 0.02, (n, 21))
        sensors[:, 0] = 0.5   # s1 flat
        sensors[:, 4] = 0.5   # s5 flat
        row = np.column_stack([np.full(n, uid), cycles, ops, sensors])
        cols = ["unit_id", "cycle", "op_set_1", "op_set_2", "op_set_3"] + sensor_cols
        frames.append(pd.DataFrame(row, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the PetroMind windowing + feature-engineering pipeline."
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Path to a C-MAPSS training file (txt/csv/xlsx). "
             "If omitted, synthetic data is generated.",
    )
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--prediction-horizon", type=int, default=30)
    p.add_argument("--rul-clip", type=int, default=125)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-engines", type=int, default=20,
                   help="Number of synthetic engines (only when --data is omitted).")
    p.add_argument("--pca-components", type=int, default=3)
    p.add_argument("--fft-top-k", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cfg = PipelineConfig(
        window_size=args.window_size,
        stride=args.stride,
        prediction_horizon=args.prediction_horizon,
        rul_clip=args.rul_clip,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        fft_top_k=args.fft_top_k,
    )

    # Step 1: Load data
    print("=" * 60)
    if args.data:
        print(f"Step 1 - Load C-MAPSS data from {args.data}")
        raw_df = load_cmapss_train(args.data)
    else:
        print(f"Step 1 - Generate synthetic data ({args.n_engines} engines)")
        raw_df = make_synthetic_cmapss(n_engines=args.n_engines)
    print("=" * 60)
    print(f"  Shape   : {raw_df.shape}")
    print(f"  Engines : {raw_df['unit_id'].nunique()}")
    print(f"  Columns : {list(raw_df.columns[:8])} ... ({len(raw_df.columns)} total)")

    # Step 2: Validate & clean
    print(f"\n{'=' * 60}")
    print("Step 2 - Validate & clean (impute, drop flat sensors)")
    print("=" * 60)
    clean_df = validate_dataframe(raw_df, cfg)
    removed = set(cfg.sensor_cols) - set(clean_df.columns)
    print(f"  Cleaned shape : {clean_df.shape}")
    print(f"  Removed flat  : {sorted(removed) if removed else 'none'}")

    # Step 3: Compute labels
    print(f"\n{'=' * 60}")
    print("Step 3 - Compute RUL + classification labels")
    print("=" * 60)
    labeled_df = compute_rul(clean_df, cfg)
    labeled_df = compute_classification_label(labeled_df, cfg)
    print(f"  RUL range     : [{labeled_df['rul'].min()}, {labeled_df['rul'].max()}]")
    print(f"  Label balance : {labeled_df['label'].value_counts().to_dict()}")

    # Step 4: Build sliding windows
    print(f"\n{'=' * 60}")
    print("Step 4 - Build sliding windows (no future leakage)")
    print("=" * 60)
    feature_cols = get_active_feature_cols(labeled_df, cfg)
    X_raw, y_cls, y_rul, engine_ids = build_sliding_windows(
        labeled_df, cfg, feature_cols=feature_cols,
    )
    print(f"  X shape       : {X_raw.shape}  (samples, window, features)")
    print(f"  y_cls shape   : {y_cls.shape}")
    print(f"  y_rul shape   : {y_rul.shape}")
    print(f"  Engines used  : {len(np.unique(engine_ids))}")

    if X_raw.shape[0] == 0:
        print("\n  No windows produced. Try a smaller --window-size.")
        sys.exit(1)

    # Step 5: Feature engineering
    print(f"\n{'=' * 60}")
    print("Step 5 - Extract engineered features per window")
    print("=" * 60)
    extractor = FeatureExtractor(cfg, n_pca_components=args.pca_components)
    X_eng = extractor.transform(X_raw)
    names = extractor.feature_names(feature_cols)
    print(f"  Engineered shape : {X_eng.shape}  (samples, features)")
    print(f"  Feature names    : {names[:5]} ... ({len(names)} total)")

    # Step 6: Build PyTorch DataLoaders
    print(f"\n{'=' * 60}")
    print("Step 6 - Time-based split -> PyTorch DataLoaders")
    print("=" * 60)

    train_raw, val_raw, ds_raw = build_dataloaders(
        X_raw, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  [Raw windows]   train batches={len(train_raw)}, val batches={len(val_raw)}")

    train_eng, val_eng, ds_eng = build_dataloaders(
        X_eng, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  [Engineered]    train batches={len(train_eng)}, val batches={len(val_eng)}")

    # Step 7: Sample batch
    print(f"\n{'=' * 60}")
    print("Step 7 - Sample batch")
    print("=" * 60)
    batch = next(iter(train_raw))
    print(f"  features : {batch['features'].shape}  dtype={batch['features'].dtype}")
    print(f"  label    : {batch['label'].shape}    dtype={batch['label'].dtype}")
    print(f"  rul      : {batch['rul'].shape}    dtype={batch['rul'].dtype}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Pipeline complete")
    print("=" * 60)
    print(f"  Window size        : {cfg.window_size}")
    print(f"  Stride             : {cfg.stride}")
    print(f"  Prediction horizon : {cfg.prediction_horizon}")
    print(f"  RUL clip           : {cfg.rul_clip}")
    print(f"  Val ratio          : {cfg.val_ratio}")
    print(f"  Total samples      : {len(ds_raw)}")
    print(f"  Raw feature dim    : {X_raw.shape[1:]}")
    print(f"  Eng. feature dim   : {X_eng.shape[1:]}")
    print()
    print("Ready for model training.  Pass train_loader to your model.")


if __name__ == "__main__":
    main()
