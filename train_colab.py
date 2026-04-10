#!/usr/bin/env python3
"""
Google Colab training script for PetroMind RUL pipeline.

Usage in Colab:
    1. Upload All_train_data.xlsx to Colab (or mount Google Drive)
    2. Run: !python train_colab.py --excel /content/All_train_data.xlsx

This script trains on ALL 4 C-MAPSS training sheets combined:
    - train_FD001(HPC Degradation)       — 100 engines
    - train_FD002(HPC Degradation)       — 260 engines
    - train_FD003(HPC+Fan_Deg)           — 100 engines
    - train_FD004(HPC+Fan_Deg)           — 249 engines
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from petromind.pipeline import (
    PipelineConfig,
    build_dataloaders,
    build_sliding_windows,
    compute_classification_label,
    compute_rul,
    validate_dataframe,
    FeatureExtractor,
    LSTMRULModel,
    Trainer,
    load_cmapss_excel_all_sheets,
)
from petromind.pipeline.utils import get_active_feature_cols


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train RUL model on C-MAPSS Excel data.")
    p.add_argument("--excel", type=str, default="/content/All_train_data.xlsx",
                   help="Path to multi-sheet Excel file.")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--prediction-horizon", type=int, default=30)
    p.add_argument("--rul-clip", type=int, default=125)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-lstm-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--model-dir", type=str, default="checkpoints")
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
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        n_lstm_layers=args.n_lstm_layers,
        dropout=args.dropout,
        early_stop_patience=args.early_stop_patience,
        model_dir=args.model_dir,
    )

    # Step 1: Load all sheets
    print("=" * 60)
    print(f"Step 1 - Load all sheets from {args.excel}")
    print("=" * 60)
    raw_df = load_cmapss_excel_all_sheets(args.excel)
    print(f"  Shape   : {raw_df.shape}")
    print(f"  Engines : {raw_df['unit_id'].nunique()}")

    # Step 2: Validate & clean
    print(f"\n{'=' * 60}")
    print("Step 2 - Validate & clean")
    print("=" * 60)
    clean_df = validate_dataframe(raw_df, cfg)
    removed = set(cfg.sensor_cols) - set(clean_df.columns)
    print(f"  Cleaned shape : {clean_df.shape}")
    print(f"  Removed flat  : {sorted(removed) if removed else 'none'}")

    # Step 3: Labels
    print(f"\n{'=' * 60}")
    print("Step 3 - Compute RUL + classification labels")
    print("=" * 60)
    labeled_df = compute_rul(clean_df, cfg)
    labeled_df = compute_classification_label(labeled_df, cfg)
    print(f"  RUL range     : [{labeled_df['rul'].min()}, {labeled_df['rul'].max()}]")
    print(f"  Label balance : {labeled_df['label'].value_counts().to_dict()}")

    # Step 4: Windows
    print(f"\n{'=' * 60}")
    print("Step 4 - Build sliding windows")
    print("=" * 60)
    feature_cols = get_active_feature_cols(labeled_df, cfg)
    X_raw, y_cls, y_rul, engine_ids = build_sliding_windows(
        labeled_df, cfg, feature_cols=feature_cols,
    )
    print(f"  X shape       : {X_raw.shape}")
    print(f"  Engines used  : {len(np.unique(engine_ids))}")

    if X_raw.shape[0] == 0:
        print("\n  No windows produced. Try a smaller --window-size.")
        sys.exit(1)

    # Step 5: Features
    print(f"\n{'=' * 60}")
    print("Step 5 - Extract engineered features")
    print("=" * 60)
    extractor = FeatureExtractor(cfg, n_pca_components=3)
    X_eng = extractor.transform(X_raw)
    print(f"  Engineered shape : {X_eng.shape}")

    # Step 6: DataLoaders
    print(f"\n{'=' * 60}")
    print("Step 6 - Time-based split -> DataLoaders")
    print("=" * 60)
    train_loader, val_loader, ds = build_dataloaders(
        X_raw, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # Step 7: Train
    print(f"\n{'=' * 60}")
    print("Step 7 - Train LSTM RUL regression model")
    print("=" * 60)
    input_dim = X_raw.shape[2]
    model = LSTMRULModel(input_dim=input_dim, cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Model       : LSTMRULModel")
    print(f"  Input dim   : {input_dim}")
    print(f"  Hidden dim  : {cfg.hidden_dim}")
    print(f"  LSTM layers : {cfg.n_lstm_layers}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Device      : {device}")
    print()

    trainer = Trainer(model=model, cfg=cfg)
    history = trainer.fit(train_loader, val_loader)

    # Step 8: Final eval
    print(f"\n{'=' * 60}")
    print("Step 8 - Final evaluation")
    print("=" * 60)
    val_loss, val_metrics = trainer.evaluate(val_loader)
    print(f"  Val loss    : {val_loss:.4f}")
    print(f"  RMSE (RUL)  : {val_metrics['rmse']:.1f}")
    print(f"  MAE  (RUL)  : {val_metrics['mae']:.1f}")
    print(f"\n  Best model saved to: {cfg.model_dir}/best_model.pt")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
