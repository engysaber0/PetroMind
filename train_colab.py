#!/usr/bin/env python3
"""
Per-subset C-MAPSS training script for PetroMind RUL pipeline.

Trains one model per C-MAPSS subset (FD001–FD004) and prints a
comparison table. Models are saved to checkpoints/FD00{N}/.

Usage in Colab:
    !python train_per_subset.py --excel /content/All_train_data.xlsx
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
)
from petromind.pipeline.utils import get_active_feature_cols


# ── Sheet names as they appear in All_train_data.xlsx ──────────────────────────
SUBSETS = {
    "FD001": "train_FD001(HPC Degradation)",
    "FD002": "train_FD002(HPC Degradation)",
    "FD003": "train_FD003(HPC+Fan_Deg)",
    "FD004": "train_FD004(HPC+Fan_Deg)",
}

# Known uninformative sensors in C-MAPSS (near-zero variance across all regimes)
SENSORS_TO_DROP = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train one RUL model per C-MAPSS subset and compare results."
    )
    p.add_argument("--excel", type=str, default="/content/All_train_data.xlsx")
    p.add_argument("--subsets", nargs="+", default=list(SUBSETS.keys()),
                   choices=list(SUBSETS.keys()),
                   help="Which subsets to train on (default: all four).")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--prediction-horizon", type=int, default=30)
    p.add_argument("--rul-clip", type=int, default=125)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-4,   # slower than combined run
                   help="Initial LR (default 5e-4 — lower than combined-dataset run).")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-lstm-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early-stop-patience", type=int, default=15,
                   help="Patience for early stopping (raised to allow slower convergence).")
    p.add_argument("--model-dir", type=str, default="checkpoints")
    p.add_argument("--no-drop-sensors", action="store_true",
                   help="Disable dropping of known low-variance sensors.")
    return p.parse_args(argv)


def load_single_sheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """Load one sheet and return a raw DataFrame."""
    print(f"  Reading sheet: '{sheet_name}'")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df


def drop_low_variance_sensors(df: pd.DataFrame, drop_set: set[str]) -> pd.DataFrame:
    """Remove columns whose names match the low-variance sensor list."""
    cols_to_drop = [c for c in df.columns if c.lower() in drop_set]
    if cols_to_drop:
        print(f"  Dropping low-variance sensors: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("  No matching low-variance sensors found in columns — skipping drop.")
    return df


def train_subset(
    subset_name: str,
    excel_path: str,
    args: argparse.Namespace,
) -> dict:
    """
    Full pipeline for one subset. Returns a results dict:
        {"subset": str, "engines": int, "windows": int,
         "rmse": float, "mae": float, "val_loss": float}
    """
    sheet_name = SUBSETS[subset_name]
    model_dir = f"{args.model_dir}/{subset_name}"

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
        model_dir=model_dir,
    )

    banner = f"  {subset_name} — {sheet_name}"
    print(f"\n{'=' * 60}")
    print(banner)
    print("=" * 60)

    # 1. Load
    raw_df = load_single_sheet(excel_path, sheet_name)
    n_engines = raw_df["unit_id"].nunique() if "unit_id" in raw_df.columns else "?"
    print(f"  Shape: {raw_df.shape}   |   Engines: {n_engines}")

    # 2. Optionally drop low-variance sensors
    if not args.no_drop_sensors:
        raw_df = drop_low_variance_sensors(raw_df, SENSORS_TO_DROP)

    # 3. Validate & clean
    clean_df = validate_dataframe(raw_df, cfg)
    removed = set(cfg.sensor_cols) - set(clean_df.columns)
    if removed:
        print(f"  Also removed by validate_dataframe: {sorted(removed)}")

    # 4. Labels
    labeled_df = compute_rul(clean_df, cfg)
    labeled_df = compute_classification_label(labeled_df, cfg)
    print(f"  RUL range: [{labeled_df['rul'].min()}, {labeled_df['rul'].max()}]")

    # 5. Sliding windows
    feature_cols = get_active_feature_cols(labeled_df, cfg)
    X_raw, y_cls, y_rul, engine_ids = build_sliding_windows(
        labeled_df, cfg, feature_cols=feature_cols,
    )
    print(f"  Windows: {X_raw.shape[0]}   |   Features per step: {X_raw.shape[2]}")

    if X_raw.shape[0] == 0:
        print("  [SKIP] No windows produced — try a smaller --window-size.")
        return {"subset": subset_name, "engines": n_engines,
                "windows": 0, "rmse": float("nan"), "mae": float("nan"),
                "val_loss": float("nan")}

    # 6. Feature engineering (PCA etc.)
    extractor = FeatureExtractor(cfg, n_pca_components=3)
    extractor.transform(X_raw)  # fit only; raw X used for LSTM

    # 7. DataLoaders
    train_loader, val_loader, _ = build_dataloaders(
        X_raw, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  Train batches: {len(train_loader)}   |   Val batches: {len(val_loader)}")

    # 8. Model
    input_dim = X_raw.shape[2]
    model = LSTMRULModel(input_dim=input_dim, cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Params: {n_params:,}   |   Device: {device}")

    # 9. Train
    trainer = Trainer(model=model, cfg=cfg)
    trainer.fit(train_loader, val_loader)

    # 10. Evaluate
    val_loss, val_metrics = trainer.evaluate(val_loader)
    rmse = val_metrics["rmse"]
    mae = val_metrics["mae"]
    print(f"\n  >> {subset_name}  RMSE={rmse:.1f}  MAE={mae:.1f}  val_loss={val_loss:.4f}")
    print(f"     Model saved to: {model_dir}/best_model.pt")

    return {
        "subset": subset_name,
        "engines": n_engines,
        "windows": X_raw.shape[0],
        "rmse": rmse,
        "mae": mae,
        "val_loss": val_loss,
    }


def print_comparison_table(results: list[dict]) -> None:
    benchmarks = {
        "FD001": (13, 19),
        "FD002": (15, 25),
        "FD003": (12, 20),
        "FD004": (18, 28),
    }

    header = f"{'Subset':<8} {'Engines':>8} {'Windows':>9} {'RMSE':>7} {'MAE':>7} {'Benchmark RMSE':>16} {'Status':>10}"
    print(f"\n{'=' * 70}")
    print("  Per-subset results")
    print("=" * 70)
    print(header)
    print("-" * 70)

    for r in results:
        lo, hi = benchmarks.get(r["subset"], (None, None))
        bench_str = f"{lo}–{hi}" if lo else "n/a"
        rmse = r["rmse"]
        if np.isnan(rmse):
            status = "SKIP"
        elif lo and rmse <= hi:
            status = "on par"
        elif lo and rmse <= hi * 1.3:
            status = "close"
        else:
            status = "needs work"
        print(
            f"  {r['subset']:<6} {r['engines']:>8} {r['windows']:>9} "
            f"{rmse:>7.1f} {r['mae']:>7.1f} {bench_str:>16} {status:>10}"
        )

    valid = [r for r in results if not np.isnan(r["rmse"])]
    if valid:
        avg_rmse = np.mean([r["rmse"] for r in valid])
        avg_mae = np.mean([r["mae"] for r in valid])
        print("-" * 70)
        print(
            f"  {'Avg':<6} {'':>8} {'':>9} "
            f"{avg_rmse:>7.1f} {avg_mae:>7.1f} {'':>16}"
        )
    print("=" * 70)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 60)
    print("PetroMind — Per-subset C-MAPSS training")
    print(f"Subsets   : {args.subsets}")
    print(f"Excel     : {args.excel}")
    print(f"LR        : {args.lr}  (vs 1e-3 in combined run)")
    print(f"Patience  : {args.early_stop_patience}  (vs 8 in combined run)")
    print(f"Drop sensors: {not args.no_drop_sensors}")
    print("=" * 60)

    results = []
    for subset in args.subsets:
        try:
            result = train_subset(subset, args.excel, args)
        except Exception as exc:
            print(f"\n  [ERROR] {subset} failed: {exc}")
            result = {"subset": subset, "engines": "?", "windows": 0,
                      "rmse": float("nan"), "mae": float("nan"),
                      "val_loss": float("nan")}
        results.append(result)

    print_comparison_table(results)
    print("\nDone. Individual models saved under checkpoints/FD00{N}/best_model.pt")


if __name__ == "__main__":
    main()
