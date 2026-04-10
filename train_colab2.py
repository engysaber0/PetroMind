#!/usr/bin/env python3
"""
train_colab_v3.py — PetroMind RUL training with composite asymmetric loss.

Key change vs previous versions
--------------------------------
Replaces pure MSE with a blend of MSE + NASA asymmetric scoring loss.
MSE is symmetric, so the model finds the globally-optimal constant prediction
(~mean RUL ~42) and stays there.  The NASA loss penalises *late* predictions
(predicting more life than remains) more harshly than early ones, breaking
the plateau.

Alpha annealing schedule
------------------------
  Epochs 1 - anneal_start : alpha = alpha_start  (more MSE, stable gradients)
  Epochs anneal_start - end : alpha decays linearly to alpha_end (more NASA)

Recommended run:
    !python train_colab_v3.py --excel /content/All_train_data.xlsx \
        --epochs 80 --alpha-start 0.95 --alpha-end 0.3 --anneal-start 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

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


# ── Loss definitions (inline — no separate loss.py needed) ────────────────────

class NASAScoringLoss(nn.Module):
    def __init__(self, h_early=10.0, h_late=13.0):
        super().__init__()
        self.h_early = h_early
        self.h_late = h_late

    def forward(self, pred, target):
        pred, target = pred.view(-1), target.view(-1)
        d = pred - target
        h = torch.where(
            d < 0,
            torch.full_like(d, self.h_early),
            torch.full_like(d, self.h_late),
        )
        return (torch.exp(d / h) - 1.0).mean()


class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.95, h_early=10.0, h_late=13.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.nasa = NASAScoringLoss(h_early, h_late)

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1.0 - self.alpha) * self.nasa(pred, target)

    def set_alpha(self, alpha):
        self.alpha = float(max(0.0, min(1.0, alpha)))


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train RUL model with asymmetric loss.")
    p.add_argument("--excel", type=str, default="/content/All_train_data.xlsx")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--prediction-horizon", type=int, default=30)
    p.add_argument("--rul-clip", type=int, default=125)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-lstm-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--model-dir", type=str, default="checkpoints")
    p.add_argument("--alpha-start", type=float, default=0.95,
                   help="Initial MSE weight (1.0=pure MSE, 0.0=pure NASA).")
    p.add_argument("--alpha-end", type=float, default=0.3,
                   help="Final MSE weight after annealing.")
    p.add_argument("--anneal-start", type=int, default=30,
                   help="Epoch at which alpha annealing begins.")
    return p.parse_args(argv)


# ── Batch unpacking ───────────────────────────────────────────────────────────

def _unpack_batch(batch):
    """Handle dict batches (PetroMind default) and tuple batches."""
    if isinstance(batch, dict):
        return batch["features"], batch["rul"]
    tensors = [b for b in batch if isinstance(b, torch.Tensor)]
    if len(tensors) >= 2:
        X = next((t for t in tensors if t.dim() == 3), tensors[0])
        y_rul = next((t for t in reversed(tensors) if t.dim() == 1), tensors[-1])
        return X, y_rul
    raise ValueError(f"Cannot unpack batch: {[type(b) for b in batch]}")


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, preds_all, targets_all = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            X, y_rul = _unpack_batch(batch)
            X = X.to(device, non_blocking=True)
            y_rul = y_rul.to(device, non_blocking=True)

            pred = model(X).squeeze(-1)
            loss = criterion(pred, y_rul.float())

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds_all.append(pred.detach().cpu())
            targets_all.append(y_rul.cpu())

    n = sum(t.size(0) for t in targets_all)
    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    return total_loss / n, rmse, mae


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print("=" * 60)
    print(f"Loss   : CompositeLoss  alpha {args.alpha_start} -> {args.alpha_end}")
    print(f"Anneal : starts epoch {args.anneal_start}")
    print(f"Device : {device}")
    print("=" * 60)

    print("\nStep 1 - Load all sheets")
    raw_df = load_cmapss_excel_all_sheets(args.excel)
    print(f"  Shape: {raw_df.shape}  |  Engines: {raw_df['unit_id'].nunique()}")

    print("\nStep 2 - Validate & clean")
    clean_df = validate_dataframe(raw_df, cfg)

    print("\nStep 3 - Labels")
    labeled_df = compute_rul(clean_df, cfg)
    labeled_df = compute_classification_label(labeled_df, cfg)

    print("\nStep 4 - Sliding windows")
    feature_cols = get_active_feature_cols(labeled_df, cfg)
    X_raw, y_cls, y_rul, engine_ids = build_sliding_windows(
        labeled_df, cfg, feature_cols=feature_cols,
    )
    print(f"  X shape: {X_raw.shape}")

    if X_raw.shape[0] == 0:
        print("No windows produced. Exiting.")
        sys.exit(1)

    print("\nStep 5 - Feature engineering")
    extractor = FeatureExtractor(cfg, n_pca_components=3)
    extractor.transform(X_raw)
    print("  Done (no RuntimeWarning expected)")

    print("\nStep 6 - DataLoaders")
    train_loader, val_loader, _ = build_dataloaders(
        X_raw, y_cls, y_rul, engine_ids, cfg,
    )
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    print("\nStep 7 - Train with CompositeLoss")
    input_dim = X_raw.shape[2]
    model = LSTMRULModel(input_dim=input_dim, cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  |  Device: {device}\n")

    criterion = CompositeLoss(alpha=args.alpha_start)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    best_rmse = float("inf")
    no_improve = 0
    best_path = f"{args.model_dir}/best_model.pt"

    alpha_range = args.alpha_start - args.alpha_end
    anneal_epochs = max(args.epochs - args.anneal_start, 1)

    for epoch in range(1, args.epochs + 1):
        if epoch >= args.anneal_start:
            frac = (epoch - args.anneal_start) / anneal_epochs
            new_alpha = args.alpha_start - alpha_range * min(frac, 1.0)
            criterion.set_alpha(new_alpha)

        tr_loss, tr_rmse, tr_mae = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        va_loss, va_rmse, va_mae = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"train={tr_loss:.2f}  val={va_loss:.2f}  "
            f"lr={lr_now:.1e}  alpha={criterion.alpha:.2f}  "
            f"RMSE={va_rmse:.1f}  MAE={va_mae:.1f}"
        )

        if va_rmse < best_rmse:
            best_rmse = va_rmse
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.early_stop_patience})")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, final_rmse, final_mae = run_epoch(
        model, val_loader, criterion, optimizer, device, train=False
    )

    print(f"\n{'=' * 60}")
    print("Step 8 - Final evaluation")
    print("=" * 60)
    print(f"  RMSE (RUL) : {final_rmse:.1f}")
    print(f"  MAE  (RUL) : {final_mae:.1f}")
    print(f"  Best model : {best_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
