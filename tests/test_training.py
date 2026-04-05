"""
Tests for model + training pipeline.

Covers:
    - DualHeadLSTM forward pass shapes
    - Trainer runs without error for a few epochs
    - Checkpoint save/load round-trip
    - Metrics computation
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from petromind.pipeline import (
    PipelineConfig, DualHeadLSTM, Trainer,
    build_dataloaders, build_sliding_windows,
    compute_classification_label, compute_rul,
    validate_dataframe,
)
from petromind.pipeline.utils import get_active_feature_cols


def _make_df(n_engines=5, min_life=40, max_life=60, seed=0):
    rng = np.random.RandomState(seed)
    frames = []
    for uid in range(1, n_engines + 1):
        life = rng.randint(min_life, max_life + 1)
        n = life
        import pandas as pd
        data = {
            "unit_id": np.full(n, uid, dtype=int),
            "cycle": np.arange(1, n + 1),
            "op_set_1": rng.randn(n),
            "op_set_2": rng.randn(n),
            "op_set_3": rng.randn(n),
        }
        for i in range(1, 22):
            data[f"s{i}"] = rng.randn(n)
        data["s1"] = np.full(n, 0.5)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _prep(cfg):
    df = _make_df()
    df = validate_dataframe(df, cfg)
    df = compute_rul(df, cfg)
    df = compute_classification_label(df, cfg)
    feat_cols = get_active_feature_cols(df, cfg)
    X, y_cls, y_rul, eids = build_sliding_windows(df, cfg, feat_cols)
    return X, y_cls, y_rul, eids


class TestModel:
    def test_forward_shapes(self):
        cfg = PipelineConfig(window_size=10, hidden_dim=32, n_lstm_layers=1)
        model = DualHeadLSTM(input_dim=8, cfg=cfg)
        x = torch.randn(4, 10, 8)
        cls_logit, rul_pred = model(x)
        assert cls_logit.shape == (4,)
        assert rul_pred.shape == (4,)
        assert (rul_pred >= 0).all()  # ReLU on RUL head

    def test_forward_different_batch(self):
        cfg = PipelineConfig(window_size=20, hidden_dim=16, n_lstm_layers=1)
        model = DualHeadLSTM(input_dim=5, cfg=cfg)
        for bs in [1, 16, 64]:
            cls_logit, rul_pred = model(torch.randn(bs, 20, 5))
            assert cls_logit.shape == (bs,)


class TestTrainer:
    def test_fit_runs(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=5, batch_size=32,
            hidden_dim=16, n_lstm_layers=1, epochs=3,
            early_stop_patience=10, model_dir=str(tmp_path),
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)
        model = DualHeadLSTM(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        history = trainer.fit(train_dl, val_dl)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert all(isinstance(m, dict) for m in history["val_metrics"])

    def test_checkpoint_roundtrip(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=5, batch_size=32,
            hidden_dim=16, n_lstm_layers=1, epochs=2,
            early_stop_patience=10, model_dir=str(tmp_path),
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)

        model = DualHeadLSTM(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        trainer.fit(train_dl, val_dl)

        ckpt_path = tmp_path / "best_model.pt"
        assert ckpt_path.exists()

        model2 = DualHeadLSTM(input_dim=X.shape[2], cfg=cfg)
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model2.eval()
        model.eval()

        x = torch.randn(2, cfg.window_size, X.shape[2])
        out1 = model(x)
        out2 = model2(x)
        torch.testing.assert_close(out1[0], out2[0])
        torch.testing.assert_close(out1[1], out2[1])

    def test_loss_decreases(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=1, batch_size=32,
            hidden_dim=32, n_lstm_layers=1, epochs=10,
            early_stop_patience=20, model_dir=str(tmp_path),
            learning_rate=1e-3,
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)
        model = DualHeadLSTM(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        history = trainer.fit(train_dl, val_dl)
        assert history["train_loss"][-1] < history["train_loss"][0]
