"""
Tests for LSTM RUL model + training pipeline.

Covers:
    - LSTMRULModel forward pass shapes
    - Trainer runs without error for a few epochs
    - Checkpoint save/load round-trip
    - Loss decreases during training
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from petromind.pipeline import (
    PipelineConfig, LSTMRULModel, Trainer,
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
        model = LSTMRULModel(input_dim=8, cfg=cfg)
        x = torch.randn(4, 10, 8)
        rul_pred = model(x)
        assert rul_pred.shape == (4,)
        assert (rul_pred >= 0).all()

    def test_forward_different_batch(self):
        cfg = PipelineConfig(window_size=20, hidden_dim=16, n_lstm_layers=1)
        model = LSTMRULModel(input_dim=5, cfg=cfg)
        for bs in [1, 16, 64]:
            rul_pred = model(torch.randn(bs, 20, 5))
            assert rul_pred.shape == (bs,)


class TestTrainer:
    def test_fit_runs(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=5, batch_size=32,
            hidden_dim=16, n_lstm_layers=1, epochs=3,
            early_stop_patience=10, model_dir=str(tmp_path),
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)
        model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        history = trainer.fit(train_dl, val_dl)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert all(isinstance(m, dict) for m in history["val_metrics"])
        assert "rmse" in history["val_metrics"][0]
        assert "mae" in history["val_metrics"][0]

    def test_checkpoint_roundtrip(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=5, batch_size=32,
            hidden_dim=16, n_lstm_layers=1, epochs=2,
            early_stop_patience=10, model_dir=str(tmp_path),
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)

        model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        trainer.fit(train_dl, val_dl)

        ckpt_path = tmp_path / "best_model.pt"
        assert ckpt_path.exists()

        model2 = LSTMRULModel(input_dim=X.shape[2], cfg=cfg)
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model2.eval()
        model.eval()

        x = torch.randn(2, cfg.window_size, X.shape[2])
        out1 = model(x)
        out2 = model2(x)
        torch.testing.assert_close(out1, out2)

    def test_loss_decreases(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=1, batch_size=32,
            hidden_dim=32, n_lstm_layers=1, epochs=10,
            early_stop_patience=20, model_dir=str(tmp_path),
            learning_rate=1e-3,
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)
        model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        history = trainer.fit(train_dl, val_dl)
        assert history["train_loss"][-1] < history["train_loss"][0]


class TestMetrics:
    def test_score_asymmetric(self):
        """NASA score penalizes late predictions more than early ones."""
        from petromind.pipeline.trainer import _compute_metrics

        # Late predictions (pred > true) should be penalized more
        y_true = np.array([50.0, 50.0, 50.0])
        y_pred_late = np.array([70.0, 80.0, 100.0])  # over-estimate
        y_pred_early = np.array([30.0, 20.0, 0.0])   # under-estimate

        metrics_late = _compute_metrics(y_true, y_pred_late)
        metrics_early = _compute_metrics(y_true, y_pred_early)

        # Late predictions should have higher (worse) score
        assert metrics_late["score"] > metrics_early["score"]

    def test_metrics_include_score(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=5, batch_size=32,
            hidden_dim=16, n_lstm_layers=1, epochs=2,
            early_stop_patience=10, model_dir=str(tmp_path),
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)
        model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        _, metrics = trainer.evaluate(val_dl)
        assert "score" in metrics


class TestPredictionExport:
    def test_export_predictions(self, tmp_path):
        cfg = PipelineConfig(
            window_size=10, stride=5, batch_size=32,
            hidden_dim=16, n_lstm_layers=1, epochs=2,
            early_stop_patience=10, model_dir=str(tmp_path),
        )
        X, y_cls, y_rul, eids = _prep(cfg)
        train_dl, val_dl, _ = build_dataloaders(X, y_cls, y_rul, eids, cfg)
        model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg)
        trainer = Trainer(model=model, cfg=cfg)
        trainer.fit(train_dl, val_dl)

        output_path = tmp_path / "predictions.csv"
        trainer.export_predictions(val_dl, str(output_path))
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert "true_rul" in df.columns
        assert "predicted_rul" in df.columns
        assert "error" in df.columns
        assert "engine_id" in df.columns


class TestSensorNormalizer:
    def test_normalizer_fit_transform(self):
        from petromind.pipeline import SensorNormalizer

        X_train = np.random.randn(100, 30, 8)
        X_val = np.random.randn(50, 30, 8)

        normalizer = SensorNormalizer()
        X_train_norm = normalizer.fit_transform(X_train)
        X_val_norm = normalizer.transform(X_val)

        # Normalized train data should have ~0 mean and ~1 std
        assert np.allclose(X_train_norm.mean(axis=(0, 1)), 0, atol=1e-6)
        assert np.allclose(X_train_norm.std(axis=(0, 1)), 1, atol=1e-6)
        assert X_train_norm.shape == X_train.shape
        assert X_val_norm.shape == X_val.shape

    def test_normalizer_engineered_features(self):
        from petromind.pipeline import SensorNormalizer

        X_train = np.random.randn(100, 50)  # (N, F_eng)
        X_val = np.random.randn(50, 50)

        normalizer = SensorNormalizer()
        X_train_norm = normalizer.fit_transform(X_train)
        X_val_norm = normalizer.transform(X_val)

        assert np.allclose(X_train_norm.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_train_norm.std(axis=0), 1, atol=1e-6)
