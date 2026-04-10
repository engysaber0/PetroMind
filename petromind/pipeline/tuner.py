"""
Hyperparameter tuning utilities.

Provides simple grid search and random search for the LSTM RUL model.
For more advanced tuning, integrate with Optuna externally.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .dataset import SensorNormalizer
from .features import FeatureExtractor
from .models import LSTMRULModel
from .trainer import Trainer


@dataclass
class TuningResult:
    """Result of a single hyperparameter configuration."""
    params: Dict[str, Any]
    val_rmse: float
    val_mae: float
    val_score: float
    best_epoch: int


def grid_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_raw: np.ndarray,
    cfg: PipelineConfig,
    param_grid: Dict[str, List[Any]],
    feature_extractor: Optional[FeatureExtractor] = None,
    n_pca_components: int = 3,
    use_raw_windows: bool = True,
    verbose: bool = True,
) -> Tuple[TuningResult, List[TuningResult]]:
    """Exhaustive grid search over hyperparameters.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
        DataLoaders created with build_dataloaders().
    X_raw : np.ndarray
        Raw windowed features (N, W, F) for creating normalizer.
    cfg : PipelineConfig
        Base configuration (will be overridden with trial params).
    param_grid : dict of list
        Hyperparameters to search. Keys must be valid PipelineConfig fields.
        Example: {"hidden_dim": [32, 64, 128], "learning_rate": [1e-3, 1e-4]}
    feature_extractor : FeatureExtractor, optional
        If provided, use engineered features instead of raw windows.
    n_pca_components : int
        Number of PCA components for feature extraction.
    use_raw_windows : bool
        If True, use raw windowed data (X_raw). If False, use engineered features.
    verbose : bool
        Print progress.

    Returns
    -------
    best_result : TuningResult
        Best hyperparameter configuration.
    all_results : list[TuningResult]
        All configurations tested, sorted by val_rmse (ascending).
    """
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    if verbose:
        print(f"Grid search: {len(combinations)} configurations")
        print(f"Parameters: {param_grid}")

    all_results: List[TuningResult] = []

    for combo in combinations:
        params = dict(zip(keys, combo))

        # Build config for this trial
        trial_cfg = _make_config(cfg, params)

        # Determine input dimension
        if use_raw_windows:
            input_dim = X_raw.shape[2]
        else:
            # Use engineered features
            if feature_extractor is None:
                raise ValueError("feature_extractor required when use_raw_windows=False")
            X_eng = feature_extractor.transform(X_raw)
            input_dim = X_eng.shape[1]

        # Create and train model
        model = LSTMRULModel(input_dim=input_dim, cfg=trial_cfg)
        trainer = Trainer(model=model, cfg=trial_cfg)

        # Train for this trial
        if verbose:
            print(f"\n  Testing: {params}")

        history = trainer.fit(train_loader, val_loader)

        # Get final validation metrics
        val_loss, val_metrics = trainer.evaluate(val_loader)

        result = TuningResult(
            params=params,
            val_rmse=val_metrics["rmse"],
            val_mae=val_metrics["mae"],
            val_score=val_metrics["score"],
            best_epoch=len(history["val_loss"]),
        )
        all_results.append(result)

        if verbose:
            print(f"  Result: RMSE={val_metrics['rmse']:.1f}, MAE={val_metrics['mae']:.1f}, Score={val_metrics['score']:.1f}")

    # Sort by RMSE (ascending)
    all_results.sort(key=lambda r: r.val_rmse)
    best_result = all_results[0]

    if verbose:
        print(f"\n  Best: {best_result.params}")
        print(f"  Best RMSE={best_result.val_rmse:.1f}, MAE={best_result.val_mae:.1f}")

    return best_result, all_results


def random_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_raw: np.ndarray,
    cfg: PipelineConfig,
    param_distributions: Dict[str, List[Any]],
    n_trials: int = 20,
    feature_extractor: Optional[FeatureExtractor] = None,
    n_pca_components: int = 3,
    use_raw_windows: bool = True,
    verbose: bool = True,
    seed: int = 42,
) -> Tuple[TuningResult, List[TuningResult]]:
    """Random search over hyperparameters.

    More efficient than grid search when searching large spaces.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
    X_raw : np.ndarray
    cfg : PipelineConfig
    param_distributions : dict of list
        Values to sample from for each parameter.
    n_trials : int
        Number of random configurations to test.
    feature_extractor : FeatureExtractor, optional
    n_pca_components : int
    use_raw_windows : bool
    verbose : bool
    seed : int

    Returns
    -------
    best_result : TuningResult
    all_results : list[TuningResult]
    """
    rng = np.random.RandomState(seed)
    keys = list(param_distributions.keys())
    values = list(param_distributions.values())

    if verbose:
        print(f"Random search: {n_trials} trials")
        print(f"Parameters: {param_distributions}")

    all_results: List[TuningResult] = []

    for trial_idx in range(n_trials):
        # Sample random combination
        params = {k: values[i][rng.randint(len(values[i]))] for i, k in enumerate(keys)}

        trial_cfg = _make_config(cfg, params)

        if use_raw_windows:
            input_dim = X_raw.shape[2]
        else:
            if feature_extractor is None:
                raise ValueError("feature_extractor required when use_raw_windows=False")
            X_eng = feature_extractor.transform(X_raw)
            input_dim = X_eng.shape[1]

        model = LSTMRULModel(input_dim=input_dim, cfg=trial_cfg)
        trainer = Trainer(model=model, cfg=trial_cfg)

        if verbose:
            print(f"\n  Trial {trial_idx + 1}/{n_trials}: {params}")

        history = trainer.fit(train_loader, val_loader)
        val_loss, val_metrics = trainer.evaluate(val_loader)

        result = TuningResult(
            params=params,
            val_rmse=val_metrics["rmse"],
            val_mae=val_metrics["mae"],
            val_score=val_metrics["score"],
            best_epoch=len(history["val_loss"]),
        )
        all_results.append(result)

        if verbose:
            print(f"  Result: RMSE={val_metrics['rmse']:.1f}, MAE={val_metrics['mae']:.1f}")

    all_results.sort(key=lambda r: r.val_rmse)
    best_result = all_results[0]

    if verbose:
        print(f"\n  Best: {best_result.params}")
        print(f"  Best RMSE={best_result.val_rmse:.1f}")

    return best_result, all_results


def _make_config(base_cfg: PipelineConfig, params: Dict[str, Any]) -> PipelineConfig:
    """Create a new PipelineConfig with overridden parameters."""
    return PipelineConfig(
        window_size=params.get("window_size", base_cfg.window_size),
        stride=params.get("stride", base_cfg.stride),
        prediction_horizon=params.get("prediction_horizon", base_cfg.prediction_horizon),
        rul_clip=params.get("rul_clip", base_cfg.rul_clip),
        val_ratio=params.get("val_ratio", base_cfg.val_ratio),
        batch_size=params.get("batch_size", base_cfg.batch_size),
        fft_top_k=params.get("fft_top_k", base_cfg.fft_top_k),
        epochs=params.get("epochs", base_cfg.epochs),
        learning_rate=params.get("learning_rate", base_cfg.learning_rate),
        weight_decay=params.get("weight_decay", base_cfg.weight_decay),
        hidden_dim=params.get("hidden_dim", base_cfg.hidden_dim),
        n_lstm_layers=params.get("n_lstm_layers", base_cfg.n_lstm_layers),
        dropout=params.get("dropout", base_cfg.dropout),
        early_stop_patience=params.get("early_stop_patience", base_cfg.early_stop_patience),
        model_dir=params.get("model_dir", base_cfg.model_dir),
    )
