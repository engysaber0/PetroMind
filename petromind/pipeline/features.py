"""
Feature engineering for RUL prediction.

Extracts a rich feature vector **per window** from the raw
(N, window_size, n_raw_features) tensor.  All computations operate on the
window *as given* — no future data ever leaks in.

Feature groups
--------------
1. **Statistical** — mean, std, min, max, skewness, kurtosis per sensor.
2. **Signal / frequency** — RMS, top-k FFT magnitude bins per sensor.
3. **Health indicators** — rolling-mean trend slope and last-value-vs-mean
   ratio (captures degradation trajectory within the window).
4. **Sensor fusion** — first *k* principal components across all sensors
   in each window (captures correlated multi-sensor degradation).

The ``FeatureExtractor`` class is stateless and purely functional: it maps
(N, W, F_raw) → (N, F_eng).  It can therefore be applied identically to
train and test sets without any fit/transform asymmetry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import stats as sp_stats

from .config import PipelineConfig


class FeatureExtractor:
    """Stateless per-window feature extraction.

    Parameters
    ----------
    cfg : PipelineConfig
        Controls ``fft_top_k`` and ``rolling_health_window``.
    n_pca_components : int
        Number of PCA components for the sensor-fusion block.
    """

    def __init__(self, cfg: PipelineConfig, n_pca_components: int = 3):
        self.cfg = cfg
        self.n_pca_components = n_pca_components

    # ── public API ────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract engineered features from raw windows.

        Parameters
        ----------
        X : np.ndarray, shape (N, W, F)

        Returns
        -------
        np.ndarray, shape (N, F_eng)   — float32
        """
        parts = [
            self._statistical_features(X),
            self._signal_features(X),
            self._health_indicators(X),
            self._sensor_fusion(X),
        ]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def feature_names(self, raw_feature_names: List[str]) -> List[str]:
        """Return human-readable names for every engineered feature."""
        names: List[str] = []
        for fn in raw_feature_names:
            names += [f"{fn}_mean", f"{fn}_std", f"{fn}_min", f"{fn}_max",
                      f"{fn}_skew", f"{fn}_kurtosis"]
        for fn in raw_feature_names:
            names.append(f"{fn}_rms")
            for k in range(self.cfg.fft_top_k):
                names.append(f"{fn}_fft_{k}")
        for fn in raw_feature_names:
            names += [f"{fn}_trend_slope", f"{fn}_tail_mean_ratio"]
        for k in range(self.n_pca_components):
            names.append(f"pca_{k}")
        return names

    # ── feature blocks ────────────────────────────────────────────────

    @staticmethod
    def _statistical_features(X: np.ndarray) -> np.ndarray:
        """Per-sensor mean, std, min, max, skewness, kurtosis."""
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        mn = X.min(axis=1)
        mx = X.max(axis=1)
        skew = sp_stats.skew(X, axis=1)
        kurt = sp_stats.kurtosis(X, axis=1)
        return np.concatenate([mean, std, mn, mx, skew, kurt], axis=1)

    def _signal_features(self, X: np.ndarray) -> np.ndarray:
        """RMS and top-k FFT magnitudes per sensor."""
        N, W, F = X.shape
        rms = np.sqrt((X ** 2).mean(axis=1))  # (N, F)

        fft_vals = np.abs(np.fft.rfft(X, axis=1))  # (N, W//2+1, F)
        # Skip DC component (index 0), take top-k by magnitude
        fft_vals = fft_vals[:, 1:, :]  # drop DC
        k = min(self.cfg.fft_top_k, fft_vals.shape[1])

        # Sort along frequency axis descending and keep top k
        top_k_indices = np.argsort(-fft_vals, axis=1)[:, :k, :]
        top_k = np.take_along_axis(fft_vals, top_k_indices, axis=1)  # (N, k, F)

        # Pad if fewer frequency bins than k
        if k < self.cfg.fft_top_k:
            pad_shape = (N, self.cfg.fft_top_k - k, F)
            top_k = np.concatenate([top_k, np.zeros(pad_shape)], axis=1)

        # Reshape: (N, fft_top_k * F)
        top_k_flat = top_k.reshape(N, -1)
        # Interleave so layout is [f0_fft0, f0_fft1, ..., f1_fft0, ...]
        top_k_reordered = top_k.transpose(0, 2, 1).reshape(N, -1)

        return np.concatenate([rms, top_k_reordered], axis=1)

    def _health_indicators(self, X: np.ndarray) -> np.ndarray:
        """Degradation-trend features computed within each window.

        * **trend_slope** — OLS slope of a small rolling mean applied to
          the last ``rolling_health_window`` timesteps.  A negative slope
          signals accelerating degradation.
        * **tail_mean_ratio** — ratio of the mean of the last
          ``rolling_health_window`` timesteps to the overall window mean.
          Values > 1 indicate the sensor is rising toward end-of-life.
        """
        N, W, F = X.shape
        hw = min(self.cfg.rolling_health_window, W)

        tail = X[:, -hw:, :]  # (N, hw, F)

        # Trend slope via least-squares on the tail
        t = np.arange(hw, dtype=np.float32)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        tail_mean_t = tail.mean(axis=1, keepdims=True)  # not used for slope
        # slope = sum((t - t_mean)*(y - y_mean)) / sum((t - t_mean)^2)
        # Broadcast t to (1, hw, 1)
        t_bc = t[np.newaxis, :, np.newaxis]
        y_mean = tail.mean(axis=1, keepdims=True)
        slope = ((t_bc - t_mean) * (tail - y_mean)).sum(axis=1) / (t_var + 1e-9)  # (N, F)

        # Tail-vs-whole ratio
        window_mean = X.mean(axis=1)  # (N, F)
        tail_mean = tail.mean(axis=1)  # (N, F)
        ratio = tail_mean / (window_mean + 1e-9)

        return np.concatenate([slope, ratio], axis=1)

    def _sensor_fusion(self, X: np.ndarray) -> np.ndarray:
        """Window-level PCA across all sensors.

        For each window the covariance matrix of the F sensors over W
        timesteps is computed, then projected onto the top-k eigenvectors.
        The resulting component *variances* serve as features — they
        capture how much correlated energy exists along each principal axis.
        """
        N, W, F = X.shape
        k = min(self.n_pca_components, F)
        out = np.zeros((N, self.n_pca_components), dtype=np.float32)

        for i in range(N):
            xi = X[i]  # (W, F)
            xi_centered = xi - xi.mean(axis=0, keepdims=True)
            cov = (xi_centered.T @ xi_centered) / max(W - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)
            # eigvalsh returns ascending; take last k (largest)
            top_eigvals = eigvals[-k:][::-1]
            out[i, :k] = top_eigvals

        return out
