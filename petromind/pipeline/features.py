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

Fix (v2)
--------
Skewness and kurtosis are numerically unstable when a sensor is nearly
constant within a window (std ≈ 0).  scipy internally computes
``(x - mean)^3 / std^3``; with std ≈ 0 this causes catastrophic
cancellation and produces NaN / ±Inf that propagate silently through
training, causing the model to output the mean RUL for every sample.

The fix: detect near-constant windows *per feature* (std < STD_FLOOR),
suppress the RuntimeWarning, and zero-fill the affected cells after the
fact.  Zero is a neutral, gradient-safe sentinel — the model sees "no
higher-moment information here" rather than garbage.
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np
from scipy import stats as sp_stats

from .config import PipelineConfig

# Threshold below which a feature's within-window std is treated as
# "near-constant" and its skew/kurt set to 0.  1e-3 works well for
# C-MAPSS sensor scales (most sensors operate in 0–1000 range after
# normalisation, so 1e-3 is effectively zero variance).
_STD_FLOOR = 1e-3


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
        """Per-sensor mean, std, min, max, skewness, kurtosis.

        Near-constant windows (std < _STD_FLOOR) receive skew=0 and
        kurt=0 rather than numerically unstable / NaN values.
        """
        mean = X.mean(axis=1)                     # (N, F)
        std  = X.std(axis=1)                      # (N, F)
        mn   = X.min(axis=1)
        mx   = X.max(axis=1)

        # Mask: True where a (window, feature) cell is near-constant
        nearly_constant = std < _STD_FLOOR        # (N, F)

        # Suppress the RuntimeWarning — we handle bad cells below
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew = sp_stats.skew(X, axis=1)       # (N, F)
            kurt = sp_stats.kurtosis(X, axis=1)   # (N, F)

        # Zero-fill unstable cells (also catches any residual NaN/Inf)
        skew[nearly_constant] = 0.0
        kurt[nearly_constant] = 0.0
        skew = np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)
        kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

        return np.concatenate([mean, std, mn, mx, skew, kurt], axis=1)

    def _signal_features(self, X: np.ndarray) -> np.ndarray:
        """RMS and top-k FFT magnitudes per sensor."""
        N, W, F = X.shape
        rms = np.sqrt((X ** 2).mean(axis=1))  # (N, F)

        fft_vals = np.abs(np.fft.rfft(X, axis=1))  # (N, W//2+1, F)
        # Skip DC component (index 0), take top-k by magnitude
        fft_vals = fft_vals[:, 1:, :]
        k = min(self.cfg.fft_top_k, fft_vals.shape[1])

        top_k_indices = np.argsort(-fft_vals, axis=1)[:, :k, :]
        top_k = np.take_along_axis(fft_vals, top_k_indices, axis=1)  # (N, k, F)

        if k < self.cfg.fft_top_k:
            pad_shape = (N, self.cfg.fft_top_k - k, F)
            top_k = np.concatenate([top_k, np.zeros(pad_shape)], axis=1)

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

        t = np.arange(hw, dtype=np.float32)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        t_bc = t[np.newaxis, :, np.newaxis]
        y_mean = tail.mean(axis=1, keepdims=True)
        slope = ((t_bc - t_mean) * (tail - y_mean)).sum(axis=1) / (t_var + 1e-9)

        window_mean = X.mean(axis=1)
        tail_mean = tail.mean(axis=1)
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
            top_eigvals = eigvals[-k:][::-1]
            out[i, :k] = top_eigvals

        return out
