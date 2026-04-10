"""
NASA C-MAPSS asymmetric scoring loss for RUL regression.

Why MSE fails on C-MAPSS
-------------------------
MSE is symmetric: being 20 cycles early costs the same as being 20 cycles
late.  On C-MAPSS the consequences are asymmetric — predicting *more* life
than remains is far more dangerous than predicting *less*.  A model trained
on MSE finds the globally optimal constant prediction (~mean RUL) and sits
there, producing RMSE ≈ 41 regardless of architecture or data splits.

The NASA scoring function
--------------------------
Defined in Saxena et al. (2008) "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation":

    s_i = exp( d_i / h ) - 1

where  d_i = RUL_pred - RUL_true  (positive = late, negative = early)
and    h   = 10  if d_i < 0  (early prediction — lighter penalty)
             13  if d_i >= 0 (late prediction  — heavier penalty)

Score = sum(s_i)  — lower is better.

As a loss we use the per-sample mean so batch size doesn't affect the scale.

Composite loss
--------------
Pure NASA score has very steep gradients for large late errors, which can
destabilise early training.  We blend it with MSE:

    loss = alpha * MSE  +  (1 - alpha) * NASA_score

Start with alpha=0.5 to keep training stable, then reduce toward 0.2 once
the model has learned the basic RUL trend (e.g. after epoch 10).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NASAScoringLoss(nn.Module):
    """Asymmetric NASA C-MAPSS scoring loss.

    Parameters
    ----------
    h_early : float
        Denominator for early predictions (d < 0).  Default 10.
    h_late : float
        Denominator for late predictions (d >= 0).  Default 13.
    reduction : str
        'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        h_early: float = 10.0,
        h_late: float = 13.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.h_early = h_early
        self.h_late = h_late
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (N,) or (N,1) predicted RUL
        target : (N,) or (N,1) true RUL
        """
        pred = pred.view(-1)
        target = target.view(-1)
        d = pred - target  # positive = late, negative = early

        h = torch.where(d < 0, torch.full_like(d, self.h_early), torch.full_like(d, self.h_late))
        score = torch.exp(d / h) - 1.0

        return score.mean() if self.reduction == "mean" else score.sum()


class CompositeLoss(nn.Module):
    """Blend of MSE and NASA scoring loss.

    Parameters
    ----------
    alpha : float
        Weight on MSE in [0, 1].  (1 - alpha) goes to NASA score.
        alpha=1.0 → pure MSE (original behaviour).
        alpha=0.0 → pure NASA score (may be unstable early in training).
        alpha=0.5 → recommended starting point.
    h_early, h_late : float
        Passed through to NASAScoringLoss.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        h_early: float = 10.0,
        h_late: float = 13.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.nasa = NASAScoringLoss(h_early=h_early, h_late=h_late)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(pred, target) + (1.0 - self.alpha) * self.nasa(pred, target)

    def set_alpha(self, alpha: float) -> None:
        """Adjust blend weight mid-training (e.g. anneal from 0.5 → 0.2)."""
        self.alpha = max(0.0, min(1.0, alpha))
