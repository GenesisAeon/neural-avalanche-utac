"""Homeostatic plasticity model → UTAC r parameter.

The brain maintains σ_b near 1 (criticality) via synaptic scaling and
intrinsic excitability adjustments. The homeostatic rate r ≈ 0.15/hour
is estimated from Hengen lab in-vivo recordings (rat V1, 96h continuous).

This maps directly to the UTAC recovery rate r.
"""

from __future__ import annotations

import numpy as np


class HomeostaticPlasticity:
    """
    Simple first-order homeostatic controller for the branching ratio.

    Update rule (Euler discretisation):
        σ_b(t + Δt) = σ_b(t) + r · (σ* - σ_b(t)) · Δt

    where σ* = target branching ratio (H* = 1.0 at criticality) and r is
    the homeostatic rate (≈ 0.15 per hour from Hengen lab data).
    """

    def __init__(
        self,
        r: float = 0.15,
        target_sigma: float = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 1.99,
    ) -> None:
        self.r = r
        self.target_sigma = target_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._history: list[float] = []

    def update(self, current_sigma: float, dt_hours: float = 1.0) -> float:
        """
        Apply one homeostatic correction step.

        Returns the updated branching ratio (clipped to [sigma_min, sigma_max]).
        """
        error = self.target_sigma - current_sigma
        new_sigma = current_sigma + self.r * error * dt_hours
        new_sigma = float(np.clip(new_sigma, self.sigma_min, self.sigma_max))
        self._history.append(new_sigma)
        return new_sigma

    def effective_r(self, sigma_history: np.ndarray) -> float:
        """
        Estimate effective homeostatic rate from branching ratio time series.

        Least-squares fit of Δσ = r · (σ* - σ) · Δt.
        Useful for comparing model predictions with experimental data.
        """
        if len(sigma_history) < 3:
            return self.r
        delta = np.diff(sigma_history.astype(float))
        errors = self.target_sigma - sigma_history[:-1].astype(float)
        denom = float(np.dot(errors, errors))
        if denom == 0.0:
            return self.r
        return float(max(0.0, np.dot(delta, errors) / denom))

    def convergence_time(self, initial_sigma: float) -> float:
        """
        Analytical time constant τ = 1/r for first-order approach to target.

        Returns the 1/e convergence time in hours.
        """
        return 1.0 / self.r if self.r > 0 else float("inf")

    @property
    def history(self) -> np.ndarray:
        return np.array(self._history, dtype=float)
