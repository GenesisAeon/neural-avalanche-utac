"""Branching ratio estimator using the Harris (1963) method.

The branching ratio σ_b = E[descendants per ancestor spike] is the UTAC
state variable H(t). At criticality: σ_b = 1 = H* (the fixed point).
"""

from __future__ import annotations

import numpy as np


class BranchingRatioEstimator:
    """
    Estimates the network branching ratio σ_b from spike train data.

    Harris (1963) estimator:
        σ̂_b = Σ_{t: n_t > 0} n_{t+1} / Σ_{t: n_t > 0} n_t

    This is unbiased when n_{t+1} is driven solely by n_t (no external input
    during active bins). Spontaneous re-seeding only fires at n_t = 0, so
    those bins are excluded from both sums.

    Reference: Harris, T.E. (1963). The Theory of Branching Processes.
    """

    def estimate(self, spikes: np.ndarray) -> float:
        """
        Estimate σ_b from (n_neurons, n_bins) spike array.

        Returns float; 0.0 if no active time bins found.
        """
        n_t = spikes.sum(axis=0).astype(float)  # (n_bins,)
        return self._harris(n_t)

    def estimate_from_counts(self, n_t: np.ndarray) -> float:
        """Estimate σ_b from population count array (n_bins,)."""
        return self._harris(n_t.astype(float))

    def _harris(self, n_t: np.ndarray) -> float:
        # Only use time steps where current activity is positive
        active_mask = n_t[:-1] > 0
        if not active_mask.any():
            return 0.0
        numerator = float(n_t[1:][active_mask].sum())
        denominator = float(n_t[:-1][active_mask].sum())
        return numerator / denominator if denominator > 0 else 0.0

    def estimate_windowed(self, spikes: np.ndarray, window_bins: int = 5000) -> np.ndarray:
        """
        Rolling-window σ_b estimate using non-overlapping windows.

        Useful for tracking temporal drift of the branching ratio.
        """
        n_t = spikes.sum(axis=0).astype(float)
        n_bins = len(n_t)
        estimates: list[float] = []
        for start in range(0, n_bins - window_bins, window_bins):
            segment = n_t[start : start + window_bins]
            estimates.append(self._harris(segment))
        return np.array(estimates)

    def ar1_coefficient(self, spikes: np.ndarray) -> float:
        """
        AR(1) autocorrelation of population activity time series.

        At criticality the system exhibits critical slowing down: AR(1) → 1.
        Maps to the CREP C-component (coherence / critical slowing down).
        """
        n_t = spikes.sum(axis=0).astype(float)
        n_t -= n_t.mean()
        std = n_t.std()
        if std < 1e-10:
            return 0.0
        ac = float(np.corrcoef(n_t[:-1], n_t[1:])[0, 1])
        return float(np.clip(ac, -1.0, 1.0))
