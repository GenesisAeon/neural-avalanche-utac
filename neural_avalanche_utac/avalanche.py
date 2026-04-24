"""Neuronal avalanche detection using the Beggs & Plenz (2003) method.

An avalanche is a contiguous sequence of active time bins (population activity > 0)
bounded on both sides by silent bins. Size S = total spike count, duration D = bin count.

At criticality: P(S) ~ S^(-3/2), P(D) ~ D^(-2)  (mean-field branching process).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Avalanche:
    """Single neuronal avalanche event."""

    size: int        # total spike count across all neurons and bins
    duration: int    # number of active time bins
    start_bin: int   # first active bin index
    end_bin: int     # last active bin index (inclusive)


class AvalancheDetector:
    """
    Detects neuronal avalanches from population spike count time series.

    Reference: Beggs, J.M. & Plenz, D. (2003). J. Neurosci. 23(35), 11167-11177.
    """

    def __init__(self, threshold: int = 0) -> None:
        # Bins with spike count > threshold are considered active
        self.threshold = threshold

    def detect(self, spikes: np.ndarray) -> list[Avalanche]:
        """
        Detect avalanches from spike array.

        Parameters
        ----------
        spikes : (n_neurons, n_bins) int array

        Returns
        -------
        List of Avalanche objects sorted by start time.
        """
        population = spikes.sum(axis=0)  # (n_bins,)
        return self.detect_from_counts(population)

    def detect_from_counts(self, population: np.ndarray) -> list[Avalanche]:
        """Detect avalanches directly from population count array (n_bins,)."""
        active = population > self.threshold
        avalanches: list[Avalanche] = []

        in_av = False
        start = 0
        size = 0
        duration = 0

        for t, is_active in enumerate(active):
            if is_active and not in_av:
                in_av = True
                start = t
                size = int(population[t])
                duration = 1
            elif is_active and in_av:
                size += int(population[t])
                duration += 1
            elif not is_active and in_av:
                avalanches.append(Avalanche(size=size, duration=duration, start_bin=start, end_bin=t - 1))
                in_av = False
                size = 0
                duration = 0

        if in_av:
            avalanches.append(Avalanche(size=size, duration=duration, start_bin=start, end_bin=len(active) - 1))

        return avalanches

    def sizes(self, avalanches: list[Avalanche]) -> np.ndarray:
        """Return array of avalanche sizes."""
        return np.array([a.size for a in avalanches], dtype=float)

    def durations(self, avalanches: list[Avalanche]) -> np.ndarray:
        """Return array of avalanche durations."""
        return np.array([a.duration for a in avalanches], dtype=float)

    def summary(self, avalanches: list[Avalanche]) -> dict:
        """Return descriptive statistics for detected avalanches."""
        if not avalanches:
            return {"n": 0, "mean_size": float("nan"), "mean_duration": float("nan"),
                    "max_size": 0, "max_duration": 0}
        s = self.sizes(avalanches)
        d = self.durations(avalanches)
        return {
            "n": len(avalanches),
            "mean_size": float(s.mean()),
            "mean_duration": float(d.mean()),
            "max_size": int(s.max()),
            "max_duration": int(d.max()),
        }
