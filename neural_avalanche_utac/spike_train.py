"""Spike train generator and loader for neural avalanche simulation.

Implements a discrete-time branching process where each active neuron
independently generates σ_b offspring on average. At σ_b = 1 (criticality)
the avalanche size distribution follows P(S) ~ S^(-3/2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SpikeTrainConfig:
    """Configuration for spike train generation."""

    n_neurons: int = 500
    duration_s: float = 60.0       # simulation duration in seconds
    dt_ms: float = 2.0             # time bin width in ms
    seed: int = 42
    branching_ratio: float = 1.0   # target σ_b; 1.0 = criticality
    reseed_rate: float = 0.005     # probability of injecting a seed spike when quiescent

    @property
    def n_bins(self) -> int:
        return int(self.duration_s * 1000 / self.dt_ms)


class SpikeTrainGenerator:
    """
    Generates synthetic spike trains from a discrete-time branching process.

    Each time bin:
      1. Offspring: n_offspring ~ Poisson(σ_b · n_active_prev), mapped to random neurons.
      2. Re-seed: if network is quiescent, inject one spike with probability reseed_rate.

    The Harris (1963) estimator Σ n_{t+1} / Σ n_t is unbiased for σ_b under this scheme
    because re-seedings only fire when n_t = 0 (excluded from the ratio denominator).
    """

    def __init__(self, config: SpikeTrainConfig | None = None) -> None:
        self.config = config or SpikeTrainConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> dict:
        """
        Generate a spike train via branching process simulation.

        Returns
        -------
        dict with keys:
          spikes       : (n_neurons, n_bins) int8 array
          dt_ms        : float
          n_neurons    : int
          n_bins       : int
          branching_ratio : float (target value)
        """
        cfg = self.config
        n = cfg.n_neurons
        n_bins = cfg.n_bins
        sigma_b = cfg.branching_ratio

        spikes = np.zeros((n, n_bins), dtype=np.int8)

        # Seed the first time bin
        spikes[self.rng.integers(0, n), 0] = 1

        for t in range(1, n_bins):
            n_prev = int(spikes[:, t - 1].sum())

            if n_prev > 0:
                n_offspring = int(self.rng.poisson(sigma_b * n_prev))
                if n_offspring > 0:
                    # Multiple offspring can target the same neuron (deduplicated by bool mask)
                    targets = self.rng.integers(0, n, size=min(n_offspring, n * 3))
                    mask = np.zeros(n, dtype=bool)
                    mask[targets] = True
                    spikes[:, t] = mask.astype(np.int8)
            else:
                # Re-seed quiescent network
                if self.rng.random() < cfg.reseed_rate:
                    spikes[self.rng.integers(0, n), t] = 1

        return {
            "spikes": spikes,
            "dt_ms": cfg.dt_ms,
            "n_neurons": n,
            "n_bins": n_bins,
            "branching_ratio": sigma_b,
        }


class SpikeTrainLoader:
    """Load/save/generate spike train data in .npz format."""

    @staticmethod
    def save(path: str | Path, data: dict) -> None:
        arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(str(path), **arrays)

    @staticmethod
    def load(path: str | Path) -> dict:
        npz = np.load(str(path))
        return dict(npz)

    @staticmethod
    def generate_synthetic(path: str | Path, seed: int = 42) -> dict:
        """
        Generate and save synthetic spike trains at three branching ratios:
          subcritical (σ_b=0.7), critical (σ_b=1.0), supercritical (σ_b=1.3).

        Used as the canonical test dataset (seed=42, GenesisAeon standard).
        """
        configs = [
            SpikeTrainConfig(n_neurons=200, duration_s=120.0, branching_ratio=0.7, seed=seed),
            SpikeTrainConfig(n_neurons=200, duration_s=120.0, branching_ratio=1.0, seed=seed + 1),
            SpikeTrainConfig(n_neurons=200, duration_s=120.0, branching_ratio=1.3, seed=seed + 2),
        ]
        labels = ["subcritical", "critical", "supercritical"]
        all_data: dict = {"dt_ms": np.array([2.0])}

        for cfg, label in zip(configs, labels):
            gen = SpikeTrainGenerator(cfg)
            result = gen.generate()
            all_data[f"spikes_{label}"] = result["spikes"]
            all_data[f"branching_{label}"] = np.array([cfg.branching_ratio])

        np.savez_compressed(str(path), **all_data)
        return all_data
