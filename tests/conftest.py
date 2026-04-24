"""Shared fixtures for neural-avalanche-utac tests."""

from __future__ import annotations

import numpy as np
import pytest

from neural_avalanche_utac.spike_train import SpikeTrainConfig, SpikeTrainGenerator


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def spikes_critical() -> np.ndarray:
    """200-neuron critical (σ_b=1) spike train, 30s, seed=42."""
    cfg = SpikeTrainConfig(n_neurons=200, duration_s=30.0, dt_ms=2.0, branching_ratio=1.0, seed=42)
    return SpikeTrainGenerator(cfg).generate()["spikes"]


@pytest.fixture(scope="session")
def spikes_subcritical() -> np.ndarray:
    """200-neuron subcritical (σ_b=0.6) spike train, 30s."""
    cfg = SpikeTrainConfig(n_neurons=200, duration_s=30.0, dt_ms=2.0, branching_ratio=0.6, seed=43)
    return SpikeTrainGenerator(cfg).generate()["spikes"]


@pytest.fixture(scope="session")
def spikes_supercritical() -> np.ndarray:
    """200-neuron supercritical (σ_b=1.4) spike train, 30s."""
    cfg = SpikeTrainConfig(n_neurons=200, duration_s=30.0, dt_ms=2.0, branching_ratio=1.4, seed=44)
    return SpikeTrainGenerator(cfg).generate()["spikes"]
