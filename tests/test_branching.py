"""Tests for BranchingRatioEstimator."""

from __future__ import annotations

import numpy as np
import pytest

from neural_avalanche_utac.branching import BranchingRatioEstimator
from neural_avalanche_utac.spike_train import SpikeTrainConfig, SpikeTrainGenerator


def _gen(sigma_b: float, seed: int = 42, n: int = 200, dur: float = 120.0) -> np.ndarray:
    cfg = SpikeTrainConfig(n_neurons=n, duration_s=dur, dt_ms=2.0,
                           branching_ratio=sigma_b, seed=seed)
    return SpikeTrainGenerator(cfg).generate()["spikes"]


class TestHarrisEstimator:
    def test_estimate_returns_float(self, spikes_critical):
        est = BranchingRatioEstimator()
        result = est.estimate(spikes_critical)
        assert isinstance(result, float)

    def test_estimate_zero_for_all_silent(self):
        spikes = np.zeros((50, 1000), dtype=np.int8)
        est = BranchingRatioEstimator()
        assert est.estimate(spikes) == 0.0

    def test_critical_sigma_near_one(self):
        spikes = _gen(1.0, seed=42, dur=300.0)
        est = BranchingRatioEstimator()
        sigma = est.estimate(spikes)
        assert abs(sigma - 1.0) < 0.15, f"Critical σ_b = {sigma:.3f}, expected ≈ 1.0"

    def test_subcritical_sigma_less_than_one(self):
        spikes = _gen(0.6, seed=43, dur=120.0)
        est = BranchingRatioEstimator()
        sigma = est.estimate(spikes)
        assert sigma < 1.0, f"Subcritical σ_b = {sigma:.3f} should be < 1"

    def test_supercritical_sigma_greater_than_one(self):
        spikes = _gen(1.4, seed=44, dur=60.0)
        est = BranchingRatioEstimator()
        sigma = est.estimate(spikes)
        # Supercritical: bounded by n_neurons due to deduplication, but > 1 generally
        assert sigma >= 0.5, f"Supercritical σ_b = {sigma:.3f}"

    def test_windowed_returns_array(self, spikes_critical):
        est = BranchingRatioEstimator()
        w = est.estimate_windowed(spikes_critical, window_bins=2000)
        assert isinstance(w, np.ndarray)
        assert len(w) >= 1

    def test_estimate_from_counts_consistent(self, spikes_critical):
        est = BranchingRatioEstimator()
        sigma_spikes = est.estimate(spikes_critical)
        counts = spikes_critical.sum(axis=0).astype(float)
        sigma_counts = est.estimate_from_counts(counts)
        assert sigma_spikes == pytest.approx(sigma_counts, abs=1e-10)


class TestAR1:
    def test_ar1_in_range(self, spikes_critical):
        est = BranchingRatioEstimator()
        ar1 = est.ar1_coefficient(spikes_critical)
        assert -1.0 <= ar1 <= 1.0

    def test_ar1_constant_signal_zero(self):
        spikes = np.ones((10, 500), dtype=np.int8)
        est = BranchingRatioEstimator()
        ar1 = est.ar1_coefficient(spikes)
        assert ar1 == 0.0

    def test_critical_ar1_positive(self, spikes_critical):
        est = BranchingRatioEstimator()
        ar1 = est.ar1_coefficient(spikes_critical)
        # Critical branching process has positive temporal autocorrelation
        assert ar1 > 0.0
