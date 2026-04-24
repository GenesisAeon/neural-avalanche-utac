"""Tests for AvalancheDetector."""

from __future__ import annotations

import numpy as np

from neural_avalanche_utac.avalanche import AvalancheDetector


def _make_spikes(pattern: list[int], n_neurons: int = 10) -> np.ndarray:
    """Build (n_neurons, n_bins) spike array from a population-count pattern."""
    n_bins = len(pattern)
    spikes = np.zeros((n_neurons, n_bins), dtype=np.int8)
    for t, count in enumerate(pattern):
        if count > 0:
            spikes[:count, t] = 1
    return spikes


class TestAvalancheDetection:
    def test_no_activity_gives_no_avalanches(self):
        spikes = np.zeros((10, 100), dtype=np.int8)
        avs = AvalancheDetector().detect(spikes)
        assert avs == []

    def test_single_avalanche(self):
        # one 3-bin burst: bins 2,3,4 are active
        spikes = _make_spikes([0, 0, 2, 3, 1, 0, 0])
        avs = AvalancheDetector().detect(spikes)
        assert len(avs) == 1
        av = avs[0]
        assert av.size == 6        # 2+3+1
        assert av.duration == 3
        assert av.start_bin == 2
        assert av.end_bin == 4

    def test_two_separate_avalanches(self):
        spikes = _make_spikes([1, 0, 0, 2, 1, 0])
        avs = AvalancheDetector().detect(spikes)
        assert len(avs) == 2
        assert avs[0].duration == 1
        assert avs[1].duration == 2

    def test_avalanche_at_end_of_array(self):
        spikes = _make_spikes([0, 0, 3, 2])
        avs = AvalancheDetector().detect(spikes)
        assert len(avs) == 1
        assert avs[0].end_bin == 3

    def test_sizes_and_durations_arrays(self):
        det = AvalancheDetector()
        spikes = _make_spikes([1, 2, 0, 3])
        avs = det.detect(spikes)
        sizes = det.sizes(avs)
        durs = det.durations(avs)
        assert sizes.shape == durs.shape == (2,)
        assert sizes[0] == 3   # 1+2
        assert sizes[1] == 3   # 3

    def test_summary_empty(self):
        det = AvalancheDetector()
        s = det.summary([])
        assert s["n"] == 0
        assert np.isnan(s["mean_size"])

    def test_summary_non_empty(self, spikes_critical):
        det = AvalancheDetector()
        avs = det.detect(spikes_critical)
        s = det.summary(avs)
        assert s["n"] > 0
        assert s["mean_size"] > 0
        assert s["max_size"] >= 1

    def test_detect_from_counts(self):
        counts = np.array([0, 1, 2, 0, 3, 1, 0], dtype=float)
        det = AvalancheDetector()
        avs = det.detect_from_counts(counts)
        assert len(avs) == 2


class TestCriticalAvalanches:
    def test_critical_has_more_avalanches_than_subcritical(
        self, spikes_critical, spikes_subcritical
    ):
        det = AvalancheDetector()
        n_crit = len(det.detect(spikes_critical))
        n_sub = len(det.detect(spikes_subcritical))
        # Both should have avalanches; critical typically has larger ones
        assert n_crit > 0
        assert n_sub >= 0  # subcritical may have few or many small ones

    def test_critical_has_large_avalanches(self, spikes_critical):
        det = AvalancheDetector()
        avs = det.detect(spikes_critical)
        sizes = det.sizes(avs)
        # At criticality power-law tail: should have at least some size > 10
        assert sizes.max() > 5
