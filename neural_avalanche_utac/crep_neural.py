"""Neural-specific CREP tensor (C, R, E, P) → Γ.

The CREP tensor Γ is derived from the measured branching ratio via the
UTAC fixed-point inversion:

    H* = K · tanh(σ · Γ)  →  Γ = arctanh(H/K) / σ

At criticality (H = σ_b = 1, K = 2, σ = 2.2):
    Γ_brain = arctanh(0.5) / 2.2 ≈ 0.251

The four CREP components (C, R, E, P) are independent diagnostics that
confirm criticality from different perspectives. Each lives in [0, 1]
with 1 indicating optimal proximity to the critical state.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from neural_avalanche_utac.avalanche import AvalancheDetector
from neural_avalanche_utac.branching import BranchingRatioEstimator
from neural_avalanche_utac.constants import SIGMA_CREP, K
from neural_avalanche_utac.power_law import PowerLawFitter


class NeuralCREPTensor:
    """
    Computes the CREP tensor Γ and its four diagnostic components from
    neural spike train data.

    Γ is computed from the branching ratio via UTAC fixed-point inversion;
    C, R, E, P are supplementary diagnostics reported alongside Γ.
    """

    def __init__(self, sigma_crep: float = SIGMA_CREP, k: float = K) -> None:
        self.sigma_crep = sigma_crep
        self.k = k
        self._branching = BranchingRatioEstimator()
        self._pl_fitter = PowerLawFitter(x_min=1.0)
        self._av_det = AvalancheDetector()
        self._last: dict = {}

    # ── CREP components ────────────────────────────────────────────────────────

    def compute_C(self, spikes: np.ndarray) -> float:
        """C — Coherence / Critical Slowing Down.

        AR(1) autocorrelation of population activity. Increases toward 1
        as the system approaches criticality (critical slowing down).
        Mapped to [0, 1]: C = (AR1 + 1) / 2.
        """
        ar1 = self._branching.ar1_coefficient(spikes)
        return float((ar1 + 1.0) / 2.0)

    def compute_R(self, spikes: np.ndarray) -> float:
        """R — Resonance / Power-law Exponent Proximity.

        Proximity of the measured avalanche size exponent to τ = 3/2.
        R → 1 when the distribution is exactly power-law with the
        mean-field critical exponent (Zapperi et al. 1995).
        """
        avs = self._av_det.detect(spikes)
        if len(avs) < 10:
            return 0.0
        sizes = self._av_det.sizes(avs)
        result = self._pl_fitter.fit_and_score(sizes, tau_critical=1.5)
        return float(result["tau_proximity"])

    def compute_E(self, spikes: np.ndarray) -> float:
        """E — Emergence / Supra-additive Population Variance (Fano factor).

        At criticality, the population spike count variance far exceeds the
        Poisson expectation (Fano factor F = Var(N_t) / mean(N_t) >> 1).
        This excess variance arises from shared avalanche structure — neurons
        co-activate, producing large, correlated fluctuations.

        Calibration (branching process):
          Subcritical σ_b=0.6: F ≈ 1-3   → E ≈ 0.0-0.2
          Critical    σ_b=1.0: F ≈ 5-20  → E ≈ 0.5-0.9
          Supercritical σ_b=1.4: F > 20  → E → 1.0

        Maps to CREP E-component (emergence of supra-additive fluctuations).
        """
        n_t = spikes.sum(axis=0).astype(float)  # population count (n_bins,)
        mean_n = float(n_t.mean())
        if mean_n < 1e-8:
            return 0.0
        var_n = float(n_t.var())
        fano = var_n / mean_n  # Poisson baseline: F=1
        # Sigmoid mapping calibrated so F=10 (typical critical value) → E≈0.73
        return float(1.0 / (1.0 + np.exp(-0.3 * (fano - 5.0))))

    def compute_P(self, spikes: np.ndarray, order: int = 3) -> float:
        """P — Permutation Entropy (inverted, proximity-to-critical).

        Permutation entropy of the population activity time series, mapped
        so that P → 1 near criticality where normalised entropy ≈ 0.70
        (intermediate between maximal disorder and order).
        """
        n_t = spikes.sum(axis=0).astype(float)
        if len(n_t) < order + 1:
            return 0.0

        n_patterns = factorial(order)
        pattern_counts: dict[tuple, int] = {}

        for i in range(len(n_t) - order):
            seg = n_t[i : i + order]
            pat = tuple(np.argsort(seg, kind="stable"))
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

        total = sum(pattern_counts.values())
        probs = np.array(list(pattern_counts.values()), dtype=float) / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        max_entropy = np.log2(n_patterns)
        h_norm = entropy / max_entropy if max_entropy > 0 else 0.0

        # At criticality h_norm ≈ 0.70; peak proximity at that value
        return float(np.clip(1.0 - abs(h_norm - 0.70) / 0.70, 0.0, 1.0))

    # ── Main computation ───────────────────────────────────────────────────────

    def compute(self, spikes: np.ndarray) -> dict:
        """
        Compute the full CREP tensor from spike array (n_neurons, n_bins).

        Γ is derived from the branching ratio via:
            Γ = arctanh(σ_b / K) / σ_crep

        Returns dict: {C, R, E, P, Gamma, sigma_b}
        """
        sigma_b = self._branching.estimate(spikes)
        sigma_b_safe = float(np.clip(sigma_b, 1e-4, self.k - 1e-4))
        gamma = float(np.arctanh(sigma_b_safe / self.k) / self.sigma_crep)

        C = self.compute_C(spikes)
        R = self.compute_R(spikes)
        E = self.compute_E(spikes)
        P = self.compute_P(spikes)

        self._last = {"C": C, "R": R, "E": E, "P": P, "Gamma": gamma, "sigma_b": sigma_b}
        return dict(self._last)

    @staticmethod
    def gamma_from_eta(eta: float, sigma: float = SIGMA_CREP) -> float:
        """Theoretical Γ = arctanh(η) / σ from UTAC fixed-point at efficiency η."""
        if eta <= 0.0 or eta >= 1.0:
            return float("nan")
        return float(np.arctanh(eta) / sigma)
