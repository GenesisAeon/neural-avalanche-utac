"""Excitation-inhibition balance as CREP Γ modulator.

E-I balance w₀ = w_E − w_I determines whether the network sits at, above,
or below the critical point. Phase transition at w₀_critical (Phys Rev Lett
134, 028401; 2025). The distance |w₀ − w₀_c| modulates CREP coupling.

At w₀ ≈ w₀_c: Γ_effective ≈ Γ_CREP (full coupling)
Away from balance: Γ_effective < Γ_CREP (reduced criticality)
"""

from __future__ import annotations

import numpy as np


class EIBalanceMonitor:
    """
    Tracks excitation-inhibition balance and modulates CREP coupling.

    Phase transition condition from Phys Rev Lett 134, 028401 (2025):
        w₀_critical = β⁻¹ · α
    where β is the inverse temperature and α is the leakage rate.

    In practice, balanced state corresponds to equal mean firing rates
    of excitatory and inhibitory populations.
    """

    def __init__(
        self,
        w0_critical: float = 0.0,
        w0_scale: float = 0.5,
    ) -> None:
        self.w0_critical = w0_critical
        self.w0_scale = w0_scale
        self._history: list[dict] = []

    def compute_balance(self, w_E: float, w_I: float) -> dict:
        """
        Compute E-I balance metrics.

        Parameters
        ----------
        w_E : total excitatory drive (e.g. normalised mean excitatory rate)
        w_I : total inhibitory drive (e.g. normalised mean inhibitory rate)
        """
        w0 = w_E - w_I
        dist = abs(w0 - self.w0_critical)
        # Gaussian proximity to critical balance point
        balance_score = float(np.exp(-(dist**2) / (2.0 * self.w0_scale**2)))

        result = {
            "w_E": w_E,
            "w_I": w_I,
            "w0": w0,
            "dist_from_critical": dist,
            "balance_score": balance_score,
            "at_critical": dist < 0.1 * self.w0_scale,
        }
        self._history.append(result)
        return result

    def gamma_modulation(self, gamma_crep: float, w_E: float, w_I: float) -> float:
        """Apply E-I balance modulation: Γ_eff = Γ_CREP · balance_score."""
        bal = self.compute_balance(w_E, w_I)
        return float(gamma_crep * bal["balance_score"])

    def from_spike_populations(
        self,
        spikes_E: np.ndarray,
        spikes_I: np.ndarray,
    ) -> dict:
        """
        Estimate E-I balance from excitatory and inhibitory spike train arrays.

        Normalises rates so that perfect balance gives w_E = w_I = 0.5.
        """
        rate_E = float(spikes_E.mean())
        rate_I = float(spikes_I.mean())
        total = rate_E + rate_I + 1e-10
        w_E = rate_E / total
        w_I = rate_I / total
        return self.compute_balance(w_E, w_I)

    def from_branching_ratio(self, sigma_b: float) -> dict:
        """
        Map branching ratio to E-I balance proxy.

        At σ_b = 1 (criticality): w₀ ≈ w₀_critical (balanced).
        At σ_b > 1 (supercritical): excitation dominant (w₀ > w₀_c).
        At σ_b < 1 (subcritical): inhibition dominant (w₀ < w₀_c).
        """
        w0_proxy = (sigma_b - 1.0) * self.w0_scale  # 0 at criticality
        w_E = 0.5 + w0_proxy / 2.0
        w_I = 0.5 - w0_proxy / 2.0
        return self.compute_balance(float(np.clip(w_E, 0, 1)), float(np.clip(w_I, 0, 1)))

    @property
    def history(self) -> list[dict]:
        return list(self._history)
