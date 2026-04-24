"""Power-law fitting for avalanche size and duration distributions.

Uses the maximum-likelihood estimator (Clauset et al. 2009) with a KS
goodness-of-fit test. Proximity to the critical exponent τ = 3/2 (size)
or α = 2 (duration) maps to the CREP R-component.
"""

from __future__ import annotations

import numpy as np


class PowerLawFitter:
    """
    Fits P(x) ~ x^(-τ) via MLE for x ≥ x_min.

    For discrete positive-integer data (avalanche sizes), uses the
    Clauset et al. (2009) discrete approximation:
        τ̂ = 1 + n · [Σ ln(x_i / (x_min - 0.5))]^(-1)

    Reference: Clauset, A., Shalizi, C.R. & Newman, M.E.J. (2009).
               SIAM Review 51(4), 661-703.
    """

    def __init__(self, x_min: float = 1.0) -> None:
        self.x_min = x_min

    def fit_mle(self, data: np.ndarray) -> dict:
        """
        Fit power-law exponent τ via MLE.

        Parameters
        ----------
        data : 1-D array of positive values

        Returns
        -------
        dict with keys: tau, x_min, n_tail, log_likelihood
        """
        x = data[data >= self.x_min]
        n = len(x)
        if n < 2:
            return {"tau": float("nan"), "x_min": self.x_min, "n_tail": n, "log_likelihood": float("nan")}

        # Discrete Hill MLE estimator
        log_ratio = np.log(x / (self.x_min - 0.5))
        tau = float(1.0 + n / log_ratio.sum())

        # Log-likelihood
        ll = float(n * np.log(tau - 1) - n * np.log(self.x_min) - tau * log_ratio.sum())

        return {"tau": tau, "x_min": self.x_min, "n_tail": n, "log_likelihood": ll}

    def ks_distance(self, data: np.ndarray, tau: float) -> dict:
        """
        Kolmogorov-Smirnov distance between empirical CDF and fitted power-law.

        Lower D = better fit. p_value is approximate (Kolmogorov distribution).
        """
        x = np.sort(data[data >= self.x_min])
        n = len(x)
        if n < 2:
            return {"D": 1.0, "p_value": 0.0}

        empirical = np.arange(1, n + 1) / n
        theoretical = 1.0 - (x / self.x_min) ** (-(tau - 1.0))
        theoretical = np.clip(theoretical, 0.0, 1.0)

        D = float(np.max(np.abs(empirical - theoretical)))
        p_value = float(np.exp(-2.0 * n * D**2))
        return {"D": D, "p_value": p_value}

    def tau_proximity(self, tau_measured: float, tau_critical: float = 1.5, width: float = 0.15) -> float:
        """
        Gaussian proximity of measured exponent to the critical value.

        Returns a score in [0, 1]:
          1.0 → tau_measured = tau_critical (exact criticality)
          0.0 → far from critical exponent

        Maps to the CREP R-component.

        Parameters
        ----------
        tau_measured  : estimated power-law exponent
        tau_critical  : expected exponent at criticality (3/2 for sizes, 2 for durations)
        width         : Gaussian width (tolerance); default 0.15
        """
        if np.isnan(tau_measured):
            return 0.0
        dist = abs(tau_measured - tau_critical)
        return float(np.exp(-(dist**2) / (2.0 * width**2)))

    def fit_and_score(
        self,
        data: np.ndarray,
        tau_critical: float = 1.5,
    ) -> dict:
        """Convenience: fit + KS test + proximity score in one call."""
        fit = self.fit_mle(data)
        tau = fit["tau"]
        ks = self.ks_distance(data, tau) if not np.isnan(tau) else {"D": 1.0, "p_value": 0.0}
        proximity = self.tau_proximity(tau, tau_critical)
        return {**fit, **ks, "tau_proximity": proximity, "tau_critical": tau_critical}
