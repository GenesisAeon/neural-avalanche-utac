"""Benchmark validation for Package 20 against Hengen & Shew 2025 targets.

Runs the full pipeline on deterministic synthetic data (seed=42) and checks
each NEURAL_TARGETS entry. Returns a structured pass/fail report.
"""

from __future__ import annotations

import numpy as np

from neural_avalanche_utac.avalanche import AvalancheDetector
from neural_avalanche_utac.branching import BranchingRatioEstimator
from neural_avalanche_utac.constants import GAMMA_AMOC, NEURAL_TARGETS
from neural_avalanche_utac.crep_neural import NeuralCREPTensor
from neural_avalanche_utac.power_law import PowerLawFitter
from neural_avalanche_utac.spike_train import SpikeTrainConfig, SpikeTrainGenerator


def _gen_critical(seed: int = 42, n_neurons: int = 300, duration_s: float = 300.0) -> np.ndarray:
    """Generate a critical (σ_b = 1) spike train for benchmarking."""
    cfg = SpikeTrainConfig(
        n_neurons=n_neurons,
        duration_s=duration_s,
        dt_ms=2.0,
        branching_ratio=1.0,
        seed=seed,
    )
    return SpikeTrainGenerator(cfg).generate()["spikes"]


def run_benchmarks(
    seed: int = 42,
    verbose: bool = True,
    n_neurons: int = 300,
    duration_s: float = 300.0,
) -> dict:
    """
    Run all benchmark validations for Package 20.

    Parameters
    ----------
    seed       : RNG seed (42 for standard GenesisAeon validation)
    verbose    : print a formatted pass/fail report
    n_neurons  : neurons for synthetic spike train
    duration_s : duration of synthetic spike train in seconds

    Returns
    -------
    dict mapping benchmark name → {measured, target, tolerance, pass}
    """
    spikes = _gen_critical(seed=seed, n_neurons=n_neurons, duration_s=duration_s)
    results: dict = {}

    # ── Benchmark 1: branching ratio ──────────────────────────────────────────
    est = BranchingRatioEstimator()
    sigma_b = est.estimate(spikes)
    t_val, t_tol = NEURAL_TARGETS["branching_ratio_critical"]
    results["branching_ratio_critical"] = {
        "measured": round(sigma_b, 4),
        "target": t_val,
        "tolerance_abs": t_tol * t_val,
        "pass": abs(sigma_b - t_val) <= t_tol * t_val,
    }

    # ── Benchmarks 2 & 3: power-law exponents ─────────────────────────────────
    det = AvalancheDetector()
    fitter = PowerLawFitter(x_min=1.0)
    avs = det.detect(spikes)

    if len(avs) >= 10:
        sizes = det.sizes(avs)
        durs = det.durations(avs)
        tau_fit = fitter.fit_mle(sizes)["tau"]
        alpha_fit = fitter.fit_mle(durs)["tau"]

        t_tau, tol_tau = NEURAL_TARGETS["power_law_tau"]
        t_alpha, tol_alpha = NEURAL_TARGETS["power_law_alpha"]

        results["power_law_tau"] = {
            "measured": round(tau_fit, 4),
            "target": t_tau,
            "tolerance_abs": tol_tau,
            "n_avalanches": len(avs),
            "pass": abs(tau_fit - t_tau) <= tol_tau,
        }
        results["power_law_alpha"] = {
            "measured": round(alpha_fit, 4),
            "target": t_alpha,
            "tolerance_abs": tol_alpha,
            "n_avalanches": len(avs),
            "pass": abs(alpha_fit - t_alpha) <= tol_alpha,
        }
    else:
        for key in ("power_law_tau", "power_law_alpha"):
            results[key] = {"measured": None, "pass": False,
                            "reason": f"too few avalanches ({len(avs)})"}

    # ── Benchmark 4: Γ_brain ──────────────────────────────────────────────────
    crep = NeuralCREPTensor()
    crep_out = crep.compute(spikes)
    gamma = crep_out["Gamma"]

    t_gamma, tol_gamma = NEURAL_TARGETS["gamma_brain"]
    results["gamma_brain"] = {
        "measured": round(gamma, 4),
        "target": t_gamma,
        "tolerance_abs": tol_gamma,
        "theoretical": round(float(np.arctanh(0.5) / 2.2), 4),
        "pass": abs(gamma - t_gamma) <= tol_gamma,
    }

    # ── Benchmark 5: Γ universality (Γ_brain ≈ Γ_AMOC) ───────────────────────
    diff = abs(gamma - GAMMA_AMOC)
    results["gamma_universality_match"] = {
        "gamma_brain": round(gamma, 4),
        "gamma_amoc": GAMMA_AMOC,
        "difference": round(diff, 4),
        "pass": diff < 0.05,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for v in results.values() if v.get("pass", False))
    n_total = len(results)
    results["_summary"] = {
        "passed": n_pass,
        "total": n_total,
        "pass_rate": round(n_pass / n_total, 2),
    }

    if verbose:
        _print_report(results)

    return results


def _print_report(results: dict) -> None:
    summary = results.get("_summary", {})
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  NeuralAvalancheUTAC — Benchmark Report  (GenesisAeon P20)  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    for key, val in results.items():
        if key.startswith("_"):
            continue
        status = "PASS ✓" if val.get("pass") else "FAIL ✗"
        m = val.get("measured", "N/A")
        t = val.get("target", val.get("gamma_amoc", "N/A"))
        print(f"  [{status}] {key:<32s} measured={m}  target={t}")
    print(f"\n  Summary: {summary.get('passed')}/{summary.get('total')} passed "
          f"({100 * summary.get('pass_rate', 0):.0f}%)\n")
