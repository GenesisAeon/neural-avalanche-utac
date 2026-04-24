"""NeuralAvalancheUTAC — Diamond-Template implementation for Package 20.

Diamond-Template contract (mandatory):
  run_cycle()        → dict
  get_crep_state()   → {C, R, E, P, Gamma}
  get_utac_state()   → {H, dH_dt, H_star, K_eff}
  get_phase_events() → list[dict]
  to_zenodo_record() → dict

Core imports stub (genesis-os not required):
  GENESIS_OS_AVAILABLE flag signals whether the live library is present.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neural_avalanche_utac.avalanche import AvalancheDetector
from neural_avalanche_utac.branching import BranchingRatioEstimator
from neural_avalanche_utac.constants import (
    GAMMA_AMOC,
    GAMMA_BRAIN,
    H_STAR,
    PACKAGE_REGISTRY_ENTRY,
    R_HOMEOSTATIC,
    SEED,
    SIGMA_CREP,
    K,
)
from neural_avalanche_utac.crep_neural import NeuralCREPTensor
from neural_avalanche_utac.ei_balance import EIBalanceMonitor
from neural_avalanche_utac.homeostasis import HomeostaticPlasticity
from neural_avalanche_utac.spike_train import SpikeTrainConfig, SpikeTrainGenerator

# ── genesis-os stub ────────────────────────────────────────────────────────────
GENESIS_OS_AVAILABLE = importlib.util.find_spec("genesis") is not None


# ── Ethics-Gate Light (Phase H) ───────────────────────────────────────────────

class EthicsGate:
    """
    Ethics-Gate Light (Phase H).

    Blocks simulation steps when the system tension exceeds a safety
    threshold. Tension combines CREP Γ with normalised deviation from
    the homeostatic setpoint — high tension indicates an unstable or
    runaway configuration.
    """

    TENSION_THRESHOLD: float = 0.85

    def check(self, state: dict[str, Any], tension: float) -> dict[str, Any]:
        """
        Returns {"allowed": True} or {"allowed": False, "reason": str}.
        """
        if tension > self.TENSION_THRESHOLD:
            return {
                "allowed": False,
                "reason": (
                    f"tension={tension:.4f} exceeds threshold={self.TENSION_THRESHOLD:.2f}; "
                    "system in unsafe regime — halt to preserve data integrity."
                ),
            }
        return {"allowed": True, "reason": "ok"}


class TensionMetric:
    """Computes system tension from CREP state + UTAC deviation."""

    def __init__(self) -> None:
        self._current: float = 0.0

    def update(self, gamma: float, h: float, h_star: float) -> None:
        deviation = abs(h - h_star) / max(h_star, 1e-8)
        # Tension = CREP level + fractional deviation, capped at 1
        self._current = float(np.clip(gamma * 2.0 + deviation * 0.15, 0.0, 1.0))

    def get_current_tension(self) -> float:
        return self._current


# ── UTAC state container ───────────────────────────────────────────────────────

@dataclass
class UTACState:
    H: float = 1.0          # branching ratio σ_b (UTAC state variable)
    dH_dt: float = 0.0      # instantaneous rate of change
    H_star: float = H_STAR  # current homeostatic target (≈ K · tanh(σ · Γ))
    K_eff: float = K        # effective supercritical ceiling


# ── Main class ─────────────────────────────────────────────────────────────────

class NeuralAvalancheUTAC:
    """
    GenesisAeon Package 20 — Brain Criticality & Neuronal Avalanche Threshold.

    Models the brain's self-organised critical state as a UTAC dynamical system.

    UTAC mapping:
      H(t)  ← branching ratio σ_b ∈ [0, 2]
      K     ← 2.0  (supercritical ceiling)
      H*    ← 1.0  (critical setpoint, σ_b = 1 at criticality)
      r     ← 0.15 (homeostatic rate per hour, Hengen lab)
      σ     ← 2.2  (CREP coupling, GenesisAeon default)
      Γ(t)  ← arctanh(σ_b / K) / σ  → 0.251 at criticality

    Central result: Γ_brain = Γ_AMOC = 0.251
      Both AMOC ocean circulation and neural criticality converge to the
      same CREP value at the η = 50% homeostatic setpoint — cross-domain
      UTAC universality.
    """

    def __init__(
        self,
        n_neurons: int = 500,
        seed: int = SEED,
        r: float = R_HOMEOSTATIC,
        sigma_crep: float = SIGMA_CREP,
        segment_s: float = 60.0,
    ) -> None:
        self.n_neurons = n_neurons
        self.seed = seed
        self.r = r
        self.sigma_crep = sigma_crep
        self.segment_s = segment_s

        self._rng = np.random.default_rng(seed)

        # Sub-components
        self._crep = NeuralCREPTensor(sigma_crep=sigma_crep, k=K)
        self._homeostasis = HomeostaticPlasticity(r=r, target_sigma=H_STAR)
        self._ei_monitor = EIBalanceMonitor()
        self._branching_est = BranchingRatioEstimator()
        self._av_det = AvalancheDetector()

        # Ethics-Gate Light (Phase H)
        self._ethics_gate = EthicsGate()
        self._tension_metric = TensionMetric()

        # State
        self._utac: UTACState = UTACState()
        self._crep_state: dict[str, Any] = {}
        self._phase_events: list[dict[str, Any]] = []
        self._cycle_log: list[dict[str, Any]] = []

    # ── Internal UTAC ODE ─────────────────────────────────────────────────────

    def _utac_step(self, H: float, gamma: float, dt_hours: float = 1.0) -> tuple[float, float]:
        """
        Euler step of the UTAC relaxation ODE.

        dH/dt = r · (H_target − H)
        H_target = K · tanh(σ · Γ)

        At Γ = 0.251: H_target = 2 · tanh(2.2 · 0.251) ≈ 1.006 ≈ H* = 1.0
        """
        H_target = K * float(np.tanh(self.sigma_crep * gamma))
        dH_dt = self.r * (H_target - H)
        H_new = float(np.clip(H + dH_dt * dt_hours, 1e-3, K - 1e-3))
        return H_new, dH_dt

    def _generate_segment(self, sigma_b: float) -> np.ndarray:
        """Generate one time segment of spike data at a given branching ratio."""
        cfg = SpikeTrainConfig(
            n_neurons=self.n_neurons,
            duration_s=self.segment_s,
            dt_ms=2.0,
            branching_ratio=float(np.clip(sigma_b, 0.05, 1.95)),
            seed=int(self._rng.integers(0, 2**31)),
        )
        return np.asarray(SpikeTrainGenerator(cfg).generate()["spikes"])

    # ── Diamond interface ─────────────────────────────────────────────────────

    def run_cycle(self, duration_seconds: float = 3600.0) -> dict[str, Any]:
        """
        Run one full UTAC cycle and return a comprehensive state dictionary.

        The simulation proceeds in segment_s-second chunks. Each chunk:
          1. Generates a spike train at the current branching ratio H.
          2. Computes CREP tensor Γ from the spike data.
          3. Updates tension metric and checks the Ethics-Gate.
          4. Detects avalanches; records phase events.
          5. Steps the UTAC ODE (H → H_new).
          6. Applies homeostatic correction.
        """
        n_steps = max(1, int(duration_seconds / self.segment_s))
        dt_h = self.segment_s / 3600.0  # segment duration in hours

        H = self._utac.H
        sigma_history: list[float] = []
        gamma_history: list[float] = []
        all_av_count = 0

        for step in range(n_steps):
            # 1. Generate spikes
            spikes = self._generate_segment(H)

            # 2. CREP tensor
            crep_out = self._crep.compute(spikes)
            gamma = crep_out["Gamma"]
            sigma_meas = crep_out["sigma_b"]
            gamma_history.append(gamma)
            sigma_history.append(sigma_meas)

            # 3. Tension + Ethics-Gate Light (Phase H)
            self._tension_metric.update(gamma, H, H_STAR)
            tension_val = float(self._tension_metric.get_current_tension())
            state_snap: dict[str, Any] = {"H": H, "step": step, "gamma": gamma}
            gate = self._ethics_gate.check(state=state_snap, tension=tension_val)
            if not gate["allowed"]:
                raise RuntimeError(f"EthicsGate blocked: {gate['reason']}")

            # 4. Avalanches / phase events
            avs = self._av_det.detect(spikes)
            all_av_count += len(avs)
            if avs:
                self._phase_events.append({
                    "step": step,
                    "t_s": step * self.segment_s,
                    "n_avalanches": len(avs),
                    "max_size": int(max(a.size for a in avs)),
                    "sigma_b": H,
                    "gamma": gamma,
                    "tension": tension_val,
                })

            # 5. UTAC ODE step
            H_new, dH_dt = self._utac_step(H, gamma, dt_hours=dt_h)
            self._utac.dH_dt = dH_dt
            self._utac.H_star = K * float(np.tanh(self.sigma_crep * gamma))

            # 6. Homeostatic correction (partial, proportional to segment)
            H = self._homeostasis.update(H_new, dt_hours=dt_h)
            H = float(np.clip(H, 1e-3, K - 1e-3))

        self._utac.H = H
        self._crep_state = crep_out  # last segment's CREP state

        result: dict[str, Any] = {
            "H_final": H,
            "sigma_b_mean": float(np.mean(sigma_history)),
            "gamma_mean": float(np.mean(gamma_history)),
            "n_avalanches_total": all_av_count,
            "n_phase_events": len(self._phase_events),
            "crep_state": self.get_crep_state(),
            "utac_state": self.get_utac_state(),
            "duration_seconds": duration_seconds,
            "n_steps": n_steps,
            "is_critical": self.is_critical(),
            "genesis_os_available": GENESIS_OS_AVAILABLE,
        }
        self._cycle_log.append(result)
        return result

    def get_crep_state(self) -> dict[str, Any]:
        """Return CREP tensor components {C, R, E, P, Gamma}."""
        if not self._crep_state:
            return {"C": 0.0, "R": 0.0, "E": 0.0, "P": 0.0, "Gamma": 0.0, "sigma_b": 0.0}
        return dict(self._crep_state)

    def get_utac_state(self) -> dict[str, Any]:
        """Return UTAC state variables {H, dH_dt, H_star, K_eff}."""
        return {
            "H": float(self._utac.H),
            "dH_dt": float(self._utac.dH_dt),
            "H_star": float(self._utac.H_star),
            "K_eff": float(self._utac.K_eff),
        }

    def get_phase_events(self) -> list[dict[str, Any]]:
        """Return list of phase transition events (avalanche cluster records)."""
        return list(self._phase_events)

    def to_zenodo_record(self) -> dict[str, Any]:
        """
        Generate a Zenodo-compatible metadata record.

        Includes UTAC state, CREP state, gamma universality result, and
        package registry information.

        Ethics-Gate Light (Phase H): blocked if current tension is unsafe.
        """
        state = self.get_utac_state()
        crep = self.get_crep_state()

        # Ethics-Gate Light (Phase H)
        tension = getattr(self, "_tension_metric", None)
        if tension is not None:
            tension_value = float(tension.get_current_tension())
            ethics_result = self._ethics_gate.check(state=state, tension=tension_value)
            if not ethics_result["allowed"]:
                raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")

        gamma_measured = crep.get("Gamma", GAMMA_BRAIN)
        universality = abs(gamma_measured - GAMMA_AMOC) < 0.05

        return {
            "title": "NeuralAvalancheUTAC — Brain Criticality as UTAC Phase Transition",
            "description": (
                "GenesisAeon Package 20. Models neuronal avalanches in the brain at "
                "criticality as a UTAC dynamical system. "
                f"Key result: Γ_brain = {gamma_measured:.3f} ≈ Γ_AMOC = {GAMMA_AMOC:.3f}. "
                "Cross-domain CREP universality at η = 50% homeostatic setpoint."
            ),
            "creators": [{"name": "Römer, Johann", "affiliation": "MOR Research Collective"}],
            "keywords": [
                "neuronal avalanches", "brain criticality", "UTAC", "CREP tensor",
                "self-organised criticality", "branching process", "GenesisAeon",
            ],
            "license": "MIT",
            "upload_type": "software",
            "doi": PACKAGE_REGISTRY_ENTRY["zenodo"],
            "related_identifiers": [
                {"relation": "references", "identifier": "10.1016/j.neuron.2025.05.020"},
                {"relation": "references", "identifier": "10.1103/PhysRevLett.134.028401"},
                {"relation": "references", "identifier": "10.3389/fncom.2025.1560691"},
            ],
            "utac_state": state,
            "crep_state": crep,
            "gamma_brain": gamma_measured,
            "gamma_amoc": GAMMA_AMOC,
            "gamma_universality_confirmed": universality,
            "n_phase_events": len(self._phase_events),
            "genesis_os_available": GENESIS_OS_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "package_registry": PACKAGE_REGISTRY_ENTRY,
        }

    # ── Scientific convenience methods ────────────────────────────────────────

    def is_critical(self) -> bool:
        """True if current branching ratio is within 5% of σ_b = 1."""
        return abs(self._utac.H - H_STAR) / H_STAR < 0.05

    def gamma_universality_check(self) -> dict[str, Any]:
        """
        THE KEY RESULT: verify Γ_brain ≈ Γ_AMOC ≈ 0.251.

        Both ocean circulation (Package 18) and neural criticality (Package 20)
        converge to the same CREP value at the η = 50% homeostatic setpoint.
        This is the cross-domain UTAC universality prediction.
        """
        crep = self.get_crep_state()
        gamma_measured = crep.get("Gamma", 0.0)
        gamma_theoretical = float(np.arctanh(0.5) / self.sigma_crep)

        match = abs(gamma_measured - GAMMA_AMOC) < 0.05

        return {
            "gamma_brain_measured": gamma_measured,
            "gamma_brain_theoretical": gamma_theoretical,
            "gamma_amoc": GAMMA_AMOC,
            "universality_confirmed": match,
            "eta_setpoint": 0.50,
            "interpretation": (
                "Cross-domain CREP universality confirmed: both AMOC ocean circulation "
                "and neural criticality converge to Γ ≈ 0.251 at η = 50%."
                if match else
                f"Γ_brain = {gamma_measured:.3f} differs from Γ_AMOC = {GAMMA_AMOC:.3f} "
                f"(Δ = {abs(gamma_measured - GAMMA_AMOC):.3f}). "
                "Run run_cycle() first to generate CREP measurements."
            ),
        }
