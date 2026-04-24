# neural-avalanche-utac

**GenesisAeon Package 20 — Brain Criticality & Neuronal Avalanche Threshold**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19645351-blue)](https://doi.org/10.5281/zenodo.19645351)

Models **neuronal avalanches in the brain at criticality** as a UTAC (Universal
Threshold Attractor Cascade) dynamical system. Implements the Diamond-Template
contract from the GenesisAeon cross-domain criticality framework.

---

## Central Scientific Result

```
Γ_brain = arctanh(H*/K) / σ = arctanh(0.5) / 2.2 ≈ 0.251

Γ_brain = Γ_AMOC = 0.251
```

Both the AMOC ocean circulation (Package 18) and neural criticality (Package 20)
converge to **Γ ≈ 0.251** at the η = 50% homeostatic setpoint — **cross-domain CREP universality**.

---

## UTAC Mapping

| Symbol | Neural interpretation |
|--------|----------------------|
| H(t)   | Network branching ratio σ_b ∈ [0, 2] |
| K      | 2.0 (supercritical ceiling) |
| H\*    | 1.0 (critical setpoint, σ_b = 1) |
| r      | 0.15/hour (homeostatic plasticity rate, Hengen lab) |
| σ      | 2.2 (CREP coupling, GenesisAeon default) |
| Γ(t)   | arctanh(σ_b / K) / σ → 0.251 at criticality |

---

## Install

```bash
pip install neural-avalanche-utac
# or with uv
uv pip install neural-avalanche-utac
```

For development:

```bash
git clone https://github.com/GenesisAeon/neural-avalanche-utac.git
cd neural-avalanche-utac
uv sync --dev
uv run pytest
```

---

## Quick Start

```python
from neural_avalanche_utac import NeuralAvalancheUTAC

# Instantiate and run
model = NeuralAvalancheUTAC(n_neurons=500, seed=42)
result = model.run_cycle(duration_seconds=3600.0)

# Diamond-Template interface
crep  = model.get_crep_state()   # {C, R, E, P, Gamma}
utac  = model.get_utac_state()   # {H, dH_dt, H_star, K_eff}
evts  = model.get_phase_events() # list of avalanche records
rec   = model.to_zenodo_record() # Zenodo-compatible metadata

# Cross-domain universality check
univ = model.gamma_universality_check()
print(univ['interpretation'])
```

---

## CLI

```bash
# Run a simulation
neural-utac run --duration 3600 --neurons 500

# Check a spike train file for criticality
neural-utac criticality-check spikes.npz --key spikes_critical

# Cross-domain Γ universality check
neural-utac gamma-universality --compare amoc

# Full benchmark suite
neural-utac benchmark

# Version
neural-utac version
```

---

## Repository Structure

```
neural-avalanche-utac/
├── neural_avalanche_utac/
│   ├── __init__.py
│   ├── system.py          # NeuralAvalancheUTAC — Diamond interface
│   ├── spike_train.py     # Branching process spike train generator
│   ├── avalanche.py       # Beggs & Plenz (2003) avalanche detection
│   ├── branching.py       # Harris (1963) branching ratio estimator
│   ├── power_law.py       # MLE power-law fitting (Clauset et al. 2009)
│   ├── crep_neural.py     # Neural CREP tensor (C, R, E, P → Γ)
│   ├── homeostasis.py     # Homeostatic plasticity → UTAC r parameter
│   ├── ei_balance.py      # E-I balance as CREP Γ modulator
│   ├── benchmark.py       # Validation vs. Hengen & Shew 2025
│   ├── cli.py             # Typer CLI
│   └── constants.py       # Physical constants + benchmark targets
├── notebooks/
│   ├── 01_neural_utac_overview.ipynb
│   ├── 02_avalanche_detection.ipynb
│   ├── 03_crep_brain_criticality.ipynb
│   └── 04_gamma_brain_universality.ipynb   # The Γ≈0.251 cross-domain result
├── data/
│   └── hengen2025_targets.yaml
└── tests/
    ├── conftest.py
    ├── test_diamond_interface.py
    ├── test_avalanche.py
    ├── test_branching.py
    └── test_power_law.py
```

---

## Benchmark Targets

| Target | Value | Tolerance | Source |
|--------|-------|-----------|--------|
| σ_b at criticality | 1.00 | ±5% | Beggs & Plenz 2003 |
| Power-law τ (size) | 1.50 | ±0.05 | Zapperi et al. 1995 |
| Power-law α (duration) | 2.00 | ±0.05 | Zapperi et al. 1995 |
| Γ_brain | 0.251 | ±0.03 | UTAC fixed-point inversion |
| Γ_brain ≈ Γ_AMOC | True | Δ < 0.05 | Cross-domain universality |

---

## Ethics-Gate Light (Phase H)

`system.py` and `to_zenodo_record()` embed an **EthicsGate** that monitors
system tension. If tension exceeds 0.85 (indicating an unstable or runaway
configuration), the gate raises `RuntimeError` before any data export:

```python
# Ethics-Gate Light (Phase H)
tension = getattr(self, '_tension_metric', None)
if tension is not None:
    tension_value = float(tension.get_current_tension())
    ethics_result = self._ethics_gate.check(state=state, tension=tension_value)
    if not ethics_result["allowed"]:
        raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")
```

---

## References

- Hengen, K.B. & Shew, W.L. (2025). "Is criticality a unified setpoint of brain function?" *Neuron* 113(16), 2582–2598. DOI: [10.1016/j.neuron.2025.05.020](https://doi.org/10.1016/j.neuron.2025.05.020)
- Sugimoto, Y. et al. (2025). "Network structure influences self-organized criticality in neural networks with dynamical synapses." *Frontiers in Computational Neuroscience*. DOI: [10.3389/fncom.2025.1560691](https://doi.org/10.3389/fncom.2025.1560691)
- *Phys. Rev. Lett.* 134, 028401 (2025). "Critical Avalanches in E-I Balanced Networks." DOI: [10.1103/PhysRevLett.134.028401](https://doi.org/10.1103/PhysRevLett.134.028401)
- Beggs, J.M. & Plenz, D. (2003). *J. Neurosci.* 23(35), 11167–11177.
- Clauset, A., Shalizi, C.R. & Newman, M.E.J. (2009). *SIAM Review* 51(4), 661–703.

## CREP Criticality Spectrum

| Domain | Package | Γ | η |
|--------|---------|---|---|
| Solar Flare | P21 | 0.014 | 1% |
| Cygnus X-1 Jet | P17 | 0.046 | 5% |
| Amazon Rainforest | P19 | 0.116 | 12% |
| **AMOC Ocean** | **P18** | **0.251** | **50%** |
| **Neural Criticality** | **P20** | **0.251** | **50%** |
| BTW Sandpile | P22 | 0.296 | 58% |
| Manna Sandpile | P22 | 0.376 | 72% |
| ERA5 Arctic | P01 | 0.920 | 99% |

---

MIT License · seed=42 · GenesisAeon Cycle 3 · MOR Research Collective
