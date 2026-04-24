# neural-avalanche-utac

**GenesisAeon Package 20 — Brain Criticality & Neuronal Avalanche Threshold**

Models neuronal avalanches in the brain at criticality as a UTAC dynamical system.

## Central Result

$$\Gamma_\text{brain} = \frac{\text{arctanh}(H^*/K)}{\sigma} = \frac{\text{arctanh}(0.5)}{2.2} \approx 0.251 = \Gamma_\text{AMOC}$$

Cross-domain CREP universality: both AMOC ocean circulation (Package 18) and
neural criticality (Package 20) converge to **Γ ≈ 0.251** at the η = 50%
homeostatic setpoint.

## Quickstart

```bash
pip install neural-avalanche-utac
```

```python
from neural_avalanche_utac import NeuralAvalancheUTAC

model = NeuralAvalancheUTAC(n_neurons=500, seed=42)
result = model.run_cycle(duration_seconds=3600.0)

crep = model.get_crep_state()   # {C, R, E, P, Gamma}
utac = model.get_utac_state()   # {H, dH_dt, H_star, K_eff}
univ = model.gamma_universality_check()
print(univ["interpretation"])
```

## CLI

```bash
neural-utac run --duration 3600 --neurons 500
neural-utac criticality-check spikes.npz
neural-utac gamma-universality
neural-utac benchmark
```

## UTAC Mapping

| Symbol | Neural interpretation |
|--------|-----------------------|
| H(t)   | Branching ratio σ_b ∈ [0, 2] |
| K      | 2.0 (supercritical ceiling) |
| H\*    | 1.0 (critical setpoint, σ_b = 1) |
| r      | 0.15/hour (homeostatic rate, Hengen lab) |
| σ      | 2.2 (CREP coupling) |
| Γ      | arctanh(σ_b / K) / σ → 0.251 at criticality |
