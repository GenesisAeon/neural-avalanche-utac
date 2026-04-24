# Scientific Background

## Brain Criticality as a UTAC System

The brain's self-organised critical state is modelled as a UTAC
(Universal Threshold Attractor Cascade) dynamical system.

### UTAC ODE

$$\frac{dH}{dt} = r \cdot (H_\text{target} - H), \quad H_\text{target} = K \cdot \tanh(\sigma \cdot \Gamma)$$

At the homeostatic fixed point H\* = 1:

$$\Gamma^* = \frac{\tanh^{-1}(H^*/K)}{\sigma} = \frac{\tanh^{-1}(0.5)}{2.2} \approx 0.251$$

### Cross-Domain Universality

| System | Package | Γ | η (efficiency) |
|--------|---------|---|----------------|
| Solar Flare | P21 | 0.014 | 1% |
| Cygnus X-1 Jet | P17 | 0.046 | 5% |
| Amazon Rainforest | P19 | 0.116 | 12% |
| **AMOC Ocean** | **P18** | **0.251** | **50%** |
| **Neural Criticality** | **P20** | **0.251** | **50%** |
| BTW Sandpile | P22 | 0.296 | 58% |
| Manna Sandpile | P22 | 0.376 | 72% |
| ERA5 Arctic | P01 | 0.920 | 99% |

**Key result:** Γ_brain = Γ_AMOC = 0.251 — both systems operate at η = 50%.

## CREP Tensor Components

| Component | Neural meaning | Critical value |
|-----------|----------------|----------------|
| C | AR(1) autocorrelation (critical slowing down) | → 1 |
| R | Power-law τ proximity to 3/2 | → 1 |
| E | Fano factor excess (avalanche co-activation) | high |
| P | Permutation entropy at intermediate disorder | peaked |

## Avalanche Statistics

At criticality (σ_b = 1), a branching process produces:

- Size distribution: P(S) ~ S^(−3/2) (mean-field exponent τ = 3/2)
- Duration distribution: P(D) ~ D^(−2) (mean-field exponent α = 2)

Reference: Zapperi et al. (1995); Beggs & Plenz (2003).

## Ethics-Gate Light (Phase H)

The `EthicsGate` monitors system tension at each simulation step:

```
tension = clip(Γ × 2 + |H − H*| / H* × 0.15, 0, 1)
```

If tension > 0.85, the gate raises `RuntimeError` before any data export,
preventing spurious results from unstable or runaway configurations.

## References

- Hengen, K.B. & Shew, W.L. (2025). *Neuron* 113(16), 2582–2598.
  DOI: [10.1016/j.neuron.2025.05.020](https://doi.org/10.1016/j.neuron.2025.05.020)
- *Phys. Rev. Lett.* 134, 028401 (2025).
  DOI: [10.1103/PhysRevLett.134.028401](https://doi.org/10.1103/PhysRevLett.134.028401)
- Sugimoto et al. (2025). *Front. Comput. Neurosci.*
  DOI: [10.3389/fncom.2025.1560691](https://doi.org/10.3389/fncom.2025.1560691)
- Beggs, J.M. & Plenz, D. (2003). *J. Neurosci.* 23(35), 11167–11177.
- Clauset, A. et al. (2009). *SIAM Review* 51(4), 661–703.
