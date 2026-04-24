# API Reference

## Diamond-Template Interface

All five methods are mandatory per the GenesisAeon Diamond-Template contract.

### `NeuralAvalancheUTAC`

```python
from neural_avalanche_utac import NeuralAvalancheUTAC

model = NeuralAvalancheUTAC(
    n_neurons=500,       # number of simulated neurons
    seed=42,             # RNG seed (GenesisAeon standard)
    r=0.15,              # homeostatic plasticity rate per hour
    sigma_crep=2.2,      # CREP coupling constant
    segment_s=60.0,      # spike train segment length in seconds
)
```

#### `run_cycle(duration_seconds=3600.0) → dict`

Run one UTAC simulation cycle. Processes spike trains in `segment_s`-second
chunks, computing CREP, stepping the UTAC ODE, and applying homeostatic
correction at each step.

**Returns:** `H_final`, `sigma_b_mean`, `gamma_mean`, `n_avalanches_total`,
`crep_state`, `utac_state`, `is_critical`

#### `get_crep_state() → dict`

Returns `{C, R, E, P, Gamma, sigma_b}`.

| Key | Meaning |
|-----|---------|
| C | AR(1) autocorrelation (critical slowing down) |
| R | Power-law exponent proximity to τ = 3/2 |
| E | Fano factor excess (supra-additive variance) |
| P | Permutation entropy proximity (intermediate disorder) |
| Gamma | CREP tensor: `arctanh(σ_b / K) / σ` |

#### `get_utac_state() → dict`

Returns `{H, dH_dt, H_star, K_eff}`.

#### `get_phase_events() → list[dict]`

Each element records one avalanche cluster: `step`, `t_s`, `n_avalanches`,
`max_size`, `sigma_b`, `gamma`, `tension`.

#### `to_zenodo_record() → dict`

Zenodo-compatible metadata including UTAC/CREP state, universality result,
and package registry info. Blocked by Ethics-Gate if tension > 0.85.

---

## Supporting Classes

### `AvalancheDetector`

Beggs & Plenz (2003) avalanche detection from (n_neurons, n_bins) spike arrays.

```python
from neural_avalanche_utac.avalanche import AvalancheDetector

det = AvalancheDetector()
avalanches = det.detect(spikes)   # list[Avalanche]
sizes     = det.sizes(avalanches)
durations = det.durations(avalanches)
```

### `BranchingRatioEstimator`

Harris (1963) estimator σ̂_b = Σ n_{t+1} / Σ n_t.

```python
from neural_avalanche_utac.branching import BranchingRatioEstimator

est = BranchingRatioEstimator()
sigma_b = est.estimate(spikes)
ar1     = est.ar1_coefficient(spikes)   # CREP C-component
```

### `PowerLawFitter`

MLE + KS test for P(x) ~ x^(-τ) (Clauset et al. 2009).

```python
from neural_avalanche_utac.power_law import PowerLawFitter

fitter = PowerLawFitter(x_min=1.0)
result = fitter.fit_and_score(sizes, tau_critical=1.5)
# result: {tau, D, p_value, tau_proximity, ...}
```

### `NeuralCREPTensor`

```python
from neural_avalanche_utac.crep_neural import NeuralCREPTensor

crep = NeuralCREPTensor()
out  = crep.compute(spikes)   # {C, R, E, P, Gamma, sigma_b}
```

### `EthicsGate`

```python
from neural_avalanche_utac.system import EthicsGate

gate = EthicsGate()
result = gate.check(state={"H": 1.0}, tension=0.5)
# {"allowed": True, "reason": "ok"}
```
