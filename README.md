# neural-avalanche-utac

> GenesisAeon Package 20 — Brain Criticality & Neuronal Avalanches as UTAC System

[![GenesisAeon](https://img.shields.io/badge/GenesisAeon-Package%2020-blueviolet)](https://github.com/GenesisAeon)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19645351.svg)](https://doi.org/10.5281/zenodo.19645351)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reference](https://img.shields.io/badge/Ref-Neuron%202025-red)](https://doi.org/10.1016/j.neuron.2025.05.020)

**Neuronal avalanches at criticality modelled as UTAC system.**

**Key result**: Γ_brain ≈ 0.251 = Γ_AMOC → cross-domain universality at η = 50 %.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
neural-utac run --duration 3600
neural-utac criticality-check
neural-utac gamma-universality
```

## Integration in genesis-os

```python
from genesis_os import GenesisOS
os = GenesisOS()
neural = os.load_package(20)
results = neural.run_cycle(duration_seconds=3600)
```

## Benchmark

Validated against Hengen & Shew (2025).

## Falsifiable Prediction

Any homeostatic system with 50 % efficiency setpoint converges to Γ ≈ 0.251.

## License

Code: MIT • Docs & Data: CC BY 4.0
