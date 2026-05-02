# neural-avalanche-utac

> GenesisAeon Package 21 — Brain Criticality & Neuronal Avalanches as UTAC System

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19645351"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19645351.svg" alt="DOI (GenesisAeon Whitepaper)"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="GPLv3 License"/></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/docs-CC%20BY%204.0-lightblue.svg" alt="CC BY 4.0"/></a>
  <a href="https://github.com/GenesisAeon/genesis-os"><img src="https://img.shields.io/badge/part%20of-genesis--os-blueviolet" alt="Part of genesis-os"/></a>
  <img src="https://img.shields.io/badge/UTAC-package%2021-orange" alt="Package 21"/>
</p>

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
