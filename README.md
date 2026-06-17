# neural-avalanche-utac

> GenesisAeon Package 20 (P20) — Brain Criticality & Neuronal Avalanches as UTAC System

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19645351"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19645351.svg" alt="DOI (GenesisAeon Whitepaper)"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"/></a>
  <a href="https://github.com/GenesisAeon/genesis-os"><img src="https://img.shields.io/badge/part%20of-genesis--os-blueviolet" alt="Part of genesis-os"/></a>
  <img src="https://img.shields.io/badge/UTAC-package%2020-orange" alt="Package 20"/>
</p>

**Neuronal avalanches at criticality modelled as UTAC system.**

**Key result**: Γ_brain ≈ 0.251 = Γ_AMOC → cross-domain universality at η = 50 %.

## Installation

```bash
pip install neural-avalanche-utac
```

For local development:

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

## Role in the GenesisAeon Ecosystem

`neural-avalanche-utac` is GenesisAeon Package **P20** (domain: neuroscience
/ cortical criticality). It models neuronal avalanches at criticality as a
UTAC dynamical system and contributes the Γ_brain ≈ 0.251 cross-domain
universality result to the broader GenesisAeon CREP Criticality Spectrum.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)

DOI will be assigned automatically on first GitHub Release once
Zenodo–GitHub integration is enabled for this repo. In the meantime, see
`CITATION.cff` for the package citation and the GenesisAeon whitepaper DOI
badge above.

## License

MIT
