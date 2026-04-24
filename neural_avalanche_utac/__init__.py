"""
neural-avalanche-utac — GenesisAeon Package 20
Brain Criticality & Neuronal Avalanche Threshold

Central result:
  Γ_brain ≈ 0.251 = Γ_AMOC  (cross-domain CREP universality at η = 50%)

Diamond-Template contract:
  NeuralAvalancheUTAC.run_cycle()        → dict
  NeuralAvalancheUTAC.get_crep_state()   → {C, R, E, P, Gamma}
  NeuralAvalancheUTAC.get_utac_state()   → {H, dH_dt, H_star, K_eff}
  NeuralAvalancheUTAC.get_phase_events() → list
  NeuralAvalancheUTAC.to_zenodo_record() → dict
"""

__version__ = "0.1.0"
__author__ = "Johann Römer / MOR Research Collective"
__license__ = "MIT"

from neural_avalanche_utac.system import NeuralAvalancheUTAC
from neural_avalanche_utac.constants import GAMMA_BRAIN, NEURAL_TARGETS, PACKAGE_REGISTRY_ENTRY

__all__ = [
    "NeuralAvalancheUTAC",
    "GAMMA_BRAIN",
    "NEURAL_TARGETS",
    "PACKAGE_REGISTRY_ENTRY",
]
