"""Test the Diamond-Template contract for NeuralAvalancheUTAC.

Verifies that all five mandatory methods exist, return the correct types,
and contain the required keys.
"""

from __future__ import annotations

import pytest

from neural_avalanche_utac.system import NeuralAvalancheUTAC


@pytest.fixture(scope="module")
def model() -> NeuralAvalancheUTAC:
    """Small model with a 30s cycle for fast tests."""
    return NeuralAvalancheUTAC(n_neurons=100, seed=42, segment_s=30.0)


@pytest.fixture(scope="module")
def cycled_model(model: NeuralAvalancheUTAC) -> NeuralAvalancheUTAC:
    """Model after one 30s cycle."""
    model.run_cycle(duration_seconds=30.0)
    return model


# ── run_cycle ─────────────────────────────────────────────────────────────────

def test_run_cycle_returns_dict(model):
    result = model.run_cycle(duration_seconds=30.0)
    assert isinstance(result, dict)


def test_run_cycle_required_keys(model):
    result = model.run_cycle(duration_seconds=30.0)
    for key in ("H_final", "sigma_b_mean", "gamma_mean", "n_avalanches_total",
                "crep_state", "utac_state", "is_critical"):
        assert key in result, f"Missing key: {key}"


def test_run_cycle_h_in_valid_range(model):
    result = model.run_cycle(duration_seconds=30.0)
    assert 0.0 < result["H_final"] < 2.0


def test_run_cycle_gamma_positive(model):
    result = model.run_cycle(duration_seconds=30.0)
    assert result["gamma_mean"] > 0.0


# ── get_crep_state ────────────────────────────────────────────────────────────

def test_get_crep_state_returns_dict(cycled_model):
    crep = cycled_model.get_crep_state()
    assert isinstance(crep, dict)


def test_get_crep_state_keys(cycled_model):
    crep = cycled_model.get_crep_state()
    for key in ("C", "R", "E", "P", "Gamma"):
        assert key in crep, f"Missing CREP key: {key}"


def test_get_crep_state_c_r_e_p_in_range(cycled_model):
    crep = cycled_model.get_crep_state()
    for key in ("C", "R", "E", "P"):
        val = crep[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} not in [0,1]"


def test_get_crep_state_gamma_positive(cycled_model):
    crep = cycled_model.get_crep_state()
    assert crep["Gamma"] > 0.0


# ── get_utac_state ────────────────────────────────────────────────────────────

def test_get_utac_state_returns_dict(cycled_model):
    utac = cycled_model.get_utac_state()
    assert isinstance(utac, dict)


def test_get_utac_state_keys(cycled_model):
    utac = cycled_model.get_utac_state()
    for key in ("H", "dH_dt", "H_star", "K_eff"):
        assert key in utac, f"Missing UTAC key: {key}"


def test_get_utac_state_h_valid(cycled_model):
    utac = cycled_model.get_utac_state()
    assert 0.0 < utac["H"] < 2.0


def test_get_utac_state_k_eff(cycled_model):
    utac = cycled_model.get_utac_state()
    assert utac["K_eff"] == pytest.approx(2.0)


# ── get_phase_events ──────────────────────────────────────────────────────────

def test_get_phase_events_returns_list(cycled_model):
    events = cycled_model.get_phase_events()
    assert isinstance(events, list)


def test_get_phase_events_structure(cycled_model):
    events = cycled_model.get_phase_events()
    if events:
        evt = events[0]
        for key in ("step", "t_s", "n_avalanches", "max_size", "sigma_b", "gamma"):
            assert key in evt, f"Missing phase event key: {key}"


# ── to_zenodo_record ──────────────────────────────────────────────────────────

def test_to_zenodo_record_returns_dict(cycled_model):
    record = cycled_model.to_zenodo_record()
    assert isinstance(record, dict)


def test_to_zenodo_record_required_keys(cycled_model):
    record = cycled_model.to_zenodo_record()
    for key in ("title", "creators", "license", "doi", "utac_state",
                "crep_state", "gamma_brain", "package_registry"):
        assert key in record, f"Missing Zenodo key: {key}"


def test_to_zenodo_record_universality_field(cycled_model):
    record = cycled_model.to_zenodo_record()
    assert "gamma_universality_confirmed" in record
    assert isinstance(record["gamma_universality_confirmed"], bool)


# ── is_critical + gamma_universality_check ───────────────────────────────────

def test_is_critical_returns_bool(cycled_model):
    assert isinstance(cycled_model.is_critical(), bool)


def test_gamma_universality_check_structure(cycled_model):
    result = cycled_model.gamma_universality_check()
    for key in ("gamma_brain_measured", "gamma_brain_theoretical",
                "gamma_amoc", "universality_confirmed", "interpretation"):
        assert key in result


def test_gamma_universality_check_theoretical_value(cycled_model):
    result = cycled_model.gamma_universality_check()
    # Γ_theoretical = arctanh(0.5) / 2.2 ≈ 0.251
    assert result["gamma_brain_theoretical"] == pytest.approx(0.2497, abs=0.001)


# ── Ethics-Gate Light ─────────────────────────────────────────────────────────

def test_ethics_gate_allows_normal_state():
    from neural_avalanche_utac.system import EthicsGate

    gate = EthicsGate()
    result = gate.check(state={"H": 1.0}, tension=0.5)
    assert result["allowed"] is True


def test_ethics_gate_blocks_high_tension():
    from neural_avalanche_utac.system import EthicsGate

    gate = EthicsGate()
    result = gate.check(state={"H": 1.0}, tension=0.9)
    assert result["allowed"] is False
    assert "reason" in result


def test_tension_metric_range():
    from neural_avalanche_utac.system import TensionMetric

    tm = TensionMetric()
    tm.update(gamma=0.251, h=1.0, h_star=1.0)
    t = tm.get_current_tension()
    assert 0.0 <= t <= 1.0
