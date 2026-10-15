"""Tests for :mod:`hermes_rheo.synthetic`.

Check that each linear-viscoelastic model produces measurements with the
correct column names, units, metadata, and closed-form values.
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes_rheo.synthetic import (
    MultiModeMaxwellModel,
    SyntheticRheometry,
)

# --- helper ----------------------------------------------------------------


def _columns(meas):
    ds = meas.datasets[0]
    return dict(zip(ds.data_array_names, [np.asarray(a) for a in ds.data_arrays]))


# --- Maxwell ---------------------------------------------------------------


def test_maxwell_stress_relaxation_matches_closed_form():
    G0, tau = 1.0e5, 0.5
    model = SyntheticRheometry.maxwell(G0=G0, tau=tau)
    t = np.geomspace(1e-3, 10.0, 50)
    meas = model.to_measurement(experiment="stress_relaxation", x=t)

    cols = _columns(meas)
    np.testing.assert_allclose(cols["time"], t)
    np.testing.assert_allclose(cols["relaxation modulus"], G0 * np.exp(-t / tau), rtol=1e-12)

    assert meas.conditions["method"].startswith("Stress Relaxation Test")
    assert meas.conditions["synthetic"] is True
    assert meas.details["model"] == "Maxwell"
    assert meas.details["parameters"] == {"G0": G0, "tau": tau}


def test_maxwell_frequency_sweep_matches_closed_form():
    G0, tau = 1.0e5, 0.5
    model = SyntheticRheometry.maxwell(G0=G0, tau=tau)
    omega = np.geomspace(1e-2, 1e2, 40)
    meas = model.to_measurement(experiment="frequency_sweep", x=omega)

    cols = _columns(meas)
    wt = omega * tau
    gp_exp = G0 * wt**2 / (1 + wt**2)
    gpp_exp = G0 * wt / (1 + wt**2)
    np.testing.assert_allclose(cols["angular frequency"], omega)
    np.testing.assert_allclose(cols["storage modulus"], gp_exp, rtol=1e-12)
    np.testing.assert_allclose(cols["loss modulus"], gpp_exp, rtol=1e-12)
    np.testing.assert_allclose(cols["complex modulus"], np.sqrt(gp_exp**2 + gpp_exp**2), rtol=1e-12)
    np.testing.assert_allclose(cols["tan(delta)"], gpp_exp / gp_exp, rtol=1e-12)


def test_maxwell_creep_matches_closed_form():
    G0, tau = 2.0e5, 1.0
    model = SyntheticRheometry.maxwell(G0=G0, tau=tau)
    t = np.geomspace(1e-3, 10.0, 30)
    meas = model.to_measurement(experiment="creep", x=t)

    cols = _columns(meas)
    J_exp = (1.0 / G0) * (1.0 + t / tau)
    np.testing.assert_allclose(cols["creep compliance"], J_exp, rtol=1e-12)


# --- Multi-mode Maxwell ----------------------------------------------------


def test_multi_mode_maxwell_stress_relaxation():
    g = np.array([1.0e5, 1.0e4])
    tau = np.array([0.1, 1.0])
    model = SyntheticRheometry.multi_mode_maxwell(g=g, tau=tau, G_inf=100.0)
    t = np.geomspace(1e-3, 10.0, 40)
    meas = model.to_measurement(experiment="stress_relaxation", x=t)

    cols = _columns(meas)
    G_exp = 100.0 + sum(gi * np.exp(-t / ti) for gi, ti in zip(g, tau))
    np.testing.assert_allclose(cols["relaxation modulus"], G_exp, rtol=1e-12)


def test_multi_mode_maxwell_frequency_sweep_at_ginf():
    g = np.array([1.0e5])
    tau = np.array([1.0])
    G_inf = 500.0
    model = SyntheticRheometry.multi_mode_maxwell(g=g, tau=tau, G_inf=G_inf)
    omega = np.array([1e-6])
    meas = model.to_measurement(experiment="frequency_sweep", x=omega)
    cols = _columns(meas)
    assert cols["storage modulus"][0] == pytest.approx(G_inf, rel=1e-6)


def test_multi_mode_maxwell_requires_equal_lengths():
    with pytest.raises(ValueError):
        MultiModeMaxwellModel(g=[1.0, 2.0], tau=[1.0])


# --- Kelvin-Voigt ----------------------------------------------------------


def test_kelvin_voigt_creep_matches_closed_form():
    G, eta = 1.0e5, 5.0e4
    tau = eta / G
    model = SyntheticRheometry.kelvin_voigt(G=G, eta=eta)
    t = np.geomspace(1e-3, 10.0, 30)
    meas = model.to_measurement(experiment="creep", x=t)
    cols = _columns(meas)
    J_exp = (1.0 / G) * (1.0 - np.exp(-t / tau))
    np.testing.assert_allclose(cols["creep compliance"], J_exp, rtol=1e-12)


def test_kelvin_voigt_frequency_sweep():
    G, eta = 1.0e5, 5.0e4
    model = SyntheticRheometry.kelvin_voigt(G=G, eta=eta)
    omega = np.geomspace(1e-2, 1e2, 20)
    meas = model.to_measurement(experiment="frequency_sweep", x=omega)
    cols = _columns(meas)
    np.testing.assert_allclose(cols["storage modulus"], np.full_like(omega, G), rtol=1e-12)
    np.testing.assert_allclose(cols["loss modulus"], eta * omega, rtol=1e-12)


# --- Jeffreys --------------------------------------------------------------


def test_jeffreys_frequency_sweep_matches_closed_form():
    G0, tau, eta_s = 1.0e5, 1.0, 10.0
    model = SyntheticRheometry.jeffreys(G0=G0, tau=tau, eta_s=eta_s)
    omega = np.geomspace(1e-2, 1e2, 20)
    meas = model.to_measurement(experiment="frequency_sweep", x=omega)
    cols = _columns(meas)
    wt = omega * tau
    denom = 1.0 + wt**2
    gp_exp = G0 * wt**2 / denom
    gpp_exp = G0 * wt / denom + eta_s * omega
    np.testing.assert_allclose(cols["storage modulus"], gp_exp, rtol=1e-12)
    np.testing.assert_allclose(cols["loss modulus"], gpp_exp, rtol=1e-12)


# --- factory ---------------------------------------------------------------


def test_factory_experiment_set_generates_multiple_experiments():
    model = SyntheticRheometry.maxwell(G0=1e5, tau=1.0)
    x_t = np.geomspace(1e-3, 10.0, 20)
    x_w = np.geomspace(1e-2, 1e2, 20)
    exp = SyntheticRheometry.experiment_set(
        model,
        experiments=[
            {"experiment": "stress_relaxation", "x": x_t},
            {"experiment": "frequency_sweep", "x": x_w},
        ],
    )
    methods = {m.conditions["method"].split(" (")[0] for m in exp.measurements}
    assert methods == {"Stress Relaxation Test", "Frequency sweep"}
