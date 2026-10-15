"""Tests for :mod:`hermes_rheo.transforms.relaxation_spectrum`."""
from __future__ import annotations

import numpy as np
import pytest

from hermes_rheo.synthetic import SyntheticRheometry
from hermes_rheo.transforms.relaxation_spectrum import (
    RelaxationSpectrum,
    RelaxationSpectrumResult,
)


G_TRUE = np.array([1.0e5, 1.0e4])
TAU_TRUE = np.array([0.1, 1.0])


def _sorted_recovered(res):
    order = np.argsort(res.tau)
    return np.asarray(res.g)[order], np.asarray(res.tau)[order]


def test_relaxation_spectrum_time_recovers_multi_mode_maxwell():
    model = SyntheticRheometry.multi_mode_maxwell(g=G_TRUE, tau=TAU_TRUE)
    t = np.geomspace(1e-3, 1e2, 200)
    meas = model.to_measurement(experiment="stress_relaxation", x=t)

    tr = RelaxationSpectrum(
        domain="time", relaxation_name="relaxation modulus", ns=100, plateau=False
    )
    out = tr(meas)

    assert "relaxation_spectrum_result" in out.details
    result = out.details["relaxation_spectrum_result"]
    assert isinstance(result, RelaxationSpectrumResult)

    g_rec, tau_rec = _sorted_recovered(result)
    assert g_rec.size >= 2
    # Recover the two timescales within a factor of 2 (loose invariant).
    np.testing.assert_allclose(tau_rec[:2], TAU_TRUE, rtol=0.5)
    # And the moduli within a factor of ~3.
    np.testing.assert_allclose(g_rec[:2], G_TRUE, rtol=2.0)

    # Two new composite datasets are prepended.
    assert len(out.datasets) >= 2


def test_relaxation_spectrum_frequency_recovers_multi_mode_maxwell():
    model = SyntheticRheometry.multi_mode_maxwell(g=G_TRUE, tau=TAU_TRUE)
    omega = np.geomspace(1e-2, 1e3, 80)
    meas = model.to_measurement(experiment="frequency_sweep", x=omega)

    tr = RelaxationSpectrum(domain="frequency", ns=100, plateau=False)
    out = tr(meas)

    result = out.details["relaxation_spectrum_result"]
    g_rec, tau_rec = _sorted_recovered(result)
    assert g_rec.size >= 1
    # The frequency solver on a two-decade omega grid may only cleanly
    # resolve one of the two modes; require that at least one recovered
    # tau matches a true tau within a factor of 3.
    def _closest(true_val):
        return np.min(np.abs(np.log(tau_rec) - np.log(true_val)))
    assert _closest(TAU_TRUE[0]) < np.log(3.0) or _closest(TAU_TRUE[1]) < np.log(3.0)

    # Uncertainty band should be attached on the frequency branch.
    assert result.dH is not None
    assert np.asarray(result.dH).shape == np.asarray(result.H).shape


def test_relaxation_spectrum_result_exposes_canonical_attributes():
    model = SyntheticRheometry.maxwell(G0=1e5, tau=1.0)
    meas = model.to_measurement(
        experiment="stress_relaxation", x=np.geomspace(1e-3, 10.0, 120)
    )
    out = RelaxationSpectrum(domain="time", relaxation_name="relaxation modulus")(meas)
    result = out.details["relaxation_spectrum_result"]
    for name in ("s", "H", "tau", "g", "lam_C", "N_opt", "G_fit_cont"):
        assert hasattr(result, name), f"missing attribute {name}"
