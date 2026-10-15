"""Tests for :mod:`hermes_rheo.transforms.frequency_to_time`.

These tests lock in:
1. end-to-end reconstruction on real fixtures,
2. no-op behavior on non-frequency methods,
3. pipeline composition after OWChirp and SplitTemperatureSweep,
4. scientific correctness on a synthetic Maxwell spectrum,
5. compatibility with ``FitModel(domain='relaxation')``.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
import pytest
from piblin.data import Measurement

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.fit_model import FitModel
from hermes_rheo.transforms.frequency_to_time import FrequencyToTime
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep

try:
    from pyRheo.oscillation_model import SAOSModel
except ImportError:
    SAOSModel = None

requires_pyrheo = pytest.mark.skipif(SAOSModel is None, reason="pyRheo not installed")


@pytest.fixture()
def datadir(tmpdir, request):
    module_dir = os.path.dirname(request.module.__file__)
    test_dir = os.path.abspath(os.path.join(module_dir, "..", "fileio", "test_data"))
    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir


def _make_synthetic_maxwell_measurement(g0=1e5, tau=2.0):
    omega = np.logspace(-2, 3, 240)
    wt = omega * tau
    gp = g0 * (wt**2) / (1.0 + wt**2)
    gpp = g0 * wt / (1.0 + wt**2)

    ds = one_d_composite_dataset.OneDimensionalCompositeDataset(
        data_arrays=[omega, gp, gpp],
        data_array_names=["angular frequency", "storage modulus", "loss modulus"],
        data_array_units=["rad/s", "Pa", "Pa"],
        default_independent_name="angular frequency",
        default_dependent_name="storage modulus",
        source="Synthetic single-mode Maxwell in frequency domain",
    )
    return Measurement(datasets=[ds], conditions={"method": "Frequency sweep (synthetic)"})


# ---------------------------------------------------------------------------
# (a) End-to-end on standalone Frequency Sweep
# ---------------------------------------------------------------------------


def test_frequency_to_time_end_to_end_frequency_sweep(datadir):
    filepath = datadir.join("test_frequency_sweep_tts_ARES.txt")
    experiment = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)

    processed = FrequencyToTime()(experiment)
    measurement = processed.measurements[0]

    assert len(measurement.datasets) >= 2

    ds_time = measurement.datasets[0]
    ds_freq = measurement.datasets[1]

    assert ds_time.data_array_names[:3] == ["time", "G_from_eta1", "G_from_eta2"]
    assert "angular frequency" in ds_freq.data_array_names
    assert "storage modulus" in ds_freq.data_array_names
    assert "loss modulus" in ds_freq.data_array_names

    t = np.asarray(ds_time.data_arrays[ds_time.data_array_names.index("time")], dtype=float)
    g1 = np.asarray(ds_time.data_arrays[ds_time.data_array_names.index("G_from_eta1")], dtype=float)
    g2 = np.asarray(ds_time.data_arrays[ds_time.data_array_names.index("G_from_eta2")], dtype=float)

    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(g1))
    assert np.all(np.isfinite(g2))
    assert np.all(t > 0)
    assert np.all(np.diff(t) > 0)

    omega = np.asarray(
        ds_freq.data_arrays[ds_freq.data_array_names.index("angular frequency")], dtype=float
    )
    omega = omega[np.isfinite(omega) & (omega > 0)]
    omega = np.unique(np.sort(omega))
    assert t[0] == pytest.approx(1.0 / np.max(omega), rel=5e-2)
    assert t[-1] == pytest.approx(1.0 / np.min(omega), rel=5e-2)


# ---------------------------------------------------------------------------
# (b) No-op on Temperature Ramp
# ---------------------------------------------------------------------------


def test_frequency_to_time_noop_on_temperature_ramp(datadir):
    filepath = datadir.join("test_temp_ramp.txt")
    mset = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)

    ds_before = mset.measurements[0].datasets[0]
    n_before = len(mset.measurements[0].datasets)
    x_before = ds_before.independent_variable_name
    y_before = ds_before.dependent_variable_name

    processed = FrequencyToTime()(mset)

    ds_after = processed.measurements[0].datasets[0]
    n_after = len(processed.measurements[0].datasets)

    assert n_after == n_before
    assert ds_after.independent_variable_name == x_before
    assert ds_after.dependent_variable_name == y_before


# ---------------------------------------------------------------------------
# (c) Chained after OWChirpAnalysis
# ---------------------------------------------------------------------------


def test_frequency_to_time_after_owchirp_analysis(datadir):
    filepath = datadir.join("test_owchirp_DHR.txt")
    mset = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)

    processed = (OWChirpAnalysis() + FrequencyToTime())(mset)
    meas = processed.measurements[0]

    assert meas.datasets[0].data_array_names[:3] == ["time", "G_from_eta1", "G_from_eta2"]
    assert "angular frequency" in meas.datasets[1].data_array_names
    assert "storage modulus" in meas.datasets[1].data_array_names


# ---------------------------------------------------------------------------
# (d) Chained after SplitTemperatureSweep
# ---------------------------------------------------------------------------


def test_frequency_to_time_after_split_temperature_sweep(datadir):
    filepath = datadir.join("test_temp_sweep_tts.txt")
    mset = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)

    processed = (SplitTemperatureSweep(n_steps=5) + FrequencyToTime())(mset)

    assert len(processed.measurements) > 1
    for meas in processed.measurements:
        assert meas.datasets[0].data_array_names[:3] == ["time", "G_from_eta1", "G_from_eta2"]
        assert "angular frequency" in meas.datasets[1].data_array_names


# ---------------------------------------------------------------------------
# (e) Scientific correctness on synthetic single-mode Maxwell
# ---------------------------------------------------------------------------


def test_frequency_to_time_reconstructs_synthetic_maxwell():
    g0 = 1e5
    tau = 2.0
    measurement = _make_synthetic_maxwell_measurement(g0=g0, tau=tau)

    processed = FrequencyToTime()(measurement)
    ds_time = processed.datasets[0]

    t = np.asarray(ds_time.data_arrays[ds_time.data_array_names.index("time")], dtype=float)
    g_recon = np.asarray(
        ds_time.data_arrays[ds_time.data_array_names.index("G_from_eta1")], dtype=float
    )

    g_exact = g0 * np.exp(-t / tau)

    interior = (t > 10.0 * t[0]) & (t < t[-1] / 10.0)
    assert np.count_nonzero(interior) > 20
    np.testing.assert_allclose(g_recon[interior], g_exact[interior], rtol=0.15)


# ---------------------------------------------------------------------------
# (f) MRAE reporting on synthetic Maxwell
# ---------------------------------------------------------------------------


def test_frequency_to_time_reports_mrae_on_synthetic_maxwell():
    g0 = 1e5
    tau = 2.0
    measurement = _make_synthetic_maxwell_measurement(g0=g0, tau=tau)

    tmp = FrequencyToTime()(measurement)
    t_report = np.asarray(tmp.datasets[0].data_arrays[0], dtype=float)
    gt = g0 * np.exp(-t_report / tau)

    tr = FrequencyToTime(experimental_gt=(t_report, gt))
    tr(measurement)

    assert tr.result is not None
    assert isinstance(tr.result.mrae_eta1, float)
    assert isinstance(tr.result.mrae_eta2, float)
    assert np.isfinite(tr.result.mrae_eta1)
    assert np.isfinite(tr.result.mrae_eta2)
    # eta2 branch is typically tighter; eta1 is more sensitive to LF anchor.
    assert tr.result.mrae_eta1 < 1.0
    assert tr.result.mrae_eta2 < 0.2


# ---------------------------------------------------------------------------
# (f2) Maxwell terminal tail fit (mirrors iRheoTempo V9 "Overlay Maxwell tail")
# ---------------------------------------------------------------------------


def test_frequency_to_time_maxwell_tail_recovers_synthetic_parameters():
    g0 = 1e5
    tau = 2.0
    measurement = _make_synthetic_maxwell_measurement(g0=g0, tau=tau)

    tr = FrequencyToTime(fit_maxwell_tail=True)
    tr(measurement)

    res = tr.result
    assert res is not None
    assert res.maxwell_tail_available is True
    assert res.maxwell_G0 == pytest.approx(g0, rel=0.2)
    assert res.maxwell_tau == pytest.approx(tau, rel=0.2)
    assert res.maxwell_wcross == pytest.approx(1.0 / tau, rel=0.5)
    assert res.maxwell_ttail.size > 0
    assert res.maxwell_Gtail.size == res.maxwell_ttail.size
    # G_tail(t) = G0 * exp(-t/tau) — spot-check at the earliest sample.
    np.testing.assert_allclose(
        res.maxwell_Gtail[0],
        res.maxwell_G0 * np.exp(-res.maxwell_ttail[0] / res.maxwell_tau),
        rtol=1e-10,
    )


def test_frequency_to_time_maxwell_tail_disabled_by_default():
    measurement = _make_synthetic_maxwell_measurement()
    tr = FrequencyToTime()
    tr(measurement)

    res = tr.result
    assert res is not None
    assert res.maxwell_tail_available is False
    assert res.maxwell_ttail.size == 0
    assert res.maxwell_Gtail.size == 0


def test_frequency_to_time_use_maxwell_eta10_overrides_polynomial_anchor():
    g0 = 1e5
    tau = 2.0
    measurement = _make_synthetic_maxwell_measurement(g0=g0, tau=tau)

    tr_poly = FrequencyToTime(fit_maxwell_tail=True, use_maxwell_eta10=False)
    tr_poly(measurement)
    tr_max = FrequencyToTime(fit_maxwell_tail=True, use_maxwell_eta10=True)
    tr_max(measurement)

    assert tr_poly.result is not None
    assert tr_max.result is not None
    # Maxwell anchor should equal G0*tau to within fmin precision.
    assert tr_max.result.eta10_fit == pytest.approx(
        tr_max.result.maxwell_G0 * tr_max.result.maxwell_tau,
        rel=1e-6,
    )
    # Polynomial anchor should differ from the Maxwell-derived anchor for a
    # sharply single-mode spectrum.
    assert tr_poly.result.eta10_fit != pytest.approx(tr_max.result.eta10_fit, rel=1e-6)


# ---------------------------------------------------------------------------
# (g) FrequencyToTime + FitModel(domain="relaxation")
# ---------------------------------------------------------------------------


@requires_pyrheo
def test_frequency_to_time_then_fitmodel_relaxation_maxwell():
    g0 = 1e5
    tau = 2.0
    measurement = _make_synthetic_maxwell_measurement(g0=g0, tau=tau)

    processed = FrequencyToTime()(measurement)
    fit = FitModel(
        domain="relaxation",
        model_name="Maxwell",
        initial_guesses="random",
        num_initial_guesses=200,
    )(processed)

    assert fit.fitted_model is not None
    assert fit.t_common is not None
    assert fit.G_relax is not None

    params = np.asarray(getattr(fit.fitted_model, "params_", []), dtype=float)
    assert params.size >= 2

    # In pyRheo RelaxationModel(Maxwell), params_ are [G_s, eta_s].
    g_s = float(params[0])
    eta_s = float(params[1])

    assert g_s == pytest.approx(g0, rel=0.2)
    assert eta_s == pytest.approx(g0 * tau, rel=0.2)


# ---------------------------------------------------------------------------
# (h) FitModel oscillation backward-compatibility smoke test
# ---------------------------------------------------------------------------


@requires_pyrheo
def test_fitmodel_backward_compat_smoke_frequency_sweep(datadir):
    filepath = datadir.join("test_frequency_sweep_tts_ARES.txt")
    experiment = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)

    # Seed numpy so the SAOSModel random-restart optimizer is deterministic
    # across OS / Python combinations. FractionalMaxwell has pathological derivatives near alpha == beta and
    # upstream pyRheo crashes with `TypeError: Value after * must be an
    # iterable, not NoneType` when none of the random restarts converge.
    np.random.seed(0)

    fit_default_domain = FitModel(
        model=SAOSModel,
        model_name="FractionalMaxwell",
        measurement_index=0,
        num_initial_guesses=200,
    )(experiment)

    fit_explicit_oscillation = FitModel(
        domain="oscillation",
        model=SAOSModel,
        model_name="FractionalMaxwell",
        measurement_index=0,
        num_initial_guesses=200,
    )(experiment)

    assert fit_default_domain.fitted_model is not None
    assert fit_explicit_oscillation.fitted_model is not None
    assert fit_default_domain.omega_common is not None
    assert fit_explicit_oscillation.omega_common is not None
    assert fit_default_domain.omega_common.shape == fit_explicit_oscillation.omega_common.shape
    assert np.all(np.isfinite(fit_default_domain.G_prime))
    assert np.all(np.isfinite(fit_default_domain.G_double_prime))
