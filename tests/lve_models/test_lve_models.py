"""Tests for linear viscoelastic models and protocols."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma

from hermes_rheo.lve_models import (
    FractionalKelvinVoigtModel,
    FractionalMaxwellGelModel,
    FractionalMaxwellLiquidModel,
    FractionalMaxwellModel,
    FractionalZenerModel,
    JeffreysModel,
    KelvinVoigtModel,
    MaxwellModel,
    MultiModeMaxwellModel,
    Protocols,
    SpringPotModel,
    invert_laplace,
)
from hermes_rheo.lve_models.base import LVEModel

T = np.geomspace(1e-3, 1e2, 40)
W = np.geomspace(1e-3, 1e3, 40)

CLASSICAL = [
    (MaxwellModel(G=1e5, tau=2.0), 0.0),
    (KelvinVoigtModel(G=1e5, eta=1e4), 1e4),
    (JeffreysModel(G=1e5, tau=2.0, eta_s=300.0), 300.0),
    (MultiModeMaxwellModel(g=[1e5, 1e4, 3e3], tau=[0.1, 1.0, 10.0], G_inf=500.0), 0.0),
]
FRACTIONAL = [
    SpringPotModel(V=3e3, alpha=0.4),
    FractionalMaxwellGelModel(G=1e4, V=3e3, alpha=0.6),
    FractionalMaxwellLiquidModel(G=2e3, eta=5e3, beta=0.4),
    FractionalMaxwellModel(G=2e3, V=3e3, alpha=0.7, beta=0.3),
    FractionalKelvinVoigtModel(G=2e3, V=3e3, alpha=0.7, beta=0.3),
    FractionalZenerModel(G=2e3, V=3e3, K=1e2, alpha=0.7, beta=0.3, kappa=0.5),
]
MODELS = [model for model, _ in CLASSICAL] + FRACTIONAL


def _model_id(model):
    return model.name


def _columns(measurement):
    dataset = measurement.datasets[0]
    return dict(zip(dataset.data_array_names, map(np.asarray, dataset.data_arrays)))


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (lambda s: 1.0 / (s + 2.0), lambda t: np.exp(-2.0 * t)),
        (lambda s: 1.0 / s**1.5, lambda t: t**0.5 / gamma(1.5)),
    ],
)
def test_invert_laplace_known_pairs(transform, expected):
    t = np.geomspace(1e-2, 1e2, 20)
    np.testing.assert_allclose(invert_laplace(transform, t), expected(t), rtol=1e-8, atol=1e-9)


def test_invert_laplace_rejects_nonpositive_time():
    with pytest.raises(ValueError):
        invert_laplace(lambda s: 1.0 / s, np.array([0.0, 1.0]))


def test_maxwell_analytic_limits():
    model = MaxwellModel(G=1e5, tau=0.5)
    wt = W * model.parameters["tau"]
    storage, loss = model.saos(W)
    np.testing.assert_allclose(model.stress_relaxation(T), 1e5 * np.exp(-T / 0.5), rtol=1e-12)
    np.testing.assert_allclose(model.creep(T), (1.0 + T / 0.5) / 1e5, rtol=1e-12)
    np.testing.assert_allclose(storage, 1e5 * wt**2 / (1.0 + wt**2), rtol=1e-12)
    np.testing.assert_allclose(loss, 1e5 * wt / (1.0 + wt**2), rtol=1e-12)


def test_kelvin_voigt_analytic_limits():
    model = KelvinVoigtModel(G=1e5, eta=1e4)
    storage, loss = model.saos(W)
    np.testing.assert_allclose(storage, 1e5)
    np.testing.assert_allclose(loss, 1e4 * W)
    np.testing.assert_allclose(model.creep(T), (1.0 - np.exp(-T / 0.1)) / 1e5, rtol=1e-12)


def test_jeffreys_creep_long_time_limit():
    G, tau, eta_s = 1e5, 2.0, 300.0
    model = JeffreysModel(G=G, tau=tau, eta_s=eta_s)
    eta = G * tau + eta_s
    t = np.array([1e3, 2e3])
    creep = model.creep(t)
    assert model.creep(np.array([0.0]))[0] == 0.0
    np.testing.assert_allclose(np.diff(creep) / np.diff(t), 1.0 / eta, rtol=1e-9)
    np.testing.assert_allclose(creep[0] - t[0] / eta, tau * G * tau / eta**2, rtol=1e-9)


def test_multimode_maxwell_analytic_relaxation_limit():
    model = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0], G_inf=500.0)
    t = np.array([0.0, 1.0, 100.0])
    expected = 500.0 + 1e5 * np.exp(-t / 0.1) + 1e4 * np.exp(-t)
    np.testing.assert_allclose(model.stress_relaxation(t), expected, rtol=1e-12)
    assert model.stress_relaxation(t)[-1] == pytest.approx(500.0, rel=1e-12)


def test_springpot_saos_power_law():
    model = SpringPotModel(V=3e3, alpha=0.4)
    storage, loss = model.saos(W)
    np.testing.assert_allclose(loss / storage, np.tan(np.pi * 0.4 / 2.0), rtol=1e-12)
    np.testing.assert_allclose(storage, 3e3 * W**0.4 * np.cos(np.pi * 0.4 / 2.0), rtol=1e-12)


def _mittag_leffler(z, alpha, n_terms=150):
    k = np.arange(n_terms)
    return np.sum(z**k / gamma(alpha * k + 1.0))


def test_fractional_maxwell_gel_relaxation_matches_mittag_leffler():
    G, alpha = 1e4, 0.6
    V = G * 2.0**alpha
    model = FractionalMaxwellGelModel(G=G, V=V, alpha=alpha)
    t = np.array([0.05, 0.3, 1.0, 2.0, 3.0])
    expected = [G * _mittag_leffler(-((value / 2.0) ** alpha), alpha) for value in t]
    np.testing.assert_allclose(model.stress_relaxation(t), expected, rtol=1e-8)


def test_fractional_maxwell_gel_approaches_maxwell():
    G, tau = 1e5, 1.0
    gel = FractionalMaxwellGelModel(G=G, V=G * tau**0.9999, alpha=0.9999)
    maxwell = MaxwellModel(G=G, tau=tau)
    t = np.geomspace(1e-2, 3.0, 10)
    np.testing.assert_allclose(gel.stress_relaxation(t), maxwell.stress_relaxation(t), rtol=1e-2)
    np.testing.assert_allclose(
        gel.saos(np.geomspace(0.1, 10.0, 10)), maxwell.saos(np.geomspace(0.1, 10.0, 10)), rtol=2e-3
    )


def test_fractional_kelvin_voigt_relaxation_matches_numerical_solution():
    model = FractionalKelvinVoigtModel(G=2e3, V=3e3, alpha=0.7, beta=0.3)
    t = np.array([0.1, 1.0, 10.0])
    expected = 2e3 * t**-0.3 / gamma(0.7) + 3e3 * t**-0.7 / gamma(0.3)
    np.testing.assert_allclose(model.stress_relaxation(t), expected, rtol=1e-12)
    np.testing.assert_allclose(LVEModel.stress_relaxation(model, t), expected, rtol=1e-7)


@pytest.mark.parametrize("model", [CLASSICAL[0][0], FRACTIONAL[0]], ids=_model_id)
def test_analytical_and_numerical_creep_agree(model):
    t = np.geomspace(1e-2, 30.0, 25)
    np.testing.assert_allclose(model.creep(t), LVEModel.creep(model, t), rtol=1e-8)


@pytest.mark.parametrize(
    "model, impulse", [CLASSICAL[0], (FRACTIONAL[0], 0.0)], ids=["Maxwell", "SpringPot"]
)
def test_creep_relaxation_convolution_identity(model, impulse):
    t_end = 5.0

    def integrand(u):
        return float(model.stress_relaxation(np.array([t_end - u]))[0]) * float(
            model.creep(np.array([u]))[0]
        )

    value, _ = quad(integrand, 0.0, t_end, limit=400, epsabs=0.0, epsrel=1e-8)
    value += impulse * float(model.creep(np.array([t_end]))[0])
    assert value == pytest.approx(t_end, rel=1e-4)


def test_invalid_model_parameters_raise():
    invalid_models = [
        lambda: SpringPotModel(V=1.0, alpha=0.0),
        lambda: SpringPotModel(V=1.0, alpha=1.0),
        lambda: FractionalMaxwellModel(G=1.0, V=1.0, alpha=0.3, beta=0.7),
        lambda: FractionalZenerModel(G=1.0, V=1.0, K=1.0, alpha=0.3, beta=0.3, kappa=0.5),
        lambda: FractionalMaxwellGelModel(G=-1.0, V=1.0, alpha=0.5),
        lambda: MaxwellModel(G=1.0, tau=0.0),
        lambda: MultiModeMaxwellModel(g=[1.0, 2.0], tau=[1.0]),
    ]
    for factory in invalid_models:
        with pytest.raises(ValueError):
            factory()


@pytest.mark.parametrize("model", MODELS, ids=_model_id)
def test_protocols_return_described_piblin_measurements(model):
    for protocol, x, names in (
        (
            "saos",
            W,
            [
                "angular frequency",
                "storage modulus",
                "loss modulus",
                "complex modulus",
                "tan(delta)",
            ],
        ),
        ("creep", T, ["time", "creep compliance"]),
        ("stress_relaxation", T, ["time", "relaxation modulus"]),
    ):
        measurement = getattr(Protocols, protocol)(model, x)
        dataset = measurement.datasets[0]
        assert dataset.data_array_names == names
        assert all(np.all(np.isfinite(values)) for values in dataset.data_arrays)
        assert measurement.conditions["simulated"] is True
        assert measurement.conditions["method"].endswith(f"(simulated {model.name})")
        assert measurement.details["model"] == model.name
        assert measurement.details["parameters"] == model.parameters
        assert measurement.details["protocol"] == protocol
        assert measurement.details["grid"] == {
            "variable": names[0],
            "points": x.size,
            "min": x.min(),
            "max": x.max(),
        }
        assert measurement.details["references"]
        assert measurement.details["software"].startswith("hermes-rheo")

    saos = _columns(Protocols.saos(model, W))
    storage, loss = model.saos(W)
    np.testing.assert_allclose(saos["angular frequency"], W)
    np.testing.assert_allclose(saos["storage modulus"], storage)
    np.testing.assert_allclose(saos["loss modulus"], loss)
    np.testing.assert_allclose(saos["complex modulus"], np.hypot(storage, loss))
    np.testing.assert_allclose(saos["tan(delta)"], loss / storage)


def test_protocol_units_and_extra_metadata():
    model = MaxwellModel(G=1e5, tau=0.5)
    measurement = Protocols.saos(model, W, conditions={"temperature": 25.0}, details={"note": "x"})
    assert measurement.datasets[0].data_array_units == ["rad/s", "Pa", "Pa", "Pa", ""]
    assert Protocols.creep(model, T).datasets[0].data_array_units == ["s", "1/Pa"]
    assert Protocols.stress_relaxation(model, T).datasets[0].data_array_units == ["s", "Pa"]
    assert measurement.conditions["temperature"] == 25.0
    assert measurement.details["note"] == "x"


def test_invalid_protocol_inputs_raise():
    model = MaxwellModel(G=1e5, tau=1.0)
    invalid_calls = [
        lambda: model.to_measurement("frequency_sweep", W),
        lambda: model.to_measurement("creep", np.array([])),
        lambda: Protocols.saos(model, np.array([0.0, 1.0])),
        lambda: Protocols.experiment_set(model, []),
        lambda: Protocols.experiment_set(model, [{"protocol": "saos"}]),
    ]
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()


def test_experiment_set_runs_several_protocols():
    model = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
    experiment = Protocols.experiment_set(
        model,
        [
            {"protocol": "stress_relaxation", "x": T},
            {"protocol": "saos", "x": W},
            {"protocol": "creep", "x": T},
        ],
    )
    assert len(experiment.measurements) == 3
