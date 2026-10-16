"""Tg detection of TemperatureRampAnalysis on synthetic temperature ramps."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from piblin.data import ExperimentSet, Measurement, OneDimensionalCompositeDataset  # noqa: E402

from hermes_rheo.transforms.temperature_ramp_analysis import (  # noqa: E402
    TemperatureRampAnalysis,
    find_tan_delta_peaks,
    storage_modulus_onset,
)

T = np.linspace(-100.0, 150.0, 801)
LOG_G = 8.0 - 1.5 * (1.0 + np.tanh((T + 20.0) / 10.0))  # onset at -30, inflection at -20


def _peak(center, height, width):
    return height * np.exp(-(((T - center) / width) ** 2))


def _ramp(tan_delta, storage_modulus=10**LOG_G):
    dataset = OneDimensionalCompositeDataset(
        data_arrays=[T, np.full_like(T, 2 * np.pi), np.ones_like(T), storage_modulus, tan_delta],
        data_array_names=[
            "temperature",
            "angular frequency",
            "frequency",
            "storage modulus",
            "tan(delta)",
        ],
        data_array_units=["°C", "rad/s", "Hz", "Pa", ""],
        default_independent_name="temperature",
        default_dependent_name="storage modulus",
    )
    return ExperimentSet(
        [Measurement(datasets=[dataset], conditions={"method": "Temperature ramp - 1"})]
    )


def test_tan_delta_peaks_resolves_two_transitions():
    tan_delta = 0.2 + _peak(-40, 1.0, 8) + _peak(60, 0.6, 10)
    peaks = find_tan_delta_peaks(T, tan_delta)

    np.testing.assert_allclose(peaks.temperatures, [-40, 60], atol=0.5)
    assert peaks.tg == pytest.approx(-40, abs=0.5)


def test_tan_delta_primary_peak_is_most_prominent_not_highest():
    # The high-temperature peak is higher but sits on a raised baseline.
    step = 0.2 * (1.0 + np.tanh((T - 30.0) / 5.0))
    tan_delta = 0.2 + _peak(-40, 0.6, 8) + _peak(80, 0.3, 10) + step
    peaks = find_tan_delta_peaks(T, tan_delta)

    assert len(peaks.temperatures) == 2
    assert peaks.heights[1] > peaks.heights[0]
    assert peaks.tg == pytest.approx(-40, abs=0.5)


def test_tan_delta_ignores_noise_and_shoulders():
    rng = np.random.default_rng(0)
    tan_delta = 0.3 + _peak(-30, 1.0, 8) + _peak(80, 0.05, 20) + 0.01 * rng.normal(size=T.size)
    peaks = find_tan_delta_peaks(T, tan_delta)

    assert len(peaks.temperatures) == 1
    assert peaks.tg == pytest.approx(-30, abs=2)


def test_tan_delta_keeps_small_narrow_peak_and_rejects_broad_rise():
    # Small secondary relaxation (~6 % of the range) plus a broad rise towards flow.
    rng = np.random.default_rng(1)
    tan_delta = (
        0.1
        + _peak(-10, 1.2, 15)
        + _peak(60, 0.08, 6)
        + _peak(140, 0.1, 45)
        + 0.003 * rng.normal(size=T.size)
    )
    peaks = find_tan_delta_peaks(T, tan_delta)

    np.testing.assert_allclose(peaks.temperatures, [-10, 60], atol=1.5)
    assert peaks.tg == pytest.approx(-10, abs=1.5)


def test_storage_modulus_onset_of_sigmoid():
    onset = storage_modulus_onset(T, 10**LOG_G)

    assert onset.t_inflection == pytest.approx(-20, abs=0.5)
    assert onset.tg_onset == pytest.approx(-30, abs=0.5)
    assert onset.tg_onset < onset.t_inflection


def test_storage_modulus_onset_without_glassy_plateau_warns():
    inside = T > -25
    with pytest.warns(UserWarning, match="glassy plateau"):
        onset = storage_modulus_onset(T[inside], 10 ** LOG_G[inside])
    assert np.isnan(onset.tg_onset) or onset.tg_onset <= onset.t_inflection


@pytest.mark.parametrize(
    ("method", "expected"), [("tan_delta", -40), ("storage_modulus", -30), ("both", -40)]
)
def test_tg_method(method, expected):
    ramp = TemperatureRampAnalysis(tg_method=method)(_ramp(0.2 + _peak(-40, 1.0, 8)))

    assert ramp.tg == pytest.approx(expected, abs=0.5)
    assert ramp.details["tg_method"] == method
    assert (ramp.onset is not None) == (method != "tan_delta")
    assert (ramp.tan_delta_peaks is not None) == (method != "storage_modulus")


def test_frequency_is_reported():
    ramp = TemperatureRampAnalysis(tg_method="both")(_ramp(0.2 + _peak(-40, 1.0, 8)))

    assert ramp.details["angular_frequency"] == pytest.approx(2 * np.pi)
    assert ramp.details["angular_frequency_unit"] == "rad/s"
    assert ramp.details["frequency"] == pytest.approx(1.0)
    assert ramp.details["frequency_unit"] == "Hz"
    assert "Frequency: 1 Hz" in ramp.summary()
    assert set(ramp.result.tg_values) == {
        "Tg (tan(delta) peak)",
        "Tg (storage modulus onset)",
        "Storage modulus inflection",
    }


def test_invalid_tg_method():
    with pytest.raises(ValueError, match="tg_method"):
        TemperatureRampAnalysis(tg_method="loss_modulus")


def test_plot_onset():
    ramp = TemperatureRampAnalysis(tg_method="storage_modulus")(_ramp(0.2 + _peak(-40, 1.0, 8)))
    fig, (ax_g, ax_d) = ramp.plot_onset()

    assert ax_g.get_ylabel() == "storage modulus (Pa)"
    assert "°C" in ax_d.get_xlabel()
    plt.close(fig)

    with pytest.raises(ValueError, match="onset"):
        TemperatureRampAnalysis()(_ramp(0.2 + _peak(-40, 1.0, 8))).plot_onset()
