"""Render tests for the plot_* functions on real and simulated piblin data."""

import os
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from piblin.data import ExperimentSet  # noqa: E402

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader  # noqa: E402
from hermes_rheo.lve_models import MultiModeMaxwellModel, Protocols  # noqa: E402
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve  # noqa: E402
from hermes_rheo.transforms.fit_model import FitModel  # noqa: E402
from hermes_rheo.transforms.relaxation_spectrum import RelaxationSpectrum  # noqa: E402
from hermes_rheo.transforms.shift_factor_models import ShiftFactorModel  # noqa: E402
from hermes_rheo.visualization import (  # noqa: E402
    plot_cole_cole,
    plot_master_curve,
    plot_model_fit,
    plot_relaxation_spectrum,
    plot_shift_factors,
    plot_spectrum_diagnostics,
    plot_vgp,
)
from hermes_rheo.visualization.units import axis_label  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def datadir(tmpdir, request):
    module_dir = os.path.dirname(request.module.__file__)
    test_dir = os.path.abspath(os.path.join(module_dir, "..", "fileio", "test_data"))
    shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir


@pytest.fixture()
def tts_experiment(datadir):
    path = str(datadir.join("test_aresg2_trios_frequency_sweep_tts.txt"))
    ms = TriosRheoReader().data_from_filepath(path, create_composite_datasets=True)
    return ExperimentSet(ms.measurements)


@pytest.fixture()
def saos():
    omega = np.logspace(-2, 2, 40)
    model = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
    return Protocols.saos(model, omega)


def test_cole_cole_uses_storage_and_loss_units(tts_experiment):
    _, ax = plot_cole_cole(tts_experiment)
    dataset = tts_experiment.measurements[0].datasets[0]
    assert ax.get_xlabel() == axis_label(dataset, "storage modulus")
    assert ax.get_ylabel() == axis_label(dataset, "loss modulus")
    assert "(" in ax.get_xlabel()
    assert ax.get_xscale() == ax.get_yscale() == "log"


def test_vgp_axes(tts_experiment):
    _, ax = plot_vgp(tts_experiment)
    assert ax.get_xlabel() == "complex modulus (Pa)"
    assert ax.get_ylabel() == "tan(delta)"
    assert ax.get_xscale() == "log"


def test_pair_plot_legend_or_colorbar(tts_experiment):
    fig, ax = plot_vgp(tts_experiment, legend=True)
    assert ax.get_legend() is not None
    fig2, ax2 = plot_vgp(tts_experiment, legend=False)
    assert ax2.get_legend() is None
    assert len(fig2.axes) == 2


def test_pair_plot_existing_axes_and_save(tts_experiment, tmp_path):
    _, ax = plt.subplots()
    _, out = plot_cole_cole(tts_experiment, ax=ax, title="x", save=str(tmp_path / "a" / "cc.png"))
    assert out is ax
    assert ax.get_title() == "x"
    assert (tmp_path / "a" / "cc.png").exists()


def test_pair_plot_missing_arrays_raises(saos):
    with pytest.raises(ValueError):
        plot_vgp(saos, x_name="no such array")


def test_plot_model_fit_oscillation(saos, tmp_path):
    experiment = ExperimentSet([saos])
    fit = FitModel(model="custom", model_name="MultiModeMaxwell", measurement_index=0)
    fit(experiment)
    fig, ax = plot_model_fit(fit, save=str(tmp_path / "fit.png"))
    assert ax.get_xlabel() == "angular frequency (rad/s)"
    assert ax.get_ylabel() == "modulus (Pa)"
    assert ax.get_xscale() == "log"
    assert (tmp_path / "fit.png").exists()

    fig, axes = plot_model_fit(fit, show_residuals=True)
    assert len(fig.axes) == 2


def test_plot_model_fit_requires_fit():
    fit = FitModel(model="custom", model_name="MultiModeMaxwell")
    with pytest.raises(AttributeError):
        plot_model_fit(fit)


@pytest.fixture()
def spectrum_result():
    t = np.logspace(-2, 1.5, 80)
    model = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
    meas = Protocols.stress_relaxation(model, t)
    out = RelaxationSpectrum(
        domain="time", relaxation_name="relaxation modulus", ns=60, plateau=False
    )(meas)
    return out, out.details["relaxation_spectrum_result"]


def test_plot_relaxation_spectrum(spectrum_result, tmp_path):
    out, result = spectrum_result
    fig, ax = plot_relaxation_spectrum(result, source=out, save=str(tmp_path / "s.png"))
    assert "(s)" in ax.get_xlabel()
    assert "(Pa)" in ax.get_ylabel()
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert (tmp_path / "s.png").exists()


def test_plot_spectrum_diagnostics(spectrum_result):
    _, result = spectrum_result
    fig, axes = plot_spectrum_diagnostics(result)
    assert len(axes) == 3


@pytest.mark.parametrize(
    ("unit", "label", "shift"), [("C", "temperature (°C)", 0.0), ("K", "temperature (K)", 273.15)]
)
@pytest.mark.parametrize("name", ["WLF", "Arrhenius"])
def test_plot_shift_factors_temperature_label(name, unit, label, shift):
    temperatures = np.linspace(20.0, 100.0, 9) + shift
    shift_factors = 10.0 ** (
        -8.0 * (temperatures - temperatures[0]) / (100.0 + temperatures - temperatures[0])
    )
    fit = ShiftFactorModel(model=name, temperature_unit=unit)((temperatures, shift_factors))
    fig, ax = plot_shift_factors(fit)
    assert ax.get_xlabel() == label
    assert len(ax.lines) >= 1


def test_plot_shift_factors_data_color():
    temperatures = np.linspace(20.0, 100.0, 9)
    shift_factors = 10.0 ** (-8.0 * (temperatures - 20.0) / (100.0 + temperatures - 20.0))
    fit = ShiftFactorModel(model="WLF")((temperatures, shift_factors))
    _, ax = plot_shift_factors(fit, data_color="tab:orange")
    assert ax.lines[0].get_color() == "tab:orange"


@pytest.mark.parametrize(("show_gp", "n_panels"), [(True, 3), (False, 2)])
def test_plot_master_curve_labels_from_dataset(tts_experiment, show_gp, n_panels):
    master = AutomatedMasterCurve(y="storage modulus")(tts_experiment)
    curve = master.datasets[0]
    fig, axes = plot_master_curve(master, show_gp=show_gp)
    assert len(axes) == n_panels
    assert axes[0].get_ylabel() == axis_label(curve, "storage modulus")
    assert axes[-1].get_xlabel() == axis_label(
        curve, "angular frequency", text="reduced angular frequency"
    )
    colorbar_ax = fig.axes[-1]
    assert colorbar_ax.get_ylabel() == "temperature (°C)"
    assert colorbar_ax.get_ylim()[0] > colorbar_ax.get_ylim()[1]  # lowest state on top


def test_plot_master_curve_highest_state_on_top(tts_experiment):
    master = AutomatedMasterCurve(y="storage modulus")(tts_experiment)
    fig, _ = plot_master_curve(master, show_gp=False, lowest_on_top=False)
    colorbar_ax = fig.axes[-1]
    assert colorbar_ax.get_ylim()[0] < colorbar_ax.get_ylim()[1]
