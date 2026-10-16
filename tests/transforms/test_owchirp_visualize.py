"""Tests for OWChirp visualization."""

from __future__ import annotations

import os

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.visualization import OWChirpVisualizer, axis_label


@pytest.fixture()
def visualizer(datadir):
    filepath = datadir / "test_dhr30_trios_owchirp.txt"
    mset = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)
    processed = OWChirpAnalysis()(mset)
    viz = OWChirpVisualizer(processed)
    yield viz
    plt.close("all")


def test_extraction_lengths(visualizer):
    """Every extracted key holds one array per wave."""
    data = visualizer.data
    assert visualizer.n_waves >= 1
    for key, arrays in data.items():
        assert len(arrays) == visualizer.n_waves, key


def test_resolve_steps(visualizer):
    """The step selector handles all the documented forms."""
    n = visualizer.n_waves
    assert visualizer._resolve_steps("all") == list(range(n))
    assert visualizer._resolve_steps(0) == [0]
    assert visualizer._resolve_steps([2, 1, 1]) == sorted({1, 2})
    fml = visualizer._resolve_steps("first-middle-last")
    assert fml == sorted({0, n // 2, n - 1})
    with pytest.raises(ValueError):
        visualizer._resolve_steps("nonsense")


def test_plot_wave_and_waves(visualizer):
    """plot_wave returns three axes; plot_waves honours the selector."""
    _, axes = visualizer.plot_wave(0)
    assert len(axes) == 3

    figs, _ = visualizer.plot_waves(steps="first-middle-last")
    assert len(figs) == len(visualizer._resolve_steps("first-middle-last"))

    _, fourier_axes = visualizer.plot_fourier(step=0)
    assert len(fourier_axes) == 4


def test_plot_labels_use_dataset_units(visualizer):
    """Plots use labels derived from the source datasets."""
    freq_ds, time_ds = visualizer._meas[0].datasets[:2]
    _, (ax_mod, ax_strain, ax_stress) = visualizer.plot_wave(0)
    assert ax_mod.get_xlabel() == axis_label(freq_ds, "angular frequency")
    assert ax_mod.get_ylabel() == axis_label(freq_ds, "storage modulus", text="modulus")
    assert ax_stress.get_xlabel() == axis_label(time_ds, "adjusted time", text="time")
    assert ax_stress.get_ylabel() == axis_label(time_ds, "adjusted stress", text="stress")
    assert ax_mod.get_title().startswith("Arbitrary Wave 1")

    _, axes = visualizer.plot_fourier(step=0)
    assert axes[1].get_xlabel() == axis_label(freq_ds, "angular frequency")


def test_save_frames(visualizer, tmp_path):
    """save_frames writes one PNG per wave."""
    out = tmp_path / "frames"
    paths = visualizer.save_frames(str(out))
    assert len(paths) == visualizer.n_waves
    assert all(os.path.exists(p) for p in paths)
