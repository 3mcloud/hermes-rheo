"""Tests for :class:`hermes_rheo.visualization.OWChirpVisualizer`.

These tests lock in that the high-level OWChirp visualiser:

1. Extracts one array set per wave from a processed OWChirp experiment.
2. Resolves the ``steps`` selector (``"first-middle-last"``, ints, ...).
3. Produces figures for :meth:`plot_wave`, :meth:`plot_waves` and
   :meth:`plot_fourier` without error.
4. Saves one PNG frame per wave with :meth:`save_frames`.
"""
from __future__ import annotations

import os
import shutil

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.visualization import OWChirpVisualizer


@pytest.fixture()
def datadir(tmpdir, request):
    module_dir = os.path.dirname(request.module.__file__)
    test_dir = os.path.abspath(os.path.join(module_dir, "..", "fileio", "test_data"))
    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir


@pytest.fixture()
def visualizer(datadir):
    filepath = datadir.join("test_owchirp_DHR.txt")
    mset = TriosRheoReader().data_from_filepath(
        str(filepath), create_composite_datasets=True
    )
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
    fig, axes = visualizer.plot_wave(0)
    assert len(axes) == 3

    figs, axes_list = visualizer.plot_waves(steps="first-middle-last")
    assert len(figs) == len(visualizer._resolve_steps("first-middle-last"))


def test_plot_fourier(visualizer):
    """plot_fourier returns a 2x2 set of axes."""
    fig, axes = visualizer.plot_fourier(step=0)
    assert len(axes) == 4


def test_save_frames(visualizer, tmp_path):
    """save_frames writes one PNG per wave."""
    out = tmp_path / "frames"
    paths = visualizer.save_frames(str(out))
    assert len(paths) == visualizer.n_waves
    assert all(os.path.exists(p) for p in paths)
