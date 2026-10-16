"""Tests for the global plotting style."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from piblin.data import OneDimensionalDataset  # noqa: E402

from hermes_rheo.visualization import reset_style, use_science_style  # noqa: E402

pytest.importorskip("scienceplots")


def _visible_spines(dataset):
    _, ax = dataset.visualize()
    visible = {side for side, spine in ax.spines.items() if spine.get_visible()}
    plt.close("all")
    return visible


def test_science_style_boxes_piblin_axes():
    dataset = OneDimensionalDataset.create(x_values=np.arange(3.0), y_values=np.arange(3.0))
    try:
        use_science_style()
        assert _visible_spines(dataset) == {"top", "right", "bottom", "left"}
    finally:
        reset_style()
    assert _visible_spines(dataset) == {"bottom", "left"}  # piblin default restored
