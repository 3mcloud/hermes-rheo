"""Array names and helpers shared by the plotting functions (private).

The names below are the ``data_array_names`` written by the file readers and
by :mod:`hermes_rheo.lve_models`. They are the only place in the plotting code
where a quantity is identified by name; units and axis labels are always read
from the dataset.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization import style

ANGULAR_FREQUENCY = "angular frequency"
STORAGE_MODULUS = "storage modulus"
LOSS_MODULUS = "loss modulus"
COMPLEX_MODULUS = "complex modulus"
COMPLEX_VISCOSITY = "complex viscosity"
TAN_DELTA = "tan(delta)"
TIME = "time"
STEP_TIME = "step time"
RELAXATION_MODULUS = "relaxation modulus"
CREEP_COMPLIANCE = "creep compliance"
TEMPERATURE = "temperature"

ADJUSTED_TIME = "adjusted time"
ADJUSTED_STRAIN = "adjusted strain"
ADJUSTED_STRESS = "adjusted stress"
FOURIER_STRAIN = "fourier_transform_strain"
FOURIER_STRESS = "fourier_transform_stress"

#: Prefix added by :class:`~hermes_rheo.transforms.fit_model.FitModel` to the
#: names of the fitted arrays.
FITTED_PREFIX = "fitted "

#: Names that can hold the independent (time or relaxation-time) axis of a
#: relaxation spectrum, and its amplitude, in order of preference.
RELAXATION_TIME_NAMES = ("relaxation time", TIME, STEP_TIME)
SPECTRUM_NAMES = ("h", STORAGE_MODULUS, RELAXATION_MODULUS, "modulus")

FALLBACK_TIME_UNIT = "s"
FALLBACK_MODULUS_UNIT = "Pa"

_TEMPERATURE_UNIT_SYMBOLS = {"C": "°C", "K": "K"}


def array_by_name(dataset: Any, name: str) -> np.ndarray | None:
    """Return the data array called *name*, or ``None`` if absent."""
    names = list(getattr(dataset, "data_array_names", None) or [])
    if name not in names:
        return None
    return np.asarray(dataset.data_arrays[names.index(name)])


def temperature_unit_symbol(unit: str) -> str:
    """Map ``"C"`` to ``"°C"``; other units are returned unchanged."""
    return _TEMPERATURE_UNIT_SYMBOLS.get(unit, unit)


def get_axes(
    ax: matplotlib.axes.Axes | None,
    figsize: tuple[float, float] = style.FIGSIZE,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Return ``(fig, ax)``, creating a figure when *ax* is ``None``."""
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.get_figure(), ax


def decorate_axes(
    ax: matplotlib.axes.Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    logx: bool = False,
    logy: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Apply labels, scales, limits and the common tick style to *ax*."""
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=style.LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=style.LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=style.TITLE_SIZE)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def add_legend(ax: matplotlib.axes.Axes, outside: bool = False, **kwargs: Any) -> Any:
    """Add a legend inside the axes, or outside to the upper right."""
    options: dict[str, Any] = {"fontsize": style.LEGEND_SIZE, "frameon": False, "loc": "best"}
    if outside:
        options.update(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            title_fontsize=style.LEGEND_SIZE,
        )
    options.update(kwargs)
    return ax.legend(**options)


def finish_figure(
    fig: matplotlib.figure.Figure,
    *,
    save: str | os.PathLike | None = None,
    dpi: int = 150,
    show: bool = False,
) -> None:
    """Tighten the layout, then optionally save and show *fig*."""
    if not fig.get_constrained_layout():
        fig.tight_layout()
    if save is not None:
        os.makedirs(os.path.dirname(os.fspath(save)) or ".", exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
