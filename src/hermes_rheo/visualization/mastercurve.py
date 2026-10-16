"""Plot a time–temperature (or other state) superposition master curve."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.units import axis_label, get_unit

_PANEL_WIDTH = 3.2
_PANEL_HEIGHT = 3.0
_COLORBAR_WIDTH = 0.8


def plot_master_curve(
    master: Any,
    cmap: str | matplotlib.colors.Colormap = style.COLORMAP,
    show_gp: bool = True,
    markersize: float = 3.0,
    show: bool = False,
    *,
    lowest_on_top: bool = True,
    figsize: tuple[float, float] | None = None,
    save: str | os.PathLike | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    r"""Plot the measured data, the Gaussian-process fits and the master curve.

    The panels show, side by side, the data before shifting, the data with
    the Gaussian-process interpolants (mean :math:`\pm` one standard
    deviation) used by the superposition, and the shifted master curve
    against the reduced abscissa :math:`a\,x`. Each state is coloured by
    its value; a single colour bar labels the state range. Names and units are read from the master-curve dataset.

    Parameters
    ----------
    master : MasterCurveMeasurement
        Output of
        :class:`~hermes_rheo.transforms.automated_mastercurve.AutomatedMasterCurve`,
        or one entry of a ``CompositeMasterCurve`` result.
    cmap : str or Colormap, default "coolwarm"
        Colormap mapping the state value to colour.
    show_gp : bool, default True
        Include the Gaussian-process panel.
    markersize : float, default 3.0
        Marker size of the data points.
    show : bool, default False
        Call :func:`matplotlib.pyplot.show` before returning.
    lowest_on_top : bool, default True
        Put the lowest state at the top of the colour bar (cold on top for
        temperature). Set ``False`` for the highest state on top.
    figsize : tuple of float, optional
        Figure size. Scales with the number of panels by default.
    save : str or path-like, optional
        Path to write the figure to.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
        The panels, left to right.

    Examples
    --------
    >>> master = AutomatedMasterCurve(y="storage modulus")(experiment)  # doctest: +SKIP
    >>> fig, axes = plot_master_curve(master)  # doctest: +SKIP
    """
    fit = master.master_curve
    curve = master.datasets[0]
    x_name, y_name, state_name = (master.details[k] for k in ("x", "y", "state"))
    states = np.asarray(fit.states, dtype=float)

    colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = matplotlib.colors.Normalize(states.min(), states.max())
    colors = colormap(norm(states))

    n_panels = 3 if show_gp else 2
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize or (_PANEL_WIDTH * n_panels + _COLORBAR_WIDTH, _PANEL_HEIGHT),
        sharey=True,
        layout="constrained",
    )
    data_axes = list(axes[:-1])
    master_ax = axes[-1]
    marker = {"marker": "o", "linestyle": "none", "markersize": markersize}

    for k, color in enumerate(colors):
        x, y = np.exp(fit.xdata[k]), np.exp(fit.ydata[k])
        for ax in data_axes:
            ax.plot(x, y, color=color, **marker)
        master_ax.plot(
            np.exp(fit.xtransformed[k]), np.exp(fit.ytransformed[k]), color=color, **marker
        )
        if show_gp:
            log_x = np.linspace(fit.xdata[k].min(), fit.xdata[k].max(), 100)
            mean, std = fit.gps[k].predict(log_x.reshape(-1, 1), return_std=True)
            axes[1].plot(np.exp(log_x), np.exp(mean), color=color, linewidth=1.0)
            axes[1].fill_between(
                np.exp(log_x), np.exp(mean - std), np.exp(mean + std), color=color, alpha=0.2
            )

    titles = ["measured", "Gaussian-process fits"][: len(data_axes)] + ["master curve"]
    for ax, title in zip(axes, titles, strict=True):
        is_master = ax is master_ax
        c.decorate_axes(
            ax,
            xlabel=axis_label(curve, x_name, text=f"reduced {x_name}" if is_master else None),
            ylabel=axis_label(curve, y_name) if ax is axes[0] else None,
            title=title,
            logx=True,
            logy=True,
        )

    _add_state_colorbar(fig, axes, colormap, norm, curve, state_name, lowest_on_top)
    c.finish_figure(fig, save=save, show=show)
    return fig, axes


def _add_state_colorbar(
    fig: matplotlib.figure.Figure,
    axes: np.ndarray,
    colormap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
    curve: Any,
    state_name: str,
    lowest_on_top: bool,
) -> None:
    """Add a colour bar labelled with the extremes of the state range."""
    unit = c.temperature_unit_symbol(get_unit(curve, state_name))
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=list(axes),
        fraction=0.04,
        pad=0.02,
    )
    colorbar.set_ticks([norm.vmin, norm.vmax])
    colorbar.set_ticklabels([f"{norm.vmin:.0f}", f"{norm.vmax:.0f}"])
    colorbar.minorticks_off()
    if lowest_on_top:
        colorbar.ax.invert_yaxis()
    colorbar.set_label(axis_label(curve, state_name, unit=unit), fontsize=style.LABEL_SIZE)
