"""Mutation number against time, per frequency and averaged over frequency."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.mastercurve import _add_state_colorbar
from hermes_rheo.visualization.units import axis_label

MUTATION_NUMBER = "mutation number"


def _labels(absolute: bool) -> tuple[str, str]:
    """y labels of the per-frequency and frequency-averaged panels."""
    mu = r"|\mathrm{Mu}|" if absolute else r"\mathrm{Mu}"
    return (
        f"Mutation Number ${mu}$",
        rf"Frequency-averaged Mutation Number $\overline{{{mu}}}$",
    )


def plot_mutation_number(
    mutation: Any,
    *,
    show_all_frequencies: bool = False,
    color_by: str | None = None,
    cmap: str = "viridis",
    threshold: float | None = 0.15,
    ylim: tuple[float, float] | None = None,
    logy: bool = False,
    figsize: tuple[float, float] | None = None,
    save: str | os.PathLike | None = None,
    show: bool = False,
) -> tuple[matplotlib.figure.Figure, Any]:
    """Plot the mutation number against time.

    Parameters
    ----------
    mutation : MutationNumberMeasurementSet
        Output of :class:`~hermes_rheo.transforms.mutation_number.MutationNumber`.
    show_all_frequencies : bool, default False
        Add a left panel with Mu at every frequency next to the panel of the
        frequency-averaged mean +/- standard deviation.
    color_by : str, optional
        Data array that sets the marker colour, e.g. ``"temperature"`` for
        a TORCh ramp; a colour bar shows its range. Default: colour by time,
        without a colour bar.
    cmap : str, default "viridis"
        Matplotlib colormap.
    threshold : float or None, default 0.15
        Quasi-static limit drawn as a dashed grey line; None to omit.
    ylim : tuple of float, optional
        y-axis limits.
    logy : bool, default False
        Logarithmic y-axis (useful with ``MutationNumber(absolute=True)``).
    figsize : tuple of float, optional
        Figure size; defaults to the style module sizes.
    save : path-like, optional
        Save the figure to this path.
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or tuple of matplotlib.axes.Axes
        ``(ax_all, ax_mean)`` if `show_all_frequencies`, else ``ax_mean``.
    """
    datasets = [m.datasets[0] for m in mutation.measurements]
    if not datasets:
        raise ValueError("No mutation-number data to plot.")
    first = datasets[0]
    colour_name = color_by or c.TIME
    if c.array_by_name(first, colour_name) is None:
        raise ValueError(f"No {colour_name!r} data array to colour by.")

    time = np.concatenate([c.array_by_name(d, c.TIME) for d in datasets])
    mu = np.concatenate([c.array_by_name(d, MUTATION_NUMBER) for d in datasets])
    colour = np.concatenate([c.array_by_name(d, colour_name) for d in datasets])

    # Frequency-averaged mean and standard deviation at each time.
    times, inverse = np.unique(time, return_inverse=True)
    counts = np.bincount(inverse)
    mean = np.bincount(inverse, mu) / counts
    std = np.sqrt(np.bincount(inverse, (mu - mean[inverse]) ** 2) / counts)
    mean_colour = np.bincount(inverse, colour) / counts

    colormap = plt.get_cmap(cmap)
    norm = plt.Normalize(colour.min(), colour.max())
    absolute = bool(mutation.measurements[0].details.get("absolute", False))
    all_label, mean_label = _labels(absolute)

    if show_all_frequencies:
        fig, (ax_all, ax_mean) = plt.subplots(
            1, 2, figsize=figsize or style.FIGSIZE_WIDE, layout="constrained"
        )
        ax_all.scatter(time, mu, c=colour, cmap=colormap, norm=norm, s=style.MARKERSIZE**2)
        panels = [(ax_all, all_label), (ax_mean, mean_label)]
    else:
        fig, ax_mean = plt.subplots(figsize=figsize or style.FIGSIZE, layout="constrained")
        panels = [(ax_mean, mean_label)]

    ax_mean.errorbar(
        times,
        mean,
        yerr=std,
        fmt="none",
        ecolor=style.COLOR_GUIDE,
        elinewidth=style.LINEWIDTH_THIN,
        capsize=2,
        zorder=1,
    )
    ax_mean.scatter(
        times, mean, c=mean_colour, cmap=colormap, norm=norm, s=style.MARKERSIZE**2, zorder=2
    )

    for ax, ylabel in panels:
        if threshold is not None:
            ax.axhline(
                threshold,
                color=style.COLOR_GUIDE,
                linestyle="--",
                linewidth=style.LINEWIDTH_THIN,
                label=rf"$\mathrm{{Mu}} = {threshold:g}$",
            )
            c.add_legend(ax)
        c.decorate_axes(ax, xlabel=axis_label(first, c.TIME), ylabel=ylabel, logy=logy, ylim=ylim)

    if color_by is not None:
        axes = [ax for ax, _ in panels]
        _add_state_colorbar(fig, axes, colormap, norm, first, color_by, lowest_on_top=False)

    c.finish_figure(fig, save=save, show=show)
    return fig, ((ax_all, ax_mean) if show_all_frequencies else ax_mean)
