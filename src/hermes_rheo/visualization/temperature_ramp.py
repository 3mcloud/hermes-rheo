"""Glass transition from the storage modulus of a temperature ramp."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.units import axis_label, get_unit, unit_suffix

STORAGE_MODULUS = "storage modulus"
TEMPERATURE = "temperature"
GUIDE_LW = 1.5
GUIDE_COLOR = "0.5"


def plot_tg_onset(
    ramp: Any,
    *,
    figsize: tuple[float, float] = (5.5, 5.5),
    colors: tuple[str, str] = ("darkblue", "k"),
    xlim: tuple[float, float] | None = None,
    save: str | os.PathLike | None = None,
    show: bool = False,
) -> tuple[matplotlib.figure.Figure, tuple[Any, Any]]:
    """Plot how the onset Tg is obtained from the storage modulus.

    Top: G' with the smoothed curve, the tangent fitted to the glassy plateau
    (shaded range), the tangent at the inflection and their intersection, the
    onset Tg. Bottom: d log10 G'/dT, whose minimum is the inflection.

    Parameters
    ----------
    ramp : TemperatureRampMeasurement
        Output of ``TemperatureRampAnalysis(tg_method="storage_modulus")`` or
        ``tg_method="both"``.
    figsize : tuple of float, default (5.5, 5.5)
        Figure size.
    colors : tuple of str, default ("darkblue", "k")
        Colours of the data and of the fitted tangents; guide lines are gray.
    xlim : tuple of float, optional
        Temperature limits, e.g. to zoom on the construction; default: the
        whole analysis range.
    save : path-like, optional
        Save the figure to this path.
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of matplotlib.axes.Axes
        ``(ax_modulus, ax_slope)``.
    """
    onset = ramp.onset
    if onset is None:
        raise ValueError("No onset construction: use tg_method='storage_modulus' or 'both'.")
    ds = ramp.dataset
    data_color, line_color = colors
    temperature = c.array_by_name(ds, TEMPERATURE)
    modulus = c.array_by_name(ds, STORAGE_MODULUS)
    t_unit = get_unit(ds, TEMPERATURE)

    fig, (ax_g, ax_d) = plt.subplots(2, 1, sharex=True, figsize=figsize, height_ratios=(2, 1))
    t = onset.temperature
    ax_g.semilogy(
        temperature, modulus, "o", color=data_color, ms=4, alpha=0.35, mew=0, label="data"
    )
    ax_g.semilogy(
        t, 10**onset.log_modulus, "-", color=data_color, lw=style.LINEWIDTH, label="smoothed"
    )
    ax_g.axvspan(*onset.glassy_range, color="0.9", lw=0, label="glassy fit range")
    for (slope, intercept), ls in ((onset.glassy_line, "--"), (onset.inflection_line, ":")):
        ax_g.semilogy(t, 10 ** (slope * t + intercept), ls, color=line_color, lw=GUIDE_LW)
    ax_g.plot(
        onset.tg_onset,
        10 ** (onset.glassy_line[0] * onset.tg_onset + onset.glassy_line[1]),
        "o",
        ms=8,
        color=line_color,
        label=f"onset $T_g$ = {onset.tg_onset:.1f} {t_unit}",
    )

    ax_d.plot(t, onset.slope, color=data_color, lw=style.LINEWIDTH)
    ax_d.plot(
        onset.t_inflection,
        onset.slope.min(),
        "o",
        ms=8,
        mew=GUIDE_LW,
        mfc="none",
        color=line_color,
        label=f"inflection = {onset.t_inflection:.1f} {t_unit}",
    )
    for ax in (ax_g, ax_d):
        ax.axvline(onset.tg_onset, color=GUIDE_COLOR, lw=GUIDE_LW, ls="--")
        ax.axvline(onset.t_inflection, color=GUIDE_COLOR, lw=GUIDE_LW, ls=":")

    if xlim is None:
        xlim = (t[0], t[-1])
    shown = (t >= xlim[0]) & (t <= xlim[1])
    ax_g.set_ylim(
        10 ** onset.log_modulus[shown].min() / 2, 10 ** onset.log_modulus[shown].max() * 2
    )
    ax_d.set_ylim(
        onset.slope[shown].min() * 1.15,
        max(0.0, onset.slope[shown].max()) + 0.1 * abs(onset.slope[shown].min()),
    )
    c.decorate_axes(ax_g, ylabel=axis_label(ds, STORAGE_MODULUS), xlim=xlim)
    c.decorate_axes(
        ax_d,
        xlabel=axis_label(ds, TEMPERATURE),
        ylabel=f"d log$_{{10}}$({STORAGE_MODULUS})/dT" + unit_suffix(f"1/{t_unit}"),
    )
    ax_d.yaxis.label.set_size(style.LABEL_SIZE * 0.8)
    for ax in (ax_g, ax_d):
        c.add_legend(ax, loc="upper right", fontsize=style.LEGEND_SIZE - 1)
    c.finish_figure(fig, save=save, show=show)
    return fig, (ax_g, ax_d)
