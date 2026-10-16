r"""Plots for the relaxation spectrum of :class:`RelaxationSpectrum`.

The relaxation spectrum :math:`h(s)` is the distribution of relaxation
strengths over relaxation time :math:`s`, with the relaxation modulus
:math:`G(t) = \int h(s)\, e^{-t/s}\, d\ln s`. The continuous spectrum is
obtained by regularised inversion of the data with the regularisation
parameter :math:`\lambda` chosen by Bayesian evidence, and a discrete set
of Maxwell modes :math:`(g_i, \tau_i)` is selected with the Akaike
criterion (pyReSpect; Takeh and Shanbhag, Appl. Rheol. 23, 24628 (2013);
Shanbhag, Macromol. Theory Simul. 28, 1900005 (2019)).

Examples
--------
>>> from hermes_rheo.transforms.relaxation_spectrum import RelaxationSpectrum
>>> out = RelaxationSpectrum(domain="frequency")(measurement)  # doctest: +SKIP
>>> fig, ax = plot_relaxation_spectrum(
...     out.details["relaxation_spectrum_result"], source=out
... )  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.units import get_unit, unit_suffix

__all__ = [
    "plot_relaxation_spectrum",
    "plot_spectrum_diagnostics",
]


def _resolve_units(source: Any | None) -> tuple[str, str]:
    """Return ``(time_unit, modulus_unit)`` read from *source*."""
    if source is None:
        return c.FALLBACK_TIME_UNIT, c.FALLBACK_MODULUS_UNIT
    time_unit = c.FALLBACK_TIME_UNIT
    for name in c.RELAXATION_TIME_NAMES:
        if unit := get_unit(source, name):
            time_unit = unit
            break
    modulus_unit = c.FALLBACK_MODULUS_UNIT
    for name in c.SPECTRUM_NAMES:
        if unit := get_unit(source, name):
            modulus_unit = unit
            break
    return time_unit, modulus_unit


def plot_relaxation_spectrum(
    result: Any,
    ax: matplotlib.axes.Axes | None = None,
    *,
    show_discrete: bool = True,
    show_continuous: bool = True,
    show_uncertainty: bool = True,
    source: Any | None = None,
    label_continuous: str | None = None,
    label_discrete: str | None = None,
    color_continuous: str = style.COLOR_LOSS,
    color_discrete: str = style.COLOR_STORAGE,
    unit_style: str = "paren",
    legend_loc: str = "best",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save: str | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""Plot the continuous and discrete relaxation spectra on log-log axes.

    Parameters
    ----------
    result : RelaxationSpectrumResult
        Result of
        :class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`,
        stored in ``details["relaxation_spectrum_result"]``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    show_discrete : bool, default True
        Draw the discrete modes :math:`(\tau_i, g_i)` as stems.
    show_continuous : bool, default True
        Draw the continuous spectrum :math:`h(s)`.
    show_uncertainty : bool, default True
        Shade the band :math:`\exp(\ln h \pm dH)`. Only available for
        frequency-domain results; skipped if ``result.dH`` is ``None``.
    source : Measurement, ExperimentSet or dataset, optional
        Object whose data array units label the axes. Defaults to
        ``s`` for time and ``Pa`` for modulus.
    label_continuous, label_discrete : str, optional
        Legend labels.
    color_continuous, color_discrete : str
        Colours of the continuous and discrete spectra.
    unit_style : {"paren", "bracket"}, default "paren"
        Brackets around the unit in axis labels.
    legend_loc : str, default "best"
        Legend location.
    title, xlabel, ylabel : str, optional
        Override the default title and labels.
    save : str, optional
        Path to write the figure to.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = c.get_axes(ax, style.FIGSIZE)
    time_unit, modulus_unit = _resolve_units(source)
    plotted = False

    if show_continuous and result.s is not None and result.h is not None:
        s = np.asarray(result.s, dtype=float)
        h = np.asarray(result.h, dtype=float)
        ax.plot(
            s,
            h,
            "-",
            color=color_continuous,
            linewidth=style.LINEWIDTH,
            label=label_continuous or f"continuous ({result.domain})",
        )
        dh = getattr(result, "dH", None)
        if show_uncertainty and dh is not None and np.shape(dh) == s.shape:
            dh = np.asarray(dh, dtype=float)
            ax.fill_between(
                s,
                np.exp(np.log(h) - dh),
                np.exp(np.log(h) + dh),
                color=color_continuous,
                alpha=0.2,
                linewidth=0,
                label=r"$\pm dH$",
            )
        plotted = True

    if show_discrete and result.tau is not None and result.g is not None and result.tau.size > 0:
        tau = np.asarray(result.tau, dtype=float)
        g = np.asarray(result.g, dtype=float)
        markerline, stemlines, _ = ax.stem(
            tau,
            g,
            linefmt="-",
            markerfmt="o",
            basefmt=" ",
            label=label_discrete or f"discrete N={tau.size}",
        )
        markerline.set(
            markerfacecolor=color_discrete,
            markeredgecolor=color_discrete,
            markersize=style.MARKERSIZE + 2,
        )
        stemlines.set(color=color_discrete, linewidth=style.LINEWIDTH_THIN)
        plotted = True

    if plotted:
        c.decorate_axes(
            ax,
            xlabel=xlabel or f"$s${unit_suffix(time_unit, style=unit_style)}",
            ylabel=ylabel or f"$h(s)$, $g_i${unit_suffix(modulus_unit, style=unit_style)}",
            title=title,
            logx=True,
            logy=True,
        )
        c.add_legend(ax, loc=legend_loc)
    c.finish_figure(fig, save=save)
    return fig, ax


def _unavailable(ax: matplotlib.axes.Axes, title: str) -> None:
    ax.set_title(f"{title} (unavailable)", fontsize=style.TITLE_SIZE)
    ax.axis("off")


def plot_spectrum_diagnostics(
    result: Any,
    axes: np.ndarray | None = None,
    *,
    save: str | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    r"""Plot the three regularisation diagnostics of a spectrum fit.

    The panels are, from left to right:

    1. L-curve: residual norm :math:`\rho(\lambda)` against solution norm
       :math:`\eta(\lambda)`, with the selected :math:`\lambda_C` marked.
    2. Log evidence :math:`\log p(\lambda)` against :math:`\lambda`.
    3. Akaike criterion against the base-weight distance of the discrete
       modes, with the optimum number of modes :math:`N_\mathrm{opt}`.

    Panels whose diagnostics are missing from *result* are left blank.

    Parameters
    ----------
    result : RelaxationSpectrumResult
        Result of ``RelaxationSpectrum``.
    axes : array of matplotlib.axes.Axes, optional
        Three axes. A new 1x3 figure is created if omitted.
    save : str, optional
        Path to write the figure to.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes, shape (3,)
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=style.FIGSIZE_WIDE, constrained_layout=True)
    else:
        axes = np.asarray(axes).ravel()
        fig = axes[0].get_figure()
    ax_l, ax_p, ax_a = axes[:3]
    lam_c = getattr(result, "lam_C", None)
    lam = getattr(result, "lam", None)
    lam_label = rf"$\lambda_C = {float(lam_c):.2e}$" if lam_c is not None else None

    rho, eta = getattr(result, "rho", None), getattr(result, "eta", None)
    if rho is not None and eta is not None:
        rho = np.asarray(rho, dtype=float)
        eta = np.asarray(eta, dtype=float)
        ax_l.plot(
            rho,
            eta,
            "-o",
            color=style.COLOR_LOSS,
            markersize=style.MARKERSIZE - 1,
            linewidth=style.LINEWIDTH,
        )
        if lam is not None and lam_c is not None:
            idx = int(np.argmin(np.abs(np.log(np.asarray(lam)) - np.log(float(lam_c)))))
            ax_l.plot(
                [rho[idx]],
                [eta[idx]],
                "o",
                color=style.COLOR_MARK,
                markersize=style.MARKERSIZE + 3,
                label=lam_label,
            )
            c.add_legend(ax_l)
        c.decorate_axes(
            ax_l,
            xlabel=r"$\rho(\lambda)$",
            ylabel=r"$\eta(\lambda)$",
            title="L-curve",
            logx=True,
            logy=True,
        )
    else:
        _unavailable(ax_l, "L-curve")

    log_p = getattr(result, "log_P", None)
    if lam is not None and log_p is not None:
        ax_p.plot(
            np.asarray(lam, dtype=float),
            np.asarray(log_p, dtype=float),
            "-o",
            color=style.COLOR_SHIFT,
            markersize=style.MARKERSIZE - 1,
            linewidth=style.LINEWIDTH,
        )
        if lam_c is not None:
            ax_p.axvline(float(lam_c), color=style.COLOR_MARK, linestyle="--", label=lam_label)
            c.add_legend(ax_p)
        c.decorate_axes(
            ax_p,
            xlabel=r"$\lambda$",
            ylabel=r"$\log p(\lambda)$",
            title="Bayesian evidence",
            logx=True,
        )
    else:
        _unavailable(ax_p, r"$\log p(\lambda)$")

    wt, aic = getattr(result, "wt_base", None), getattr(result, "AIC_bst", None)
    if wt is not None and aic is not None:
        wt = np.asarray(wt, dtype=float)
        aic = np.asarray(aic, dtype=float)
        ax_a.plot(
            wt,
            aic,
            "-o",
            color=style.COLOR_STRESS,
            markersize=style.MARKERSIZE - 1,
            linewidth=style.LINEWIDTH,
        )
        idx = int(np.argmin(aic))
        ax_a.plot(
            [wt[idx]],
            [aic[idx]],
            "o",
            color=style.COLOR_MARK,
            markersize=style.MARKERSIZE + 3,
            label=rf"$N_\mathrm{{opt}} = {int(result.N_opt)}$",
        )
        c.add_legend(ax_a)
        c.decorate_axes(
            ax_a,
            xlabel="base-weight distance",
            ylabel="AIC",
            title=f"AIC scan ({result.domain})",
        )
    else:
        _unavailable(ax_a, "AIC scan")

    c.finish_figure(fig, save=save)
    return fig, axes
