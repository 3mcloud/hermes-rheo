"""Plotting helpers for the pyReSpect relaxation-spectrum transform.

Two public helpers work directly on a
:class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrumResult`
returned by
:class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`:

* :func:`plot_relaxation_spectrum` --- CRS :math:`h(s)` curve with
  optional DRS :math:`g_i` stems overlayed, and shaded uncertainty band
  when :math:`dH` is available (frequency branch only).
* :func:`plot_spectrum_diagnostics` --- three-panel diagnostic:
  L-curve :math:`\\rho(\\lambda)-\\eta(\\lambda)`, log evidence
  :math:`\\log p(\\lambda)`, and AIC scan
  :math:`\\mathrm{AIC}(w_\\mathrm{base})`.

Both functions accept an optional ``source`` (an
:class:`~piblin.data.data_collections.experiment_set.ExperimentSet` /
:class:`~piblin.data.data_collections.measurement.Measurement`)
so axis-unit strings track the actual data — the default
:math:`s \\in [s]` and :math:`h \\in \\mathrm{Pa}` are used when no
source is provided.

Examples
--------
>>> from hermes_rheo.visualization import plot_relaxation_spectrum
>>> from hermes_rheo.transforms.relaxation_spectrum import RelaxationSpectrum
>>> out = RelaxationSpectrum(domain="frequency")(measurement)  # doctest: +SKIP
>>> fig, ax = plot_relaxation_spectrum(
...     out.details["relaxation_spectrum_result"]
... )  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization.units import get_unit, unit_suffix


__all__ = [
    "plot_relaxation_spectrum",
    "plot_spectrum_diagnostics",
]


def _resolve_units(source: Optional[Any]) -> Tuple[str, str]:
    """Return ``(time_unit, modulus_unit)`` for the axis labels."""
    if source is None:
        return "s", "Pa"
    time_unit = (
        get_unit(source, "relaxation time")
        or get_unit(source, "time")
        or get_unit(source, "step time")
        or "s"
    )
    modulus_unit = (
        get_unit(source, "h")
        or get_unit(source, "storage modulus")
        or get_unit(source, "relaxation modulus")
        or get_unit(source, "modulus")
        or "Pa"
    )
    return time_unit, modulus_unit


def plot_relaxation_spectrum(
    result: Any,
    ax: Optional[matplotlib.axes.Axes] = None,
    *,
    show_discrete: bool = True,
    show_continuous: bool = True,
    show_uncertainty: bool = True,
    source: Optional[Any] = None,
    label_continuous: Optional[str] = None,
    label_discrete: Optional[str] = None,
    color_continuous: str = "tab:blue",
    color_discrete: str = "tab:red",
    unit_style: str = "bracket",
    legend_loc: str = "best",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot continuous and/or discrete relaxation spectra on log-log axes.

    Parameters
    ----------
    result : RelaxationSpectrumResult
        Result produced by
        :class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`.
    ax : matplotlib.axes.Axes or None, optional
        Target axes.  A new figure and axes pair is created when ``None``.
    show_discrete : bool, default True
        Overlay the discrete Prony spectrum :math:`(g_i, \\tau_i)` as
        stem markers.
    show_continuous : bool, default True
        Draw the continuous CRS :math:`h(s)` curve.
    show_uncertainty : bool, default True
        Shade the :math:`\\pm dH` uncertainty band (frequency branch only;
        silently skipped when ``result.dH`` is ``None``).
    source : piblin container or None, optional
        Object used to resolve axis-label units via
        :func:`~hermes_rheo.visualization.get_unit`.  Falls back to
        ``s`` for time and ``Pa`` for modulus when ``None``.
    label_continuous, label_discrete : str or None
        Legend labels.
    color_continuous, color_discrete : str
        Matplotlib colours.
    unit_style : {"paren", "bracket"}, default "bracket"
        Format for the unit suffix in axis labels.  Same convention as
        :func:`hermes_rheo.visualization.unit_suffix`.
    legend_loc : str, default "best"
        Legend location, passed to matplotlib's ``ax.legend(loc=...)``.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes

    Notes
    -----
    Both the CRS grid and the DRS stems are plotted on log-log axes; the
    CRS convention is that :math:`h(s) = \\exp H(s)` is in Pa.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    time_unit, modulus_unit = _resolve_units(source)
    time_label = f"$s${unit_suffix(time_unit, style=unit_style)}"
    modulus_label = f"$h(s)$, $g_i${unit_suffix(modulus_unit, style=unit_style)}"

    plotted_any = False

    if show_continuous and result.s is not None and result.h is not None:
        s = np.asarray(result.s, dtype=float)
        h = np.asarray(result.h, dtype=float)
        (line,) = ax.plot(
            s, h,
            "-", color=color_continuous, linewidth=2,
            label=label_continuous or f"continuous ({result.domain})",
        )
        if show_uncertainty and getattr(result, "dH", None) is not None:
            dH = np.asarray(result.dH, dtype=float)
            if dH.shape == s.shape:
                # dH is the log-space band on H = ln h.  Convert to h space.
                h_low = np.exp(np.log(h) - dH)
                h_high = np.exp(np.log(h) + dH)
                ax.fill_between(
                    s, h_low, h_high,
                    color=color_continuous, alpha=0.2, linewidth=0,
                    label=r"$\pm dH$",
                )
        plotted_any = True

    if show_discrete and result.tau is not None and result.g is not None and result.tau.size > 0:
        tau = np.asarray(result.tau, dtype=float)
        g = np.asarray(result.g, dtype=float)
        markerline, stemlines, baseline = ax.stem(
            tau, g,
            linefmt="-",
            markerfmt="o",
            basefmt=" ",
            label=label_discrete or f"discrete N={tau.size}",
        )
        plt.setp(markerline, markerfacecolor=color_discrete,
                 markeredgecolor=color_discrete, markersize=7)
        plt.setp(stemlines, color=color_discrete, linewidth=1.5)
        plotted_any = True

    if plotted_any:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(time_label)
        ax.set_ylabel(modulus_label)
        ax.grid(False, which="both", alpha=0.3)
        ax.legend(loc=legend_loc, frameon=False)

    return fig, ax


def plot_spectrum_diagnostics(
    result: Any,
    axes: Optional[np.ndarray] = None,
    *,
    unit_style: str = "bracket",
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """Three-panel diagnostics for a pyReSpect fit.

    Panels (left to right):

    1. **L-curve** :math:`\\rho(\\lambda)` vs :math:`\\eta(\\lambda)` with
       the selected :math:`\\lambda_C` marked.
    2. **Log evidence** :math:`\\log p(\\lambda)` as a function of
       :math:`\\lambda`, with :math:`\\lambda_C` marked.
    3. **AIC scan** across the discrete-mode base-weight sweep, with the
       optimum marked.

    Parameters
    ----------
    result : RelaxationSpectrumResult
    axes : ndarray of matplotlib.axes.Axes or None, optional
        Three pre-allocated axes; a new 1x3 figure is created when
        ``None``.
    unit_style : {"paren", "bracket"}, default "bracket"

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (3,)

    Notes
    -----
    Panels are silently skipped when the corresponding diagnostic array
    is missing (``None``).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    else:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure

    # --- L-curve
    ax_l = axes[0]
    if (
        getattr(result, "rho", None) is not None
        and getattr(result, "eta", None) is not None
    ):
        rho = np.asarray(result.rho, dtype=float)
        eta = np.asarray(result.eta, dtype=float)
        ax_l.plot(rho, eta, "-o", color="tab:blue", markersize=4)
        lam_C = float(result.lam_C)
        # Mark the point closest to lam_C on the L-curve.
        if getattr(result, "lam", None) is not None:
            lam = np.asarray(result.lam, dtype=float)
            idx = int(np.argmin(np.abs(np.log(lam) - np.log(lam_C))))
            ax_l.plot([rho[idx]], [eta[idx]], "o", color="tab:red", markersize=9,
                      label=fr"$\lambda_C = {lam_C:.2e}$")
            ax_l.legend(loc="best", frameon=False)
        ax_l.set_xscale("log")
        ax_l.set_yscale("log")
        ax_l.set_xlabel(r"$\rho(\lambda)$")
        ax_l.set_ylabel(r"$\eta(\lambda)$")
        ax_l.set_title("L-curve")
        ax_l.grid(False, which="both", alpha=0.3)
    else:
        ax_l.set_title("L-curve (unavailable)")
        ax_l.axis("off")

    # --- Log evidence
    ax_p = axes[1]
    if (
        getattr(result, "lam", None) is not None
        and getattr(result, "log_P", None) is not None
    ):
        lam = np.asarray(result.lam, dtype=float)
        log_P = np.asarray(result.log_P, dtype=float)
        ax_p.semilogx(lam, log_P, "-o", color="tab:green", markersize=4)
        ax_p.axvline(float(result.lam_C), color="tab:red", linestyle="--",
                     label=fr"$\lambda_C = {result.lam_C:.2e}$")
        ax_p.set_xlabel(r"$\lambda$")
        ax_p.set_ylabel(r"$\log p(\lambda)$")
        ax_p.set_title("Bayesian evidence")
        ax_p.grid(False, which="both", alpha=0.3)
        ax_p.legend(loc="best", frameon=False)
    else:
        ax_p.set_title(r"$\log p(\lambda)$ (unavailable)")
        ax_p.axis("off")

    # --- AIC scan
    ax_a = axes[2]
    if (
        getattr(result, "wt_base", None) is not None
        and getattr(result, "AIC_bst", None) is not None
    ):
        wt = np.asarray(result.wt_base, dtype=float)
        aic = np.asarray(result.AIC_bst, dtype=float)
        ax_a.plot(wt, aic, "-o", color="tab:purple", markersize=4)
        idx = int(np.argmin(aic))
        ax_a.plot([wt[idx]], [aic[idx]], "o", color="tab:red", markersize=9,
                  label=fr"$N_\mathrm{{opt}} = {int(result.N_opt)}$")
        ax_a.set_xlabel(r"base-weight distance")
        ax_a.set_ylabel("AIC")
        ax_a.set_title(f"AIC scan (domain={result.domain})")
        ax_a.grid(False, which="both", alpha=0.3)
        ax_a.legend(loc="best", frameon=False)
    else:
        ax_a.set_title("AIC scan (unavailable)")
        ax_a.axis("off")

    return fig, axes
