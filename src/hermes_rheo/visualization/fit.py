"""Visualisation helpers for rheological-model and shift-factor fits.

This module renders the output of
:class:`~hermes_rheo.transforms.fit_model.FitModel` (storage/loss-modulus
fits) and
:class:`~hermes_rheo.transforms.shift_factor_models.ShiftFactorModel`
(WLF / Arrhenius fits of the TTS shift factors).

Both helpers follow the ``hermes_rheo`` plotting conventions:

* they create a Matplotlib ``Figure``/``Axes`` only when one is not
  supplied, so they compose cleanly into multi-panel figures;
* they respect whatever global style is active (e.g. after
  :func:`hermes_rheo.visualization.use_science_style`) instead of
  hard-coding fonts and tick styles;
* they *return* the ``(figure, axes)`` they draw on and never call
  ``plt.show`` implicitly (pass ``show=True`` to display).
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def plot_model_fit(
    fit_model,
    ax: Optional[matplotlib.axes.Axes] = None,
    show_residuals: bool = False,
    data_label: Optional[str] = None,
    fit_label: Optional[str] = None,
    show: bool = False,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""Plot a fitted :class:`FitModel` (moduli data + model prediction).

    Parameters
    ----------
    fit_model : hermes_rheo.transforms.fit_model.FitModel
        A ``FitModel`` instance that has already been applied to data
        (so ``omega_common``, ``G_prime``, ``G_double_prime`` and
        ``fitted_model`` are populated).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.  When ``None`` (default) a new figure and axis
        are created.  Ignored when ``show_residuals`` is ``True``.
    show_residuals : bool, optional
        When ``True`` add a lower panel with the relative residuals
        ``(data - fit) / data`` for both moduli.  Defaults to ``False``.
    data_label, fit_label : str, optional
        Legend labels for the data markers and the fit line.  Sensible
        defaults derived from the model name are used when omitted.
    show : bool, optional
        Call ``plt.show`` before returning.  Defaults to ``False``.
    **kwargs
        Optional ``figsize``, ``xlabel``, ``ylabel``, ``title``,
        ``xlim``, ``ylim`` overrides.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and the moduli axis.
    """
    if fit_model.fitted_model is None or fit_model.omega_common is None:
        raise AttributeError("FitModel has no fitted model or data to plot.")

    omega = np.asarray(fit_model.omega_common, dtype=float)
    gp = np.asarray(fit_model.G_prime, dtype=float)
    gpp = np.asarray(fit_model.G_double_prime, dtype=float)
    gp_pred, gpp_pred = fit_model.fitted_model.predict(omega)

    model_name = getattr(fit_model.fitted_model, "model", "model")
    fit_label = fit_label or f"{model_name} fit"

    figsize = kwargs.pop("figsize", (7, 5))
    if show_residuals:
        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax_res = None

    ax.plot(omega, gp, "o", ms=6, color="tab:red",
            label=data_label or r"$G'$ data")
    ax.plot(omega, gpp, "s", ms=6, mfc="none", color="tab:blue",
            label=data_label if data_label else r"$G''$ data")
    ax.plot(omega, gp_pred, "-", lw=2, color="k", label=fit_label)
    ax.plot(omega, gpp_pred, "--", lw=2, color="k")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(kwargs.pop("ylabel", r"$G',\ G''$ (Pa)"))
    ax.legend(loc="best", frameon=False)
    if "title" in kwargs:
        ax.set_title(kwargs.pop("title"))
    if "xlim" in kwargs:
        ax.set_xlim(kwargs.pop("xlim"))
    if "ylim" in kwargs:
        ax.set_ylim(kwargs.pop("ylim"))

    xlabel = kwargs.pop("xlabel", r"$\omega$ (rad s$^{-1}$)")
    if show_residuals:
        with np.errstate(divide="ignore", invalid="ignore"):
            res_gp = (gp - gp_pred) / gp
            res_gpp = (gpp - gpp_pred) / gpp
        ax_res.axhline(0.0, color="0.6", lw=1)
        ax_res.plot(omega, res_gp, "o", ms=4, color="tab:red")
        ax_res.plot(omega, res_gpp, "s", ms=4, mfc="none", color="tab:blue")
        ax_res.set_xscale("log")
        ax_res.set_ylabel("rel. resid.")
        ax_res.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_shift_factors(
    shift_model,
    ax: Optional[matplotlib.axes.Axes] = None,
    n_curve: int = 200,
    data_label: str = "shift factors",
    fit_label: Optional[str] = None,
    color: str = "k",
    linestyle: str = "-",
    show_data: bool = True,
    show: bool = False,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""Plot a fitted :class:`ShiftFactorModel` (``a_T`` on a log axis vs temperature).

    Parameters
    ----------
    shift_model : hermes_rheo.transforms.shift_factor_models.ShiftFactorModel
        A fitted ``ShiftFactorModel`` (``temperatures``,
        ``log_shift_factors`` and ``fitted_model`` populated).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.  A new figure/axis is created when ``None``.
    n_curve : int, optional
        Number of points used to render the smooth fit curve.  Defaults
        to ``200``.
    data_label : str, optional
        Legend label for the data markers.
    fit_label : str, optional
        Legend label for the fit line.  Defaults to a compact description
        of the fitted model (WLF constants or activation energy).
    color : str, optional
        Colour of the fit line.  Defaults to ``"k"``.
    linestyle : str, optional
        Line style of the fit line (e.g. ``"-"``, ``"--"``).  Defaults to
        ``"-"``.
    show_data : bool, optional
        Whether to draw the data markers.  Set to ``False`` when overlaying
        a second fit on the same axis to avoid duplicate markers.  Defaults
        to ``True``.
    show : bool, optional
        Call ``plt.show`` before returning.  Defaults to ``False``.
    **kwargs
        Optional ``figsize``, ``xlabel``, ``ylabel``, ``title`` overrides.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn on.
    """
    if shift_model.fitted_model is None:
        raise AttributeError("ShiftFactorModel has not been fitted yet.")

    temperatures = np.asarray(shift_model.temperatures, dtype=float)
    log_aT = np.asarray(shift_model.log_shift_factors, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 5)))
    else:
        fig = ax.figure

    t_dense = np.linspace(temperatures.min(), temperatures.max(), n_curve)
    log_aT_fit = shift_model.predict(t_dense)

    # Plot the shift factor a_T itself on a logarithmic y-axis (rather than
    # log10(a_T) on a linear axis).
    aT = 10.0 ** log_aT
    aT_fit = 10.0 ** log_aT_fit

    fit_label = fit_label or _shift_fit_label(shift_model)
    if show_data:
        ax.plot(temperatures, aT, "o", ms=6, color="tab:green", label=data_label)
    ax.plot(t_dense, aT_fit, linestyle, lw=2, color=color, label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel(kwargs.pop("xlabel", r"Temperature ($^\circ$C)"))
    ax.set_ylabel(kwargs.pop("ylabel", r"$a_T$"))
    if "title" in kwargs:
        ax.set_title(kwargs.pop("title"))
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _shift_fit_label(shift_model) -> str:
    """Build a compact legend label describing the fitted shift-factor model."""
    model = shift_model.fitted_model
    if getattr(model, "model", None) == "WLF":
        return f"WLF ($C_1$={model.C1:.3g}, $C_2$={model.C2:.3g})"
    if getattr(model, "model", None) == "Arrhenius":
        return f"Arrhenius ($E_a$={model.activation_energy / 1000.0:.3g} kJ/mol)"
    return "fit"
