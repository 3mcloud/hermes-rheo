"""Plots of rheological model fits and time-temperature shift-factor fits.

The functions render the output of
:class:`~hermes_rheo.transforms.fit_model.FitModel` and
:class:`~hermes_rheo.transforms.shift_factor_models.ShiftFactorModel`.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.units import axis_label

__all__ = ["plot_model_fit", "plot_shift_factors"]

_DATA_MARKERS = (style.MARKER_STORAGE, style.MARKER_LOSS)
_DATA_COLORS = (style.COLOR_STORAGE, style.COLOR_LOSS)


def _fit_arrays(fit_measurement):
    """Split a fit measurement into the dataset, x name and (data, fitted) names.

    The first array is the independent variable. Arrays named
    ``"fitted <name>"`` are the model predictions of the array ``<name>``.
    """
    dataset = fit_measurement.datasets[0]
    names = list(dataset.data_array_names)
    pairs = [
        (name, c.FITTED_PREFIX + name)
        for name in names[1:]
        if not name.startswith(c.FITTED_PREFIX) and c.FITTED_PREFIX + name in names
    ]
    return dataset, names[0], pairs


def plot_model_fit(
    fit_model,
    ax: matplotlib.axes.Axes | None = None,
    show_residuals: bool = False,
    data_label: str | None = None,
    fit_label: str | None = None,
    show: bool = False,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot measured data and the fitted model from a fitted ``FitModel``.

    Names and units of the axes are read from ``fit_model.fit_measurement``.
    Oscillation fits show the storage and loss moduli against angular
    frequency; relaxation fits show the relaxation modulus against time.

    Parameters
    ----------
    fit_model : FitModel
        A ``FitModel`` that has been applied to data.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Ignored if *show_residuals* is true.
    show_residuals : bool, default False
        Add a lower panel with the relative residuals ``(data - fit) / data``.
    data_label, fit_label : str, optional
        Legend labels. The data label defaults to the array name and the
        fit label to ``"<model> fit"``.
    show : bool, default False
        Call :func:`matplotlib.pyplot.show` before returning.
    title, xlabel, ylabel : str, optional
        Override the default title and the labels read from the data.
    xlim, ylim : tuple of float, optional
        Axis limits.
    figsize : tuple of float, optional
        Size of a newly created figure.
    save : str, optional
        Path to write the figure to.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The main axes (not the residual panel).

    Raises
    ------
    AttributeError
        If *fit_model* has not been applied to data.

    Examples
    --------
    >>> fit = FitModel(model="custom", model_name="MultiModeMaxwell")
    >>> _ = fit(experiment_set)  # doctest: +SKIP
    >>> fig, ax = plot_model_fit(fit, show_residuals=True)  # doctest: +SKIP
    """
    if getattr(fit_model, "fit_measurement", None) is None:
        raise AttributeError("FitModel has no fitted model or data to plot.")

    dataset, x_name, pairs = _fit_arrays(fit_model.fit_measurement)
    x = c.array_by_name(dataset, x_name)
    model_name = getattr(fit_model.fitted_model, "model", "model")
    fit_label = fit_label or f"{model_name} fit"

    if show_residuals:
        fig, (ax, ax_res) = plt.subplots(
            2,
            1,
            figsize=figsize or style.FIGSIZE_RESIDUALS,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = c.get_axes(ax, figsize or style.FIGSIZE)
        ax_res = None

    for i, (name, fitted_name) in enumerate(pairs):
        y = c.array_by_name(dataset, name)
        y_fit = c.array_by_name(dataset, fitted_name)
        marker = _DATA_MARKERS[i % 2]
        color = _DATA_COLORS[i % 2]
        face = "none" if i else color
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=marker,
            markersize=style.MARKERSIZE,
            markerfacecolor=face,
            color=color,
            label=data_label or name,
        )
        ax.plot(
            x,
            y_fit,
            linestyle="-" if i == 0 else "--",
            linewidth=style.LINEWIDTH,
            color=style.COLOR_MODEL,
            label=fit_label if i == 0 else None,
        )
        if ax_res is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                residual = (y - y_fit) / y
            ax_res.plot(
                x,
                residual,
                linestyle="none",
                marker=marker,
                markersize=style.MARKERSIZE,
                markerfacecolor=face,
                color=color,
            )

    if len(pairs) == 1:
        y_default = axis_label(dataset, pairs[0][0])
    else:
        y_default = axis_label(dataset, pairs[0][0], text="modulus")
    x_label = xlabel if xlabel is not None else axis_label(dataset, x_name)
    c.decorate_axes(
        ax,
        xlabel=None if ax_res is not None else x_label,
        ylabel=ylabel if ylabel is not None else y_default,
        title=title,
        logx=True,
        logy=True,
        xlim=xlim,
        ylim=ylim,
    )
    c.add_legend(ax)

    if ax_res is not None:
        ax_res.axhline(0.0, color=style.COLOR_GUIDE, linewidth=style.LINEWIDTH_THIN)
        c.decorate_axes(ax_res, xlabel=x_label, ylabel="relative residual", logx=True)

    c.finish_figure(fig, save=save, show=show)
    return fig, ax


def plot_shift_factors(
    shift_model,
    ax: matplotlib.axes.Axes | None = None,
    n_curve: int = 200,
    data_label: str = "shift factors",
    fit_label: str | None = None,
    color: str = style.COLOR_MODEL,
    linestyle: str = "-",
    show_data: bool = True,
    data_color: str = style.COLOR_SHIFT,
    show: bool = False,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""Plot shift factors :math:`a_T` and the fitted WLF or Arrhenius curve.

    The shift factor is drawn on a logarithmic axis against temperature.
    The WLF equation (Williams, Landel and Ferry, J. Am. Chem. Soc. 77,
    3701 (1955)) describes :math:`\log_{10} a_T` above the glass
    transition; the Arrhenius form describes activated processes.

    Parameters
    ----------
    shift_model : ShiftFactorModel
        A fitted model with ``temperatures``, ``log_shift_factors`` and
        ``fitted_model``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    n_curve : int, default 200
        Number of points of the fitted curve.
    data_label : str, default "shift factors"
        Legend label of the data.
    fit_label : str, optional
        Legend label of the fit. Defaults to the fitted constants.
    color : str, default "k"
        Colour of the fit line.
    linestyle : str, default "-"
        Line style of the fit line.
    show_data : bool, default True
        Draw the data markers. Use ``False`` when overlaying a second fit.
    data_color : str, default "tab:green"
        Colour of the data markers.
    show : bool, default False
        Call :func:`matplotlib.pyplot.show` before returning.
    title, xlabel, ylabel : str, optional
        Override the default title and labels.
    figsize : tuple of float, optional
        Size of a newly created figure.
    save : str, optional
        Path to write the figure to.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    AttributeError
        If *shift_model* has not been fitted.

    Examples
    --------
    >>> fig, ax = plot_shift_factors(shift_model)  # doctest: +SKIP
    """
    if shift_model.fitted_model is None:
        raise AttributeError("ShiftFactorModel has not been fitted yet.")

    temperatures = np.asarray(shift_model.temperatures, dtype=float)
    shift = 10.0 ** np.asarray(shift_model.log_shift_factors, dtype=float)
    t_dense = np.linspace(temperatures.min(), temperatures.max(), n_curve)
    shift_fit = 10.0 ** np.asarray(shift_model.predict(t_dense), dtype=float)

    fig, ax = c.get_axes(ax, figsize or style.FIGSIZE)
    if show_data:
        ax.plot(
            temperatures,
            shift,
            linestyle="none",
            marker=style.MARKER_STORAGE,
            markersize=style.MARKERSIZE,
            color=data_color,
            label=data_label,
        )
    ax.plot(
        t_dense,
        shift_fit,
        linestyle=linestyle,
        linewidth=style.LINEWIDTH,
        color=color,
        label=fit_label or _shift_fit_label(shift_model),
    )

    unit = c.temperature_unit_symbol(getattr(shift_model, "temperature_unit", ""))
    default_xlabel = f"temperature ({unit})" if unit else "temperature"
    c.decorate_axes(
        ax,
        xlabel=xlabel if xlabel is not None else default_xlabel,
        ylabel=ylabel if ylabel is not None else r"$a_T$",
        title=title,
        logy=True,
    )
    c.add_legend(ax)
    c.finish_figure(fig, save=save, show=show)
    return fig, ax


def _shift_fit_label(shift_model) -> str:
    """Return a legend label with the fitted constants."""
    model = shift_model.fitted_model
    kind = getattr(model, "model", None)
    if kind == "WLF":
        return f"WLF ($C_1$={model.C1:.3g}, $C_2$={model.C2:.3g})"
    if kind == "Arrhenius":
        return f"Arrhenius ($E_a$={model.activation_energy / 1000.0:.3g} kJ/mol)"
    return "fit"
