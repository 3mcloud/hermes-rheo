"""Plots of linear viscoelastic data: van Gurp-Palmen and Cole-Cole.

Both functions draw one trace per measurement, coloured by temperature, and
read axis names and units from the datasets. They also work on a single
measurement without temperature, such as simulated data from
:mod:`hermes_rheo.lve_models`.
"""

from __future__ import annotations

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors as mcolors
import matplotlib.figure
import numpy as np

from hermes_rheo.visualization import _common as c
from hermes_rheo.visualization import style
from hermes_rheo.visualization.units import axis_label, get_unit

__all__ = ["plot_cole_cole", "plot_vgp"]


def _measurements(source) -> list:
    """Return the measurements held by *source* (a set or one measurement)."""
    measurements = getattr(source, "measurements", None)
    if measurements is not None:
        return list(measurements)
    return [source]


def _temperature(measurement) -> float | None:
    """Return a scalar temperature for *measurement*, or ``None``.

    Conditions are checked first, then details, then the mean of a
    ``temperature`` data array.
    """
    for mapping in (measurement.conditions, measurement.details):
        for key, value in mapping.items():
            if "temperature" in key.lower():
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    for dataset in measurement.datasets:
        for name, array in zip(dataset.data_array_names, dataset.data_arrays):
            if name.lower() == c.TEMPERATURE:
                try:
                    return float(np.nanmean(array))
                except (TypeError, ValueError):
                    continue
    return None


def _trace_colors(temperatures: list[float | None], cmap) -> list:
    """Return one colour per trace, scaled to the temperature range."""
    valid = [t for t in temperatures if t is not None]
    if len(temperatures) == 1:
        return [None]
    if len(valid) >= 2 and max(valid) > min(valid):
        norm = mcolors.Normalize(vmin=min(valid), vmax=max(valid))
        return [cmap(norm(t)) if t is not None else cmap(0.5) for t in temperatures]
    n = len(temperatures)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _add_temperature_key(
    fig, ax, temperatures, cmap, label, legend, legend_max, has_labels
) -> None:
    """Draw a legend or a temperature colorbar on *ax*."""
    valid = [t for t in temperatures if t is not None]
    n_distinct = len(set(valid))
    if legend is True or legend == "legend":
        use_colorbar = False
    elif legend is False or legend == "colorbar":
        use_colorbar = True
    else:
        use_colorbar = n_distinct > legend_max
    if use_colorbar and n_distinct >= 2:
        norm = mcolors.Normalize(vmin=min(valid), vmax=max(valid))
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, pad=0.02).set_label(label, fontsize=style.LABEL_SIZE)
    elif has_labels:
        c.add_legend(ax, outside=len(temperatures) > 1, title=label.split(" (")[0])


def _plot_pair(
    experiment_set,
    *,
    x_name: str,
    y_name: str,
    logx: bool,
    logy: bool,
    default_title: str,
    ax,
    colormap,
    dataset_index: int,
    legend,
    legend_max: int,
    title,
    xlabel,
    ylabel,
    save,
    plot_kwargs: dict,
):
    """Shared implementation of the ``y`` versus ``x`` trace plots."""
    cmap = matplotlib.colormaps[style.COLORMAP] if colormap is None else colormap
    measurements = _measurements(experiment_set)
    temperatures = [_temperature(m) for m in measurements]
    colors = _trace_colors(temperatures, cmap)
    fig, ax = c.get_axes(ax, style.FIGSIZE)

    options = {
        "linestyle": "none",
        "marker": style.MARKER_STORAGE,
        "markersize": style.MARKERSIZE,
    }
    options.update(plot_kwargs)

    reference = None
    t_unit = ""
    has_labels = False
    for measurement, temp, color in zip(measurements, temperatures, colors):
        if len(measurement.datasets) <= dataset_index:
            continue
        dataset = measurement.datasets[dataset_index]
        x = c.array_by_name(dataset, x_name)
        y = c.array_by_name(dataset, y_name)
        if x is None or y is None:
            continue
        if reference is None:
            reference = dataset
        t_unit = t_unit or get_unit(dataset, c.TEMPERATURE) or get_unit(dataset, "Temperature")
        if temp is not None:
            label = f"{temp:.3g}" + (f" {t_unit}" if t_unit else "")
        else:
            label = measurement.conditions.get("method")
        has_labels = has_labels or bool(label)
        line_options = dict(options)
        if color is not None and "color" not in line_options:
            line_options["color"] = color
        ax.plot(x, y, label=label, **line_options)

    if reference is None:
        raise ValueError(
            f"No measurement could be plotted: no dataset at index {dataset_index} "
            f"contains both '{x_name}' and '{y_name}'."
        )

    c.decorate_axes(
        ax,
        xlabel=xlabel if xlabel is not None else axis_label(reference, x_name),
        ylabel=ylabel if ylabel is not None else axis_label(reference, y_name),
        title=default_title if title is None else title,
        logx=logx,
        logy=logy,
    )
    _add_temperature_key(
        fig,
        ax,
        temperatures,
        cmap,
        "temperature" + (f" ({t_unit})" if t_unit else ""),
        legend,
        legend_max,
        has_labels,
    )
    c.finish_figure(fig, save=save)
    return fig, ax


def plot_vgp(
    experiment_set,
    ax: matplotlib.axes.Axes | None = None,
    colormap=None,
    x_name: str = c.COMPLEX_MODULUS,
    y_name: str = c.TAN_DELTA,
    dataset_index: int = 0,
    legend: str | bool = "auto",
    legend_max: int = 12,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save: str | None = None,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the loss tangent against the complex modulus (van Gurp-Palmen).

    The van Gurp-Palmen plot shows the phase angle, here as
    ``tan(delta)``, against ``|G*|`` with frequency as the implicit
    parameter. Because the plot does not contain frequency or temperature
    explicitly, curves measured at different temperatures overlap on a
    single curve when time-temperature superposition holds; vertical or
    horizontal gaps between them indicate that it fails.

    Parameters
    ----------
    experiment_set : ExperimentSet, MeasurementSet or Measurement
        One trace is drawn per measurement.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    colormap : matplotlib colormap, optional
        Colormap for the temperature colours; ``coolwarm`` by default.
    x_name, y_name : str
        Names of the data arrays on the x and y axes.
    dataset_index : int, default 0
        Dataset to use within each measurement.
    legend : {"auto", "legend", "colorbar", True, False}, default "auto"
        Temperature key. ``"auto"`` uses a legend for up to *legend_max*
        distinct temperatures and a colorbar for more. ``True`` forces the
        legend; ``False`` forces the colorbar.
    legend_max : int, default 12
        Largest number of distinct temperatures shown in a legend.
    title, xlabel, ylabel : str, optional
        Override the default title and the labels read from the dataset.
    save : str, optional
        Path to write the figure to.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If no measurement contains both *x_name* and *y_name*.

    Examples
    --------
    >>> fig, ax = plot_vgp(experiment_set)  # doctest: +SKIP
    """
    return _plot_pair(
        experiment_set,
        x_name=x_name,
        y_name=y_name,
        logx=True,
        logy=False,
        default_title="Van Gurp-Palmen plot",
        ax=ax,
        colormap=colormap,
        dataset_index=dataset_index,
        legend=legend,
        legend_max=legend_max,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save=save,
        plot_kwargs=kwargs,
    )


def plot_cole_cole(
    experiment_set,
    ax: matplotlib.axes.Axes | None = None,
    colormap=None,
    x_name: str = c.STORAGE_MODULUS,
    y_name: str = c.LOSS_MODULUS,
    dataset_index: int = 0,
    legend: str | bool = "auto",
    legend_max: int = 12,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save: str | None = None,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the loss modulus against the storage modulus (Cole-Cole).

    In this representation frequency is the implicit parameter. A single
    relaxation process (Maxwell model) gives a semicircle in linear
    axes, and curves from different temperatures superpose when
    time-temperature superposition holds. Both axes are logarithmic here.

    Parameters
    ----------
    experiment_set : ExperimentSet, MeasurementSet or Measurement
        One trace is drawn per measurement.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    colormap : matplotlib colormap, optional
        Colormap for the temperature colours; ``coolwarm`` by default.
    x_name, y_name : str
        Names of the data arrays on the x and y axes.
    dataset_index : int, default 0
        Dataset to use within each measurement.
    legend : {"auto", "legend", "colorbar", True, False}, default "auto"
        Temperature key; see :func:`plot_vgp`.
    legend_max : int, default 12
        Largest number of distinct temperatures shown in a legend.
    title, xlabel, ylabel : str, optional
        Override the default title and the labels read from the dataset.
    save : str, optional
        Path to write the figure to.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If no measurement contains both *x_name* and *y_name*.

    Examples
    --------
    >>> fig, ax = plot_cole_cole(experiment_set)  # doctest: +SKIP
    """
    return _plot_pair(
        experiment_set,
        x_name=x_name,
        y_name=y_name,
        logx=True,
        logy=True,
        default_title="Cole-Cole plot",
        ax=ax,
        colormap=colormap,
        dataset_index=dataset_index,
        legend=legend,
        legend_max=legend_max,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save=save,
        plot_kwargs=kwargs,
    )
