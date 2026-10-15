"""
Rheology-specific visualisation functions for ExperimentSet.

Two plot functions are provided:

* :func:`plot_vgp`       – van Gurp-Palmen plot:  tan(δ) vs ``|G*|``
* :func:`plot_cole_cole` – Cole-Cole plot:          G'' vs G'

Both functions iterate over every measurement in an
:py:class:`~piblin.data.ExperimentSet`, colour-coding each trace
by temperature using a ``coolwarm`` colormap.

Usage::

    from hermes_rheo.visualization import plot_vgp, plot_cole_cole

    fig, ax = plot_vgp(experiment)
    fig, ax = plot_cole_cole(experiment)
"""

from __future__ import annotations

import copy

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_temperature(measurement) -> float | None:
    """Extract a scalar temperature label from a *Measurement*.

    Looks for temperature information in the following order:

    1. ``measurement.conditions`` – any key whose name contains
       ``'temperature'`` (case-insensitive).
    2. ``measurement.details`` – same rule.
    3. The first dataset's data arrays – looks for an array named
       ``'temperature'`` and returns its mean value.

    Parameters
    ----------
    measurement : piblin.data.Measurement

    Returns
    -------
    float or None
        Scalar temperature in whatever unit the data were stored, or
        *None* when no temperature information can be found.
    """
    # 1. conditions
    for key, val in measurement.conditions.items():
        if "temperature" in key.lower():
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    # 2. details
    for key, val in measurement.details.items():
        if "temperature" in key.lower():
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    # 3. dataset variable
    for ds in measurement.datasets:
        names = getattr(ds, "_data_array_names", [])
        arrays = getattr(ds, "_data_arrays", [])
        for arr_name, arr in zip(names, arrays):
            if arr_name.lower() == "temperature":
                try:
                    return float(np.nanmean(arr))
                except Exception:
                    pass

    return None


def _extract_xy(
    dataset,
    x_name: str,
    y_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_values, y_values)`` from a dataset without mutating it.

    A deep-copy of the dataset is used so that the caller's dataset is
    not affected by the coordinate switch.

    Parameters
    ----------
    dataset : OneDimensionalCompositeDataset
    x_name : str
        Name of the data array to use as the independent variable.
    y_name : str
        Name of the data array to use as the dependent variable.

    Returns
    -------
    x_values : np.ndarray
    y_values : np.ndarray

    Raises
    ------
    ValueError
        If *x_name* or *y_name* are not present in the dataset's
        ``data_array_names``.
    """
    ds = copy.deepcopy(dataset)
    ds.switch_coordinates(independent_name=x_name, dependent_name=y_name)
    return ds.x_values, ds.y_values


def _build_color_getter(temperatures, colormap):
    """Return a callable ``get_color(temp)`` normalised to the temperature range.

    Parameters
    ----------
    temperatures : list
        Scalar temperature values (may contain *None*).
    colormap : matplotlib colormap

    Returns
    -------
    get_color : callable
        Accepts a temperature value and returns an RGBA tuple.
    """
    valid = [t for t in temperatures if t is not None]
    if len(valid) >= 2:
        norm = mcolors.Normalize(vmin=min(valid), vmax=max(valid))
        return lambda t: colormap(norm(t)) if t is not None else colormap(0.5)

    # Fallback: evenly-spaced colours across the full colormap range.
    n = max(len(temperatures), 1)
    mapping = {t: colormap(i / max(n - 1, 1)) for i, t in enumerate(temperatures)}
    return lambda t: mapping.get(t, colormap(0.5))


def _get_array_unit(dataset, array_name: str) -> str:
    """Return the unit string for a named data array in a dataset.

    Thin wrapper around :func:`hermes_rheo.visualization.units.get_unit`
    kept for backward-compatibility with internal call sites.

    Parameters
    ----------
    dataset : OneDimensionalCompositeDataset
    array_name : str
        The name of the data array whose unit is requested.

    Returns
    -------
    str
        The unit string, or an empty string when not found.
    """
    from hermes_rheo.visualization.units import get_unit

    return get_unit(dataset, array_name)


def _first_valid_dataset(experiment_set, dataset_index: int = 0):
    """Return the first dataset found in *experiment_set* at *dataset_index*.

    Iterates over measurements and returns the first dataset at *dataset_index*
    that exists.  Used to sample unit strings before the main plotting loop
    without mutating any measurement.

    Parameters
    ----------
    experiment_set : piblin.data.ExperimentSet
    dataset_index : int, optional
        Which dataset slot to look in.  Defaults to ``0``.

    Returns
    -------
    OneDimensionalCompositeDataset or None
        The first matching dataset, or *None* if the experiment set is empty
        or no measurement has a dataset at *dataset_index*.
    """
    for m in experiment_set.measurements:
        if len(m.datasets) > dataset_index:
            return m.datasets[dataset_index]
    return None


def _add_temperature_colorbar(fig, ax, temperatures, colormap, label="Temperature (°C)") -> bool:
    """Attach a temperature colorbar to *ax*; return *True* on success.

    Falls back (returns *False*) when fewer than two numeric temperatures
    are available, in which case a colorbar would be meaningless.
    """
    valid = [t for t in temperatures if isinstance(t, (int, float))]
    if len(valid) < 2:
        return False
    norm = mcolors.Normalize(vmin=min(valid), vmax=max(valid))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=11)
    return True


def _apply_temperature_key(
    fig, ax, temperatures, colormap, legend, legend_max, legend_title=None
) -> None:
    """Add either a discrete legend or a temperature colorbar to *ax*.

    Parameters
    ----------
    legend : {'auto', 'colorbar', True, False}
        ``'auto'`` (default) draws a discrete legend when the number of
        distinct temperatures is ``<= legend_max`` and a colorbar
        otherwise.  ``True`` forces the legend; ``False`` or
        ``'colorbar'`` forces the colorbar.
    legend_max : int
        Threshold (number of distinct temperatures) above which the
        ``'auto'`` mode switches to a colorbar.
    legend_title : str, optional
        Title for the discrete legend.
    """
    valid = [t for t in temperatures if isinstance(t, (int, float))]
    n_distinct = len(set(valid)) if valid else len(temperatures)

    if legend is True:
        mode = "legend"
    elif legend is False or legend == "colorbar":
        mode = "colorbar"
    else:  # 'auto'
        mode = "colorbar" if n_distinct > legend_max else "legend"

    if mode == "colorbar" and _add_temperature_colorbar(fig, ax, temperatures, colormap):
        return

    legend_kwargs = dict(
        fontsize=10,
        title_fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
    )
    if legend_title:
        legend_kwargs["title"] = legend_title
    ax.legend(**legend_kwargs)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_vgp(
    experiment_set,
    ax: matplotlib.axes.Axes | None = None,
    colormap=None,
    x_name: str = "complex modulus",
    y_name: str = "tan(delta)",
    dataset_index: int = 0,
    legend: str | bool = "auto",
    legend_max: int = 12,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Van Gurp-Palmen (vGP) plot: tan(δ) vs complex modulus ``|G*|``.

    Each measurement within the ExperimentSet (typically recorded at a
    distinct temperature) is drawn as an individual trace.  Traces are
    colour-coded by temperature using the ``coolwarm`` colormap (cold
    colours → low temperature; warm colours → high temperature).

    The complex modulus axis is rendered on a logarithmic scale.
    Tick marks are drawn on all four sides pointing inward.
    The legend is positioned outside the plot area to the upper-right.

    Parameters
    ----------
    experiment_set : piblin.data.ExperimentSet
        The experiment set to visualise.  Each measurement must contain a
        :class:`~piblin.data.OneDimensionalCompositeDataset` with
        (at minimum) the ``'complex modulus'`` and ``'tan(delta)'``
        data arrays.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw into.  A new figure and axes are
        created when *None* (default).
    colormap : matplotlib colormap, optional
        Colormap used to colour traces by temperature.
        Defaults to :data:`matplotlib.cm.coolwarm`.
    x_name : str, optional
        Name of the data array to use as the x-axis.
        Defaults to ``'complex modulus'``.
    y_name : str, optional
        Name of the data array to use as the y-axis.
        Defaults to ``'tan(delta)'``.
    dataset_index : int, optional
        Index of the dataset within each measurement to use.
        Defaults to ``0``.
    legend : {'auto', 'colorbar', True, False}, optional
        Temperature key to draw.  ``'auto'`` (default) uses a discrete
        legend when there are ``<= legend_max`` distinct temperatures and
        a colorbar otherwise (ideal for high-resolution temperature
        ramps).  ``True`` forces the legend; ``False`` or ``'colorbar'``
        forces the colorbar.
    legend_max : int, optional
        Distinct-temperature threshold above which ``'auto'`` switches to
        a colorbar.  Defaults to ``12``.
    **kwargs
        Additional keyword arguments forwarded verbatim to
        :py:meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If any measurement's dataset does not contain the required
        ``x_name`` or ``y_name`` arrays (these measurements are silently
        skipped rather than raising, but a ``ValueError`` is raised if
        *no* measurements could be plotted).

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.visualization import plot_vgp
    >>> from piblin.data import ExperimentSet
    >>> ms = TriosRheoReader().data_from_filepath('trios_tts.txt', create_composite_datasets=True)
    >>> experiment = ExperimentSet(ms.measurements)
    >>> fig, ax = plot_vgp(experiment)

    Notes
    -----
    The van Gurp-Palmen representation [1]_ is useful for probing
    time-temperature superposition (TTS) compliance: traces at different
    temperatures should collapse onto a single master curve for a
    thermorheologically simple material.

    References
    ----------
    .. [1] van Gurp, M. & Palmen, J. (1998).  *Time-temperature
       superposition for polymeric blends*.  Rheology Bulletin, 67(1), 5–8.
    """
    if colormap is None:
        colormap = plt.cm.coolwarm

    measurements = experiment_set.measurements
    temperatures = [_get_temperature(m) for m in measurements]
    get_color = _build_color_getter(temperatures, colormap)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Resolve axis units from the first available dataset.
    _ref_ds = _first_valid_dataset(experiment_set, dataset_index)
    x_unit = _get_array_unit(_ref_ds, x_name) if _ref_ds is not None else ""
    y_unit = _get_array_unit(_ref_ds, y_name) if _ref_ds is not None else ""

    x_label = f"|G*| ({x_unit})" if x_unit else "|G*|"
    y_label = f"tan(δ) ({y_unit})" if y_unit else "tan(δ)"

    # Default style: markers only, no connecting lines.
    plot_style = {"linestyle": "none", "marker": "o", "markersize": 5}
    plot_style.update(kwargs)

    plotted = 0
    for measurement, temp in zip(measurements, temperatures):
        dataset = measurement.datasets[dataset_index]
        try:
            x_vals, y_vals = _extract_xy(dataset, x_name, y_name)
        except (ValueError, AttributeError):
            continue  # dataset does not have the required variables

        label = f"T = {temp:.3g} °C" if isinstance(temp, float) else f"T = {temp}"
        ax.plot(x_vals, y_vals, label=label, color=get_color(temp), **plot_style)
        plotted += 1

    if plotted == 0:
        raise ValueError(
            f"No measurements could be plotted. "
            f"Ensure each dataset contains '{x_name}' and '{y_name}' arrays."
        )

    # --- axes formatting ---------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title("Van Gurp-Palmen plot", fontsize=14)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    _apply_temperature_key(
        fig,
        ax,
        temperatures,
        colormap,
        legend,
        legend_max,
        legend_title="Temperature",
    )
    fig.tight_layout()
    return fig, ax


def plot_cole_cole(
    experiment_set,
    ax: matplotlib.axes.Axes | None = None,
    colormap=None,
    x_name: str = "storage modulus",
    y_name: str = "loss modulus",
    dataset_index: int = 0,
    legend: str | bool = "auto",
    legend_max: int = 12,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Cole-Cole plot: loss modulus G'' vs storage modulus G'.

    Each measurement within the ExperimentSet (typically recorded at a
    distinct temperature) is drawn as an individual trace.  Traces are
    colour-coded by temperature using the ``coolwarm`` colormap.

    Both axes are rendered on a logarithmic scale.
    Tick marks are drawn on all four sides pointing inward.
    The legend is positioned outside the plot area to the upper-right.

    Parameters
    ----------
    experiment_set : piblin.data.ExperimentSet
        The experiment set to visualise.  Each measurement must contain a
        :class:`~piblin.data.OneDimensionalCompositeDataset` with
        (at minimum) the ``'storage modulus'`` and ``'loss modulus'``
        data arrays.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw into.  A new figure and axes are
        created when *None* (default).
    colormap : matplotlib colormap, optional
        Colormap used to colour traces by temperature.
        Defaults to :data:`matplotlib.cm.coolwarm`.
    x_name : str, optional
        Name of the data array to use as the x-axis.
        Defaults to ``'storage modulus'``.
    y_name : str, optional
        Name of the data array to use as the y-axis.
        Defaults to ``'loss modulus'``.
    dataset_index : int, optional
        Index of the dataset within each measurement to use.
        Defaults to ``0``.
    legend : {'auto', 'colorbar', True, False}, optional
        Temperature key to draw.  ``'auto'`` (default) uses a discrete
        legend when there are ``<= legend_max`` distinct temperatures and
        a colorbar otherwise (ideal for high-resolution temperature
        ramps).  ``True`` forces the legend; ``False`` or ``'colorbar'``
        forces the colorbar.
    legend_max : int, optional
        Distinct-temperature threshold above which ``'auto'`` switches to
        a colorbar.  Defaults to ``12``.
    **kwargs
        Additional keyword arguments forwarded verbatim to
        :py:meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If no measurements could be plotted because the required data
        arrays are absent.

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.visualization import plot_cole_cole
    >>> from piblin.data import ExperimentSet
    >>> ms = TriosRheoReader().data_from_filepath('trios_tts.txt', create_composite_datasets=True)
    >>> experiment = ExperimentSet(ms.measurements)
    >>> fig, ax = plot_cole_cole(experiment)
    """
    if colormap is None:
        colormap = plt.cm.coolwarm

    measurements = experiment_set.measurements
    temperatures = [_get_temperature(m) for m in measurements]
    get_color = _build_color_getter(temperatures, colormap)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Resolve axis units from the first available dataset.
    _ref_ds = _first_valid_dataset(experiment_set, dataset_index)
    x_unit = _get_array_unit(_ref_ds, x_name) if _ref_ds is not None else ""
    y_unit = _get_array_unit(_ref_ds, y_name) if _ref_ds is not None else ""

    x_label = f"G' ({x_unit})" if x_unit else "G'"
    y_label = f"G'' ({y_unit})" if y_unit else "G''"

    # Default style: markers only, no connecting lines.
    plot_style = {"linestyle": "none", "marker": "o", "markersize": 5}
    plot_style.update(kwargs)

    plotted = 0
    for measurement, temp in zip(measurements, temperatures):
        dataset = measurement.datasets[dataset_index]
        try:
            x_vals, y_vals = _extract_xy(dataset, x_name, y_name)
        except (ValueError, AttributeError):
            continue

        label = f"T = {temp:.3g} °C" if isinstance(temp, float) else f"T = {temp}"
        ax.plot(x_vals, y_vals, label=label, color=get_color(temp), **plot_style)
        plotted += 1

    if plotted == 0:
        raise ValueError(
            f"No measurements could be plotted. "
            f"Ensure each dataset contains '{x_name}' and '{y_name}' arrays."
        )

    # --- axes formatting ---------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title("Cole-Cole plot", fontsize=14)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    _apply_temperature_key(
        fig,
        ax,
        temperatures,
        colormap,
        legend,
        legend_max,
        legend_title="Temperature",
    )
    fig.tight_layout()
    return fig, ax
