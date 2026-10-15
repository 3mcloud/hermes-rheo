"""Unit-metadata helpers for building axis labels from piblin datasets.

The two public helpers make it easy for callers (and tutorials) to keep
plot labels in sync with whatever unit the file-reader actually stored,
rather than hard-coding strings that can drift from the underlying data:

* :func:`get_unit` — resolve the unit of a named data array from a
  :class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
  or from any container that exposes ``measurements`` / ``datasets``.
* :func:`unit_suffix` — format a unit as an axis-label suffix.

Examples
--------
>>> from hermes_rheo.visualization import get_unit, unit_suffix
>>> u = get_unit(experiment_set, "storage modulus")   # doctest: +SKIP
>>> ax.set_ylabel(f"$G'${unit_suffix(u)}")             # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_unit", "unit_suffix"]


def get_unit(source: Any, name: str) -> str:
    """Return the unit of a named data array in a piblin composite dataset.

    The helper walks one level down through common piblin containers:

    * :class:`~piblin.data.ExperimentSet` /
      :class:`~piblin.data.MeasurementSet` /
      :class:`~piblin.data.Measurement`
      (peeled to their first measurement's first dataset), or
    * any object exposing a ``datasets`` attribute (peeled to
      ``datasets[0]``), or
    * a dataset directly — anything exposing ``data_array_names`` and
      ``data_array_units``.

    Parameters
    ----------
    source : object
        An ``ExperimentSet``, ``MeasurementSet``, ``Measurement``, or a
        composite dataset.
    name : str
        The name of the data array whose unit is requested.  Matched
        exactly against ``data_array_names``.

    Returns
    -------
    str
        The stored unit string, or ``""`` when *name* is not registered
        or the resolved dataset has no unit metadata.

    Examples
    --------
    >>> from hermes_rheo import read_file  # doctest: +SKIP
    >>> exp = read_file("data.txt", ...)          # doctest: +SKIP
    >>> get_unit(exp, "storage modulus")            # doctest: +SKIP
    'MPa'
    """
    ds = source
    if hasattr(ds, "measurements"):
        measurements = getattr(ds, "measurements", None)
        if measurements:
            ds = measurements[0]
    if hasattr(ds, "datasets"):
        datasets = getattr(ds, "datasets", None)
        if datasets:
            ds = datasets[0]

    names = list(getattr(ds, "data_array_names", []) or [])
    units = list(getattr(ds, "data_array_units", []) or [])
    if name in names:
        idx = names.index(name)
        if idx < len(units) and units[idx]:
            return str(units[idx])
    return ""


def unit_suffix(unit: str, style: str = "paren", sep: str = " ") -> str:
    """Format *unit* as a plot-label suffix, or ``""`` if the unit is empty.

    Parameters
    ----------
    unit : str
        The unit string to wrap.  An empty (or falsy) value returns
        ``""`` so labels stay clean for dimensionless quantities.
    style : {"paren", "bracket"}, optional
        Wrap the unit in ``"( )"`` (default) or ``"[ ]"``.
    sep : str, optional
        Separator inserted between the label and the suffix.  Defaults
        to a single space.

    Returns
    -------
    str
        The formatted suffix, ready to concatenate onto an axis label.

    Examples
    --------
    >>> unit_suffix("MPa")
    ' (MPa)'
    >>> unit_suffix("MPa", style="bracket")
    ' [MPa]'
    >>> unit_suffix("")
    ''
    """
    if not unit:
        return ""
    if style == "paren":
        return f"{sep}({unit})"
    if style == "bracket":
        return f"{sep}[{unit}]"
    raise ValueError(f"style must be 'paren' or 'bracket', got {style!r}")
