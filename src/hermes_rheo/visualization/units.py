"""Read units from piblin datasets and build axis labels.

Examples
--------
>>> from hermes_rheo.visualization import get_unit, unit_suffix
>>> u = get_unit(experiment_set, "storage modulus")   # doctest: +SKIP
>>> ax.set_ylabel("storage modulus" + unit_suffix(u))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

__all__ = ["axis_label", "get_unit", "unit_suffix"]


def _first_dataset(source: Any) -> Any:
    """Peel a set, measurement or container down to its first dataset."""
    ds = source
    if getattr(ds, "measurements", None):
        ds = ds.measurements[0]
    if getattr(ds, "datasets", None):
        ds = ds.datasets[0]
    return ds


def get_unit(source: Any, name: str) -> str:
    """Return the unit of a named data array.

    Parameters
    ----------
    source : ExperimentSet, MeasurementSet, Measurement or dataset
        Containers are reduced to their first measurement and its first
        dataset.
    name : str
        Name of the data array, matched exactly against
        ``data_array_names``.

    Returns
    -------
    str
        The stored unit, or ``""`` if *name* is absent or has no unit.

    Examples
    --------
    >>> get_unit(experiment, "storage modulus")  # doctest: +SKIP
    'MPa'
    """
    ds = _first_dataset(source)

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


def axis_label(
    source: Any,
    name: str,
    *,
    text: str | None = None,
    unit: str | None = None,
    style: str = "paren",
) -> str:
    """Build a ``"name (unit)"`` axis label from a dataset.

    Parameters
    ----------
    source : ExperimentSet, MeasurementSet, Measurement or dataset
        Where to look up the unit; see :func:`get_unit`.
    name : str
        Name of the data array.
    text : str, optional
        Label text to use instead of *name*.
    unit : str, optional
        Unit to use instead of the one stored in *source*.
    style : {"paren", "bracket"}, default "paren"
        Bracket style passed to :func:`unit_suffix`.

    Returns
    -------
    str
        For example ``"storage modulus (kPa)"``; the unit is omitted if
        it is empty.

    Examples
    --------
    >>> axis_label(dataset, "storage modulus")  # doctest: +SKIP
    'storage modulus (Pa)'
    """
    if unit is None:
        unit = get_unit(source, name)
    return (text if text is not None else name) + unit_suffix(unit, style=style)
