"""Matplotlib style helpers built on top of
`SciencePlots <https://github.com/garrettj403/SciencePlots>`_.

``SciencePlots`` (John Garrett *et al.*) installs a family of Matplotlib
style-sheets that render publication-quality figures out of the box.
This module wraps the two style-combinations most useful for
`hermes-rheo` users:

* :func:`use_science_style` — the default ``science`` style with LaTeX
  disabled so it works on any machine (including corporate laptops
  without a TeX distribution installed).
* :func:`reset_style` — restore Matplotlib's default rcParams.

If ``scienceplots`` is not installed, the helper logs a clear warning
and leaves the current style untouched — no crash — so notebooks
importing ``hermes_rheo.visualization`` on a minimal environment still
run.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from hermes_rheo.visualization import use_science_style
>>>
>>> use_science_style()             # science + no-latex
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [1, 4, 9])

Add the ``grid`` and ``ieee`` variants::

>>> use_science_style(extras=['grid', 'ieee'])
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)

#: Default composite style used by :func:`use_science_style` — the
#: ``science`` base plus the ``notebook`` variant (readable ticks and
#: fonts on notebook-sized figures), with LaTeX rendering disabled so
#: the style works on any machine.
DEFAULT_STYLES: tuple[str, ...] = ("science", "notebook", "no-latex")


def use_science_style(
    latex: bool = False,
    extras: Optional[Iterable[str]] = None,
) -> list[str]:
    """Apply the SciencePlots ``science + notebook`` style-sheet globally.

    Parameters
    ----------
    latex : bool, optional
        If *True*, enable LaTeX rendering (requires a working TeX
        distribution — ``latex`` and ``dvipng`` on the PATH).  Defaults
        to *False* so figures render everywhere.
    extras : iterable of str, optional
        Additional SciencePlots sub-styles to layer on top (e.g.
        ``['grid']``, ``['ieee']``, ``['bright']``, ``['high-contrast']``).
        See the `SciencePlots README
        <https://github.com/garrettj403/SciencePlots>`_ for the full
        list.

    Returns
    -------
    list of str
        The list of style-sheet names actually applied.  Empty if
        ``scienceplots`` is not installed.

    Notes
    -----
    Import side-effect: ``import scienceplots`` registers the style
    sheets with Matplotlib.  We import it lazily inside this function so
    that ``hermes_rheo.visualization`` remains importable on
    installations without SciencePlots.
    """
    try:
        import scienceplots  # noqa: F401 (side-effect: registers styles)
    except ImportError:
        _logger.warning(
            "scienceplots is not installed; use_science_style() is a no-op. "
            "Install it with:  pip install scienceplots",
        )
        return []

    styles: list[str] = ["science", "notebook"]
    if not latex:
        styles.append("no-latex")
    if extras:
        styles.extend(extras)

    plt.style.use(styles)

    # Guarantee a full frame with ticks on all four sides (major *and*
    # minor), pointing inward.  SciencePlots enables top/right ticks,
    # but we set these explicitly so the four-sided box is consistent
    # regardless of the active sub-styles or any downstream overrides.
    plt.rcParams.update(
        {
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.top": True,
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        }
    )
    return styles


def reset_style() -> None:
    """Reset Matplotlib rcParams to the built-in default."""
    plt.style.use("default")
