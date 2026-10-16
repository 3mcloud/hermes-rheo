"""Matplotlib style and shared layout constants for hermes-rheo plots.

The layout constants (figure sizes, font sizes, line widths, markers and
colours) are used by every ``plot_*`` function so that figures look the same
regardless of which one produced them. :func:`use_science_style` additionally
applies the `SciencePlots <https://github.com/garrettj403/SciencePlots>`_
style sheets; it is optional and a no-op when ``scienceplots`` is missing.

Examples
--------
>>> from hermes_rheo.visualization import use_science_style
>>> use_science_style()  # doctest: +SKIP
['science', 'notebook', 'no-latex']
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
from piblin.data.datasets.abc.dataset import Dataset

_logger = logging.getLogger(__name__)

# piblin's own axes styling, which hides the top and right spines and ticks.
_PIBLIN_STYLE_AXES = Dataset.__dict__["_style_axes"]

#: Style sheets applied by :func:`use_science_style` (LaTeX disabled).
DEFAULT_STYLES: tuple[str, ...] = ("science", "notebook", "no-latex")

#: Figure sizes in inches: one panel, a wide row of panels, a 2x2 grid.
FIGSIZE = (6.5, 4.8)
FIGSIZE_WIDE = (12.0, 4.8)
FIGSIZE_GRID = (10.0, 7.0)
FIGSIZE_RESIDUALS = (6.5, 6.0)

#: Font sizes in points.
LABEL_SIZE = 12
TITLE_SIZE = 13
LEGEND_SIZE = 10

#: Line widths in points and marker sizes.
LINEWIDTH = 2.0
LINEWIDTH_THIN = 1.0
MARKERSIZE = 5.0

#: Marker symbols for storage-like (filled) and loss-like (open) data.
MARKER_STORAGE = "o"
MARKER_LOSS = "s"

#: Colours for the recurring quantities.
COLOR_STORAGE = "tab:red"
COLOR_LOSS = "tab:blue"
COLOR_MODEL = "k"
COLOR_SHIFT = "tab:green"
COLOR_STRAIN = "seagreen"
COLOR_STRESS = "darkorange"
COLOR_MARK = "tab:red"
COLOR_GUIDE = "0.6"

#: Default colormap for traces ordered by temperature.
COLORMAP = "coolwarm"


def use_science_style(
    latex: bool = False,
    extras: Iterable[str] | None = None,
) -> list[str]:
    """Apply the SciencePlots ``science`` and ``notebook`` styles globally.

    Axes are boxed, with ticks pointing inward on all four sides and minor
    ticks shown. piblin's ``Dataset.visualize`` normally hides the top and
    right spines; while this style is active it follows these settings
    instead, so piblin and hermes-rheo plots look the same.

    Parameters
    ----------
    latex : bool, default False
        Render text with LaTeX. Requires ``latex`` and ``dvipng`` on the
        PATH.
    extras : iterable of str, optional
        Further SciencePlots style names, for example ``["grid"]`` or
        ``["ieee"]``.

    Returns
    -------
    list of str
        The style names applied; empty if ``scienceplots`` is not installed.
    """
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        _logger.warning(
            "scienceplots is not installed; use_science_style() does nothing. "
            "Install it with: pip install scienceplots"
        )
        return []

    styles: list[str] = ["science", "notebook"]
    if not latex:
        styles.append("no-latex")
    if extras:
        styles.extend(extras)

    plt.style.use(styles)
    Dataset._style_axes = staticmethod(_style_axes_from_rcparams)
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
    """Restore the Matplotlib default style and piblin's axes styling."""
    plt.style.use("default")
    Dataset._style_axes = _PIBLIN_STYLE_AXES


def _style_axes_from_rcparams(axes) -> None:
    """Show spines and ticks on the sides enabled in ``plt.rcParams``."""
    rc = plt.rcParams
    for side in ("top", "right", "bottom", "left"):
        axes.spines[side].set_visible(rc[f"axes.spines.{side}"])
    axes.tick_params(
        top=rc["xtick.top"],
        bottom=rc["xtick.bottom"],
        left=rc["ytick.left"],
        right=rc["ytick.right"],
        which="both",
    )
