"""Kymograph transform: temperature x frequency maps of a viscoelastic quantity.

Builds a 2-D map from a :class:`~piblin.data.data_collections.measurement_set.MeasurementSet`,
e.g. one read by :class:`hermes_rheo.file_readers.trios_rheo_txt.TriosRheoReader`
and processed by ``OWChirpAnalysis``. It is intended for TORCh (Thermally
Optimised Ramped Chirp) data [Vadillo2026]_ but is not limited to it: each
axis is taken from a dataset column (e.g. ``temperature``) or, failing that,
from ``measurement.conditions`` (e.g. ``Angular Frequency`` on a
:class:`~hermes_rheo.transforms.mutation_number.MutationNumberMeasurementSet`).

Examples
--------
>>> from hermes_rheo import read_file
>>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
>>> from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
>>> from hermes_rheo.transforms.kymograph import Kymograph
>>> experiment = read_file(filepath, file_reader=TriosRheoReader,
...                        create_composite_datasets=True)  # doctest: +SKIP
>>> pipeline = OWChirpAnalysis() + Kymograph(quantity="tan(delta)")
>>> result = pipeline.apply_to(experiment)  # doctest: +SKIP
>>> result.plot()  # doctest: +SKIP

Frequency on the x-axis, temperature on the y-axis::

    Kymograph("tan(delta)", x="angular frequency", y="temperature",
              log_x=True, log_y=False)

Mutation number map of a TORCh ramp::

    from hermes_rheo.transforms.mutation_number import MutationNumber
    result = (
        MutationNumber(absolute=True, owch_length=10)
        + Kymograph(quantity="mutation number")
    ).apply_to(experiment)
    result.plot(colormap="magma", vmax=0.15)

References
----------
.. [Vadillo2026] Vadillo, D. C., Perego, A. *et al.*, "TORCh: Thermally
   Optimised Ramped Chirp for time-temperature spectroscopy of
   polymers", Macromolecules (2026).
   https://doi.org/10.1021/acs.macromol.6c00679
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import piblin.data.data_collections.measurement_set as measurement_set_mod
import piblin.data.datasets.abc.split_datasets.two_dimensional_dataset as two_dimensional_dataset_mod
from piblin.data import Measurement
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
from scipy.interpolate import griddata

from hermes_rheo import __version__
from hermes_rheo.visualization import style

#: Accepted column-name aliases for each canonical quantity.
QUANTITY_ALIASES: dict[str, list[str]] = {
    "complex modulus": ["|complex modulus|", "complex modulus", "g*", "|g*|"],
    "storage modulus": ["storage modulus", "g'", "g′"],
    "loss modulus": ["loss modulus", "g''", "g″", 'g"'],
    "tan(delta)": ["tan(delta)", "tan δ", "tan(d)", "loss factor"],
    "complex viscosity": ["|complex viscosity|", "complex viscosity", "|η*|", "eta*"],
    "phase angle": ["phase angle"],
    "dynamic viscosity": ["dynamic viscosity"],
    "mutation number": ["mutation number", "mu", "m_u"],
}

#: Default units for each canonical quantity key.
_DEFAULT_UNITS: dict[str, str] = {
    "complex modulus": "Pa",
    "storage modulus": "Pa",
    "loss modulus": "Pa",
    "tan(delta)": "",
    "complex viscosity": "Pa·s",
    "phase angle": "°",
    "dynamic viscosity": "Pa·s",
    "mutation number": "",
}


def _resolve_column(column_names: list[str], quantity: str) -> str | None:
    """Return the first column name that matches *quantity* (case-insensitive).

    Searches the :data:`QUANTITY_ALIASES` look-up first, then falls back to a
    direct substring match against *column_names*.

    Parameters
    ----------
    column_names : list of str
        Names available in the dataset.
    quantity : str
        The canonical quantity name (e.g. ``"tan(delta)"``).

    Returns
    -------
    str or None
        The matching column name, or *None* if not found.
    """
    q_lower = quantity.lower().strip()
    candidates = QUANTITY_ALIASES.get(q_lower, [q_lower])
    name_map = {n.lower(): n for n in column_names}
    for cand in candidates:
        if cand.lower() in name_map:
            return name_map[cand.lower()]
    # substring fallback
    for orig in column_names:
        if q_lower in orig.lower():
            return orig
    return None


def _shared_items(dicts: list[dict]) -> dict:
    """Return the key/value pairs identical across all *dicts*."""
    if not dicts:
        return {}
    shared = dict(dicts[0])
    for other in dicts[1:]:
        shared = {k: v for k, v in shared.items() if k in other and np.array_equal(other[k], v)}
    return shared


class KymographResult(two_dimensional_dataset_mod.TwoDimensionalDataset):
    """Gridded 2-D piblin dataset produced by :class:`Kymograph`.

    A ``TwoDimensionalDataset`` (``x_values``, ``y_values``, ``z_values``)
    that also keeps the raw scatter points and provides slicing, plotting and
    Excel export. ``conditions`` and ``details`` carry the metadata shared by
    the source measurements plus the gridding parameters; :meth:`to_measurement`
    wraps the dataset and this metadata in a piblin ``Measurement``.

    Parameters
    ----------
    raw_x : np.ndarray
        Scattered x-values (e.g. temperature in °C).
    raw_y : np.ndarray
        Scattered y-values (e.g. angular frequency in rad/s).
    raw_z : np.ndarray
        Scattered z-values (the viscoelastic quantity).
    x_grid : np.ndarray, shape (n_x,)
        Regular x-axis grid.
    y_grid : np.ndarray, shape (n_y,)
        Regular y-axis grid.
    z_grid : np.ndarray, shape (n_x, n_y)
        Interpolated grid values.
    quantity : str
        Canonical quantity name (stored as ``z_name``).
    x_name : str
        Name of the x-axis variable.
    y_name : str
        Name of the y-axis variable.
    x_unit : str
        Physical unit of x.
    y_unit : str
        Physical unit of y.
    z_unit : str
        Physical unit of z.
    log_y : bool
        Whether the y-axis uses a logarithmic scale.
    log_x : bool, optional
        Whether the x-axis uses a logarithmic scale.  Defaults to ``False``.
    conditions : dict, optional
        Conditions shared by the source measurements.
    details : dict, optional
        Details shared by the source measurements and gridding parameters.
    """

    def __init__(
        self,
        *,
        raw_x: np.ndarray,
        raw_y: np.ndarray,
        raw_z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        quantity: str,
        x_name: str,
        y_name: str,
        x_unit: str,
        y_unit: str,
        z_unit: str,
        log_y: bool,
        log_x: bool = False,
        conditions: dict | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            dependent_variable_data=z_grid,
            dependent_variable_names=[quantity],
            dependent_variable_units=[z_unit] if z_unit else None,
            independent_variable_data=[x_grid, y_grid],
            independent_variable_names=[x_name, y_name],
            independent_variable_units=[x_unit, y_unit],
        )
        self.raw_x = np.asarray(raw_x, float)
        self.raw_y = np.asarray(raw_y, float)
        self.raw_z = np.asarray(raw_z, float)
        self.quantity = quantity
        self.log_x = log_x
        self.log_y = log_y
        self.conditions = dict(conditions) if conditions else {}
        self.details = dict(details) if details else {}

    def to_measurement(self) -> Measurement:
        """Return a piblin ``Measurement`` with this dataset, conditions and details."""
        details = {k: v for k, v in self.details.items() if k not in self.conditions}
        return Measurement(datasets=[self], conditions=self.conditions, details=details)

    @property
    def grid(self) -> KymographResult:
        """The gridded dataset (this object)."""
        return self

    def slice_at_x(
        self,
        x_values: float | list[float],
        *,
        use_raw: bool = False,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Return z vs y at one or more fixed x (e.g. temperature) values.

        Parameters
        ----------
        x_values : float or list of float
            The x-axis value(s) at which to slice.
        use_raw : bool, optional
            If *True*, search the raw scatter data (nearest-x match).
            If *False* (default), interpolate from the regular grid.

        Returns
        -------
        dict
            ``{x_val: (y_array, z_array)}`` for each requested x.

        Examples
        --------
        >>> slices = result.slice_at_x([25, 100])
        >>> for T, (freq, modulus) in slices.items():
        ...     plt.loglog(freq, modulus, label=f"T={T} °C")
        """
        if np.ndim(x_values) == 0:
            x_values = [float(x_values)]

        out: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        x_grid = self.x_values
        y_grid = self.y_values
        z_grid = self.z_values  # shape (n_x, n_y)

        for xv in x_values:
            if use_raw:
                mask = np.isclose(self.raw_x, xv, atol=0.5)
                idx = np.argsort(self.raw_y[mask])
                out[xv] = (self.raw_y[mask][idx], self.raw_z[mask][idx])
            else:
                xi = int(np.argmin(np.abs(x_grid - xv)))
                out[float(x_grid[xi])] = (y_grid, z_grid[xi, :])

        return out

    def slice_at_y(
        self,
        y_values: float | list[float],
        *,
        use_raw: bool = False,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Return z vs x at one or more fixed y (e.g. frequency) values.

        Parameters
        ----------
        y_values : float or list of float
            The y-axis value(s) at which to slice.
        use_raw : bool, optional
            If *True*, search the raw scatter data (nearest-y match).
            If *False* (default), interpolate from the regular grid.

        Returns
        -------
        dict
            ``{y_val: (x_array, z_array)}`` for each requested y.

        Examples
        --------
        >>> slices = result.slice_at_y([1.0, 6.28])   # rad/s
        >>> for omega, (temp, modulus) in slices.items():
        ...     plt.semilogy(temp, modulus, label=f"ω={omega:.2f} rad/s")
        """
        if np.ndim(y_values) == 0:
            y_values = [float(y_values)]

        out: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        x_grid = self.x_values
        y_grid = self.y_values
        z_grid = self.z_values  # shape (n_x, n_y)

        for yv in y_values:
            if use_raw:
                mask = np.isclose(self.raw_y, yv, rtol=0.05)
                idx = np.argsort(self.raw_x[mask])
                out[yv] = (self.raw_x[mask][idx], self.raw_z[mask][idx])
            else:
                yi = int(np.argmin(np.abs(y_grid - yv)))
                out[float(y_grid[yi])] = (x_grid, z_grid[:, yi])

        return out

    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        colormap: str = "RdYlBu_r",
        log_z: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        add_scatter: bool = False,
        add_contours: bool = False,
        contour_levels: int | list[float] | None = None,
        contour_colors: str = "k",
        figsize: tuple[float, float] = style.FIGSIZE,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the 2-D kymograph (heatmap).

        Which variable is on which axis is set when the map is built, e.g.
        ``Kymograph(x="angular frequency", y="temperature", log_x=True,
        log_y=False)`` puts frequency on the horizontal axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.  A new figure is created when *None*.
        colormap : str, optional
            A Matplotlib colormap name.  Defaults to ``"RdYlBu_r"``.
        log_z : bool, optional
            Use a logarithmic colour scale.
        vmin, vmax : float, optional
            Colour-scale limits.  Default: the range of the data (positive
            values only when *log_z*).
        add_scatter : bool, optional
            Overlay the raw data points as black dots.
        add_contours : bool, optional
            Overlay contour lines on the heatmap.  Defaults to ``False``.
        contour_levels : int or list of float, optional
            Number of contour levels (int) or explicit level values (list).
            When *None* (default), Matplotlib chooses automatically.
            Only used when *add_contours* is ``True``.
        contour_colors : str, optional
            Colour of the contour lines.  Defaults to ``"k"`` (black).
        figsize : tuple, optional
            Figure size in inches (width, height).
        title : str, optional
            Figure title override.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        x_grid = self.x_values  # (n_x,)
        y_grid = self.y_values  # (n_y,)
        z_grid = self.z_values.T  # (n_y, n_x) for pcolormesh

        if log_z:
            positive = z_grid[z_grid > 0]
            norm = mcolors.LogNorm(
                vmin=vmin if vmin is not None else positive.min(),
                vmax=vmax if vmax is not None else positive.max(),
            )
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        mesh = ax.pcolormesh(
            x_grid,
            y_grid,
            z_grid,
            cmap=colormap,
            norm=norm,
            shading="auto",
        )
        colorbar = fig.colorbar(mesh, ax=ax)
        colorbar.set_label(self._z_label(), fontsize=style.LABEL_SIZE)

        if add_contours:
            levels = contour_levels if contour_levels is not None else 8
            ax.contour(
                x_grid,
                y_grid,
                z_grid,
                levels=levels,
                colors=contour_colors,
                linewidths=0.8,
                linestyles="solid",
            )

        if add_scatter:
            ax.scatter(self.raw_x, self.raw_y, s=4, c="k", alpha=0.4, zorder=5, label="raw data")

        ax.set_xlabel(self._x_label(), fontsize=style.LABEL_SIZE)
        ax.set_ylabel(self._y_label(), fontsize=style.LABEL_SIZE)
        if self.log_x:
            ax.set_xscale("log")
        if self.log_y:
            ax.set_yscale("log")
        ax.set_title(title or self._default_title(), fontsize=style.TITLE_SIZE)

        fig.tight_layout()
        return fig, ax

    def plot_slice_at_x(
        self,
        x_values: float | list[float],
        *,
        ax: plt.Axes | None = None,
        log_x: bool = False,
        log_y_axis: bool = True,
        figsize: tuple[float, float] = (7, 4),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot z vs y (frequency) at fixed x (temperature) values.

        Parameters
        ----------
        x_values : float or list of float
            Temperature (or other x-axis) value(s) at which to slice.
        ax : matplotlib.axes.Axes, optional
        log_x, log_y_axis : bool, optional
            Axis scaling.  *log_y_axis* defaults to *True* for moduli.
        figsize : tuple, optional

        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        slices = self.slice_at_x(x_values)
        for xv, (yarr, zarr) in slices.items():
            ax.plot(
                yarr,
                zarr,
                marker="o",
                linestyle="none",
                markersize=4,
                label=f"{self.x_name}={xv:.1f} {self.x_unit or ''}".strip(),
            )

        if log_x or self.log_y:
            ax.set_xscale("log")
        if log_y_axis:
            ax.set_yscale("log")
        ax.set_xlabel(self._y_label())
        ax.set_ylabel(self._z_label())
        ax.set_title(f"{self.quantity} vs {self.y_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, ax

    def plot_slice_at_y(
        self,
        y_values: float | list[float],
        *,
        ax: plt.Axes | None = None,
        log_y_axis: bool = True,
        figsize: tuple[float, float] = (7, 4),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot z vs x (temperature) at fixed y (frequency) values.

        Parameters
        ----------
        y_values : float or list of float
            Angular frequency (or other y-axis) value(s) at which to slice.
        ax : matplotlib.axes.Axes, optional
        log_y_axis : bool, optional
            Log scale on the z-axis.  Defaults to *True*.
        figsize : tuple, optional

        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        slices = self.slice_at_y(y_values)
        if self.log_x:
            ax.set_xscale("log")
        for yv, (xarr, zarr) in slices.items():
            ax.plot(
                xarr,
                zarr,
                marker="o",
                linestyle="none",
                markersize=4,
                label=f"{self.y_name}={yv:.3g} {self.y_unit or ''}".strip(),
            )

        if log_y_axis:
            ax.set_yscale("log")
        ax.set_xlabel(self._x_label())
        ax.set_ylabel(self._z_label())
        ax.set_title(f"{self.quantity} vs {self.x_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, ax

    def to_excel(self, path: str) -> None:
        """Export raw and gridded data to an Excel workbook.

        Parameters
        ----------
        path : str
            File path (should end in ``.xlsx``).

        Notes
        -----
        The workbook contains three sheets:

        * **raw_data** — longform scatter (x, y, z).
        * **grid_data** — the interpolated 2-D grid.
        * **metadata** — quantity, axis names, units.
        """
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required for Excel export. Install it with: pip install openpyxl"
            ) from exc

        wb = Workbook()

        ws_raw = wb.active
        ws_raw.title = "raw_data"
        ws_raw.append([self._x_label(), self._y_label(), self._z_label()])
        for xv, yv, zv in zip(self.raw_x, self.raw_y, self.raw_z):
            ws_raw.append(
                [
                    None if not np.isfinite(xv) else float(xv),
                    None if not np.isfinite(yv) else float(yv),
                    None if not np.isfinite(zv) else float(zv),
                ]
            )

        ws_grid = wb.create_sheet("grid_data")
        x_grid = self.x_values
        y_grid = self.y_values
        z_grid = self.z_values  # (n_x, n_y)

        header = [f"{self._x_label()} \\ {self._y_label()}"] + [float(y) for y in y_grid]
        ws_grid.append(header)
        for xi, xv in enumerate(x_grid):
            row = [float(xv)] + [
                None if not np.isfinite(z_grid[xi, yi]) else float(z_grid[xi, yi])
                for yi in range(len(y_grid))
            ]
            ws_grid.append(row)

        ws_meta = wb.create_sheet("metadata")
        ws_meta.append(["Field", "Value"])
        ws_meta.append(["quantity", self.quantity])
        ws_meta.append(["x_name", self.x_name])
        ws_meta.append(["x_unit", self.x_unit])
        ws_meta.append(["y_name", self.y_name])
        ws_meta.append(["y_unit", self.y_unit])
        ws_meta.append(["z_name", self.quantity])
        ws_meta.append(["z_unit", self.z_unit])
        ws_meta.append(["log_x", str(self.log_x)])
        ws_meta.append(["log_y", str(self.log_y)])
        ws_meta.append(["n_raw_points", str(len(self.raw_x))])
        ws_meta.append(["grid_shape", f"{len(x_grid)} × {len(y_grid)}"])

        wb.save(path)

    def _x_label(self) -> str:
        return f"{self.x_name} ({self.x_unit})" if self.x_unit else self.x_name

    def _y_label(self) -> str:
        return f"{self.y_name} ({self.y_unit})" if self.y_unit else self.y_name

    def _z_label(self) -> str:
        return f"{self.quantity} ({self.z_unit})" if self.z_unit else self.quantity

    def _default_title(self) -> str:
        return f"{self.quantity} kymograph"

    def __repr__(self) -> str:
        return (
            f"KymographResult(TwoDimensionalDataset) "
            f"quantity='{self.quantity}', "
            f"n_raw={len(self.raw_x)}, "
            f"grid={len(self.x_values)}×{len(self.y_values)}, "
            f"x=[{self.x_values.min():.1f}, {self.x_values.max():.1f}] {self.x_unit or ''}, "
            f"y=[{self.y_values.min():.3g}, {self.y_values.max():.3g}] {self.y_unit or ''}"
        )


class Kymograph(MeasurementSetTransform):
    """Map a viscoelastic quantity onto a regular (x, y) grid.

    Collects ``(x, y, z)`` points from all datasets in a ``MeasurementSet``
    (x: slow variable, typically temperature; y: fast variable, typically
    angular frequency; z: the quantity), interpolates them onto an
    ``(n_x, n_y)`` mesh with :func:`scipy.interpolate.griddata` and returns a
    :class:`KymographResult`.

    Parameters
    ----------
    quantity : str, optional
        The viscoelastic quantity to map.  Must be one of the keys in
        :data:`QUANTITY_ALIASES` or a column name that exists in the data.
        Defaults to ``"tan(delta)"``.
    x : str, optional
        Name of the x-axis (slowly varying) variable.  Resolved first
        against dataset column names, then against
        ``measurement.conditions`` (a scalar condition is broadcast
        across the accompanying z array).  Defaults to ``"temperature"``.
    y : str, optional
        Name of the y-axis (fast varying) variable.  Same resolution
        rule as *x*.  Defaults to ``"angular frequency"``.
    n_x : int, optional
        Number of grid points along the x-axis.  Defaults to 200.
    n_y : int, optional
        Number of grid points along the y-axis.  Defaults to 80.
    interpolation : str, optional
        Interpolation method passed to :func:`scipy.interpolate.griddata`.
        One of ``"cubic"``, ``"linear"``, or ``"nearest"``.
        Defaults to ``"cubic"``.
    log_y : bool, optional
        Interpolate in log-space along y and plot y on a log scale
        (recommended for frequency axes).  Defaults to ``True``.
    log_x : bool, optional
        Same as *log_y* for the x-axis, e.g. when frequency is on x.
        Defaults to ``False``.
    x_min : float or None, optional
        Override the minimum x-value of the grid (e.g. to crop a cool-down
        ramp).  When *None* (default), the minimum observed x-value is used.
    fill_nearest : bool, optional
        After cubic/linear interpolation, fill remaining NaN pixels using
        nearest-neighbour interpolation.  Defaults to ``True``.

    Examples
    --------
    >>> from hermes_rheo import read_file
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
    >>> from hermes_rheo.transforms.kymograph import Kymograph
    >>>
    >>> experiment = read_file(
    ...     "path/to/torch_data.txt",
    ...     file_reader=TriosRheoReader,
    ...     create_composite_datasets=True,
    ... )
    >>> result = (OWChirpAnalysis() + Kymograph(quantity="tan(delta)")).apply_to(experiment)
    >>> result.plot()
    """

    def __init__(
        self,
        quantity: str = "tan(delta)",
        *,
        x: str = "temperature",
        y: str = "angular frequency",
        n_x: int = 200,
        n_y: int = 80,
        interpolation: str = "cubic",
        log_y: bool = True,
        log_x: bool = False,
        x_min: float | None = None,
        fill_nearest: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.quantity = quantity.lower().strip()
        self.x = x.lower().strip()
        self.y = y.lower().strip()
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.interpolation = interpolation
        self.log_y = log_y
        self.log_x = log_x
        self.x_min = x_min
        self.fill_nearest = fill_nearest

    def _apply(
        self,
        target: measurement_set_mod.MeasurementSet,
        **kwargs,
    ) -> KymographResult:
        """Extract scatter data from *target* and return a :class:`KymographResult`.

        Parameters
        ----------
        target : MeasurementSet
            The measurement set produced by reading a TORCh file.

        Returns
        -------
        KymographResult

        Raises
        ------
        ValueError
            If the requested quantity, x- or y-variable cannot be found in
            any dataset.
        """
        x_all: list[float] = []
        y_all: list[float] = []
        z_all: list[float] = []

        x_unit = ""
        y_unit = ""
        z_unit = _DEFAULT_UNITS.get(self.quantity, "")

        found_quantity = False
        found_x = False
        found_y = False

        measurements = list(getattr(target, "measurements", []))

        for m in measurements:
            datasets = getattr(m, "datasets", [])
            conditions = getattr(m, "conditions", {}) or {}
            # Case-insensitive alias table for scalar per-measurement
            # values (used when the requested axis is not a dataset
            # column — e.g. Mutation Number results carry
            # ``Angular Frequency`` in ``conditions`` rather than as a
            # column).
            cond_lookup = {str(k).lower(): (str(k), v) for k, v in conditions.items()}

            for ds in datasets:
                names = list(getattr(ds, "data_array_names", []) or [])
                arrays = list(getattr(ds, "data_arrays", []) or [])
                units_list = list(getattr(ds, "data_array_units", []) or [])

                # Compatibility with plain OneDimensionalDataset (the
                # shape produced by e.g. ``MutationNumber``), which
                # exposes independent / dependent arrays separately
                # rather than a single flat ``data_arrays`` list.
                if not names:
                    ind_names = list(getattr(ds, "independent_variable_names", []) or [])
                    ind_data = list(getattr(ds, "independent_variable_data", []) or [])
                    ind_units = list(getattr(ds, "independent_variable_units", []) or [])
                    dep_names = list(getattr(ds, "dependent_variable_names", []) or [])
                    dep_data_raw = getattr(ds, "dependent_variable_data", None)
                    dep_units = list(getattr(ds, "dependent_variable_units", []) or [])

                    # ``dependent_variable_data`` may be a single array
                    # (1-D dataset with a single dependent) or a list of
                    # arrays (composite); normalise to a list.
                    if dep_data_raw is None:
                        dep_data = []
                    else:
                        arr = np.asarray(dep_data_raw)
                        if arr.dtype == object or (arr.ndim > 1 and len(dep_names) > 1):
                            dep_data = list(dep_data_raw)
                        else:
                            dep_data = [dep_data_raw]

                    names = ind_names + dep_names
                    arrays = ind_data + dep_data
                    units_list = ind_units + dep_units

                if not names:
                    continue

                # Pad units if shorter
                while len(units_list) < len(names):
                    units_list.append("")

                unit_map = {n.lower(): units_list[i] for i, n in enumerate(names)}

                x_col = _resolve_column(names, self.x)
                y_col = _resolve_column(names, self.y)
                z_col = _resolve_column(names, self.quantity)

                if z_col is None:
                    continue

                zi = names.index(z_col)
                z_arr = np.asarray(arrays[zi], dtype=float)

                # Axis values come from a dataset column, else a scalar
                # measurement condition broadcast over z.
                if x_col is not None:
                    xi = names.index(x_col)
                    x_arr = np.asarray(arrays[xi], dtype=float)
                elif self.x in cond_lookup:
                    x_arr = np.full_like(z_arr, float(cond_lookup[self.x][1]))
                else:
                    continue

                if y_col is not None:
                    yi = names.index(y_col)
                    y_arr = np.asarray(arrays[yi], dtype=float)
                elif self.y in cond_lookup:
                    y_arr = np.full_like(z_arr, float(cond_lookup[self.y][1]))
                else:
                    continue

                if np.iscomplexobj(z_arr):
                    z_arr = np.abs(z_arr)

                mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
                if not mask.any():
                    continue

                x_all.extend(x_arr[mask].tolist())
                y_all.extend(y_arr[mask].tolist())
                z_all.extend(z_arr[mask].tolist())

                # capture units from first successful dataset (unit
                # metadata is only available for dataset-sourced axes;
                # condition-sourced axes fall back to '').
                if not found_x:
                    x_unit = unit_map.get(x_col.lower(), "") if x_col else ""
                    found_x = True
                if not found_y:
                    y_unit = unit_map.get(y_col.lower(), "") if y_col else ""
                    found_y = True
                if not found_quantity:
                    z_unit_candidate = unit_map.get(z_col.lower(), "")
                    if z_unit_candidate:
                        z_unit = z_unit_candidate
                    found_quantity = True

        if not x_all:
            raise ValueError(
                f"No data found for quantity='{self.quantity}', "
                f"x='{self.x}', y='{self.y}'. "
                "Check that these column names exist in the datasets."
            )

        raw_x = np.asarray(x_all, float)
        raw_y = np.asarray(y_all, float)
        raw_z = np.asarray(z_all, float)

        grid, x_grid, y_grid = self._interpolate(raw_x, raw_y, raw_z)

        conditions = _shared_items([dict(getattr(m, "conditions", {}) or {}) for m in measurements])
        details = _shared_items([dict(getattr(m, "details", {}) or {}) for m in measurements])
        details.update(
            {
                "analysis": "kymograph",
                "quantity": self.quantity,
                "x": self.x,
                "y": self.y,
                "n_x": self.n_x,
                "n_y": self.n_y,
                "interpolation": self.interpolation,
                "log_x": self.log_x,
                "log_y": self.log_y,
                "x_min": self.x_min,
                "fill_nearest": self.fill_nearest,
                "n_source_measurements": len(measurements),
                "software": f"hermes-rheo {__version__}; scipy.interpolate.griddata",
            }
        )

        return KymographResult(
            conditions=conditions,
            details=details,
            raw_x=raw_x,
            raw_y=raw_y,
            raw_z=raw_z,
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=grid,
            quantity=self.quantity,
            x_name=self.x,
            y_name=self.y,
            x_unit=x_unit,
            y_unit=y_unit,
            z_unit=z_unit,
            log_y=self.log_y,
            log_x=self.log_x,
        )

    def _interpolate(
        self,
        raw_x: np.ndarray,
        raw_y: np.ndarray,
        raw_z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate scattered points onto a regular grid.

        Returns
        -------
        grid : np.ndarray, shape (n_x, n_y)
        x_grid : np.ndarray, shape (n_x,)
        y_grid : np.ndarray, shape (n_y,)
        """
        keep = np.ones(raw_x.shape, bool)
        if self.log_x:
            keep &= raw_x > 0
        if self.log_y:
            keep &= raw_y > 0
        # Grid and interpolate in log10 space along log axes.
        x_points = np.log10(raw_x[keep]) if self.log_x else raw_x[keep]
        y_points = np.log10(raw_y[keep]) if self.log_y else raw_y[keep]
        z_points = raw_z[keep]

        x_min = x_points.min()
        if self.x_min is not None:
            x_min = np.log10(self.x_min) if self.log_x else float(self.x_min)
        x_grid = np.linspace(x_min, x_points.max(), self.n_x)
        y_grid = np.linspace(y_points.min(), y_points.max(), self.n_y)

        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")  # (n_x, n_y)
        points = np.column_stack([x_points, y_points])
        grid = griddata(points, z_points, (XX, YY), method=self.interpolation)

        if self.fill_nearest and np.any(~np.isfinite(grid)):
            grid_near = griddata(points, z_points, (XX, YY), method="nearest")
            grid = np.where(np.isfinite(grid), grid, grid_near)

        if self.log_x:
            x_grid = 10.0**x_grid
        if self.log_y:
            y_grid = 10.0**y_grid
        return grid, x_grid, y_grid
