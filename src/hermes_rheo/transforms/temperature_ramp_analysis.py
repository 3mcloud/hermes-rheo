"""Temperature-ramp analysis: merge temperature steps, detect Tg, extract values.

Tg is taken at the most prominent tan(delta) peak (default), at the onset of
the storage-modulus drop (TA Instruments application note RH100), or both
(``tg_method``). The test frequency is read from the data.

Works on an :class:`~piblin.data.data_collections.experiment_set.ExperimentSet`
(or measurement set) containing temperature-ramp or temperature-sweep steps,
and is meant to be chained after
:class:`~hermes_rheo.transforms.rheo_analysis.RheoAnalysis`::

    pipeline = RheoAnalysis() + TemperatureRampAnalysis(
        steps_to_merge=["Temperature Sweep - 1", "Temperature Sweep - 2"],
        use_temps=[25.0, 100.0],
    )
    result_exp = pipeline(experiment_set)

    ramp = result_exp                 # TemperatureRampMeasurement (piblin Measurement)
    print(ramp.summary())             # frequency and Tg values
    ramp.to_excel("output.xlsx")

The result is a piblin :class:`~piblin.data.data_collections.measurement.Measurement`
holding the merged dataset and the dataset interpolated at ``use_temps``;
Tg, the parameters and the method are in its ``details``. A
:class:`TemperatureRampResult` view is also available as ``ramp.result`` and
``transform.result``.

Step matching
-------------
``steps_to_merge`` entries are matched case-insensitively as substrings of
each measurement's ``conditions["method"]``. For TRIOS files this value has
the form ``"Step name\\tTemperature Ramp - 1"``, so both
``"Temperature Ramp - 1"`` and ``"Temperature Ramp"`` match. When
``steps_to_merge`` is empty, measurements whose method contains a temperature
keyword (``"temperature sweep"``, ``"temperature ramp"``, ...) are used.

Variable names and units are taken from the
:class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
objects and are preserved unchanged in the output datasets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.data import Measurement
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
from scipy.signal import find_peaks

from hermes_rheo import __version__

_TEMP_METHOD_KEYWORDS = (
    "temperature sweep",
    "temperature ramp",
    "temperature sweep dma",
    "temperature sweep (multifrequency)",
)

TG_METHODS = ("tan_delta", "storage_modulus", "both")

_TAN_DELTA_NAMES = frozenset(
    {"tan(delta)", "tan(δ)", "tan delta", "tan δ", "loss factor", "loss tangent"}
)


def _shared_items(dicts: list[dict]) -> dict:
    """Return the key/value pairs identical across all *dicts*."""
    if not dicts:
        return {}
    shared = dict(dicts[0])
    for other in dicts[1:]:
        shared = {k: v for k, v in shared.items() if k in other and np.array_equal(other[k], v)}
    return shared


def _write_dataset_to_ws(
    ws,
    ds: one_d_composite_dataset.OneDimensionalCompositeDataset,
    start_row: int = 1,
) -> None:
    """Write *ds* to an openpyxl worksheet.

    Variable names go in row *start_row*, units in the next row, and one
    row per data point follows.
    """
    names = list(ds.data_array_names)
    units = list(ds.data_array_units)

    for col, (name, unit) in enumerate(zip(names, units), 1):
        ws.cell(row=start_row, column=col, value=name)
        ws.cell(row=start_row + 1, column=col, value=unit)

    for row_offset, values in enumerate(zip(*ds.data_arrays), 2):
        for col, val in enumerate(values, 1):
            try:
                ws.cell(row=start_row + row_offset, column=col, value=float(val))
            except (TypeError, ValueError):
                ws.cell(row=start_row + row_offset, column=col, value=val)


def _array(ds, names: tuple[str, ...] | frozenset[str]) -> tuple[np.ndarray, str, str] | None:
    """Return ``(values, name, unit)`` of the first array of *ds* matching *names*."""
    for name, unit, values in zip(ds.data_array_names, ds.data_array_units, ds.data_arrays):
        if name.lower() in names:
            return np.asarray(values, dtype=float), name, unit
    return None


@dataclass
class TanDeltaPeaks:
    """Peaks of tan(delta) against temperature.

    Attributes
    ----------
    temperatures, heights, prominences : numpy.ndarray
        Temperature, tan(delta) value and prominence of each peak, in order of
        increasing temperature.
    primary : int
        Index of the most prominent peak, taken as Tg.
    """

    temperatures: np.ndarray
    heights: np.ndarray
    prominences: np.ndarray
    primary: int

    @property
    def tg(self) -> float:
        """Temperature of the most prominent peak."""
        return float(self.temperatures[self.primary])


def _uniform(temperature: np.ndarray, values: np.ndarray, smoothing: float):
    """Sort, interpolate on a uniform grid and Savitzky-Golay smooth *values*.

    Returns the grid, the interpolated values, the smoothed values, the grid
    step and the window length (points).
    """
    from scipy.signal import savgol_filter

    order = np.argsort(temperature)
    temp, vals = temperature[order], values[order]
    step = float(np.median(np.diff(np.unique(temp))))
    grid = np.arange(temp[0], temp[-1] + 0.5 * step, step)
    vals = np.interp(grid, temp, vals)
    window = min(max(5, int(round(smoothing / step)) | 1), len(grid) - (1 - len(grid) % 2))
    return grid, vals, savgol_filter(vals, window, 2), step, window


def find_tan_delta_peaks(
    temperature: np.ndarray,
    tan_delta: np.ndarray,
    prominence: float = 0.03,
    max_width_ratio: float = 1.5,
    smoothing: float = 5.0,
) -> TanDeltaPeaks | None:
    """Find the tan(delta) peaks of a temperature ramp.

    tan(delta) is smoothed over *smoothing* degrees (Savitzky-Golay,
    quadratic) to remove noise and the overlaps between ramp steps, and peaks
    are found with :func:`scipy.signal.find_peaks`. The most prominent peak is
    Tg. A further peak, e.g. a second phase or a secondary relaxation, is kept
    when its prominence is at least *prominence* times the tan(delta) range
    and its width at half prominence is at most *max_width_ratio* times that
    of the Tg peak; broad rises of tan(delta) towards flow are rejected. Peak
    temperatures are the maxima of the measured tan(delta) within half a
    smoothing window of the smoothed peaks.

    Parameters
    ----------
    temperature, tan_delta : numpy.ndarray
        Ramp data; non-finite points are ignored.
    prominence : float, default 0.03
        Minimum peak prominence as a fraction of ``max - min`` of tan(delta).
    max_width_ratio : float, default 1.5
        Maximum width of a further peak relative to the Tg peak.
    smoothing : float, default 5.0
        Smoothing window, in temperature units.

    Returns
    -------
    TanDeltaPeaks or None
        ``None`` with fewer than five finite points. Without an interior
        peak, the maximum of tan(delta) is returned with zero prominence.
    """
    from scipy.signal import peak_widths

    valid = np.isfinite(temperature) & np.isfinite(tan_delta)
    if valid.sum() < 5:
        return None
    temp, tan = temperature[valid], tan_delta[valid]
    grid, _, smooth, step, _ = _uniform(temp, tan, smoothing)
    peaks, props = find_peaks(smooth, prominence=prominence * np.ptp(smooth))
    if peaks.size == 0:
        i = int(np.argmax(tan))
        return TanDeltaPeaks(temp[[i]], tan[[i]], np.zeros(1), 0)

    prominences = props["prominences"]
    widths = peak_widths(
        smooth,
        peaks,
        rel_height=0.5,
        prominence_data=(prominences, props["left_bases"], props["right_bases"]),
    )[0]
    keep = widths <= max_width_ratio * widths[np.argmax(prominences)]
    peaks, prominences = peaks[keep], prominences[keep]

    temperatures, heights = [], []
    for peak in peaks:
        near = np.flatnonzero(np.abs(temp - grid[peak]) <= max(smoothing / 2, step))
        k = near[np.argmax(tan[near])] if near.size else int(np.argmin(np.abs(temp - grid[peak])))
        temperatures.append(temp[k])
        heights.append(tan[k])
    return TanDeltaPeaks(
        np.asarray(temperatures), np.asarray(heights), prominences, int(np.argmax(prominences))
    )


@dataclass
class StorageModulusOnset:
    """Construction of the onset Tg from the drop of the storage modulus.

    Following TA Instruments application note RH100, the onset Tg is where the
    tangent to the glassy plateau of log10 G' meets the tangent at the
    inflection point, the steepest decrease of log10 G' (minimum of
    d log10 G'/dT).

    Attributes
    ----------
    temperature : numpy.ndarray
        Uniform temperature grid of the analysis range.
    log_modulus, slope : numpy.ndarray
        Smoothed log10 G' and its derivative with respect to temperature.
    glassy_range : tuple of float
        Temperatures over which the glassy tangent is fitted.
    glassy_line, inflection_line : tuple of float
        ``(slope, intercept)`` of the two tangents of log10 G' against T.
    t_inflection, tg_onset : float
        Inflection temperature and onset Tg (intersection of the tangents).
    """

    temperature: np.ndarray
    log_modulus: np.ndarray
    slope: np.ndarray
    glassy_range: tuple[float, float]
    glassy_line: tuple[float, float]
    inflection_line: tuple[float, float]
    t_inflection: float
    tg_onset: float


def storage_modulus_onset(
    temperature: np.ndarray,
    storage_modulus: np.ndarray,
    temperature_range: tuple[float, float] | None = None,
    glassy_fraction: float = 0.125,
    smoothing: float = 5.0,
) -> StorageModulusOnset | None:
    """Onset Tg from the intersection of the glassy and inflection tangents.

    log10 G' is interpolated on a uniform temperature grid and smoothed with a
    Savitzky-Golay filter (quadratic, window *smoothing* wide), which also
    gives the derivative. The glassy tangent is a straight-line fit over the
    first *glassy_fraction* of the analysis range (12.5 % is the TRIOS
    default); the inflection tangent passes through the derivative minimum.

    Parameters
    ----------
    temperature, storage_modulus : numpy.ndarray
        Ramp data; non-finite and non-positive moduli are ignored.
    temperature_range : tuple of float, optional
        Analysis range; it must start on the glassy plateau and contain the
        inflection. Default: from the first data point to as far above the
        steepest decrease of G' as that lies above the first point.
    glassy_fraction : float, default 0.125
        Fraction of the analysis range used for the glassy tangent.
    smoothing : float, default 5.0
        Width of the smoothing window, in temperature units.

    Returns
    -------
    StorageModulusOnset or None
        ``None`` if fewer than ten points lie in the analysis range.
    """
    from scipy.signal import savgol_filter

    valid = np.isfinite(temperature) & np.isfinite(storage_modulus) & (storage_modulus > 0)
    if temperature_range is not None:
        valid &= (temperature >= min(temperature_range)) & (temperature <= max(temperature_range))
    if valid.sum() < 10:
        return None
    grid, log_g, smooth, step, window = _uniform(
        temperature[valid], np.log10(storage_modulus[valid]), smoothing
    )
    slope = savgol_filter(log_g, window, 2, deriv=1, delta=step)

    edge = window // 2 if len(grid) > 3 * window else 0
    i = edge + int(np.argmin(slope[edge : len(grid) - edge]))
    t_inflection = float(grid[i])
    inflection_line = (float(slope[i]), float(smooth[i] - slope[i] * grid[i]))

    range_end = grid[-1] if temperature_range is not None else min(grid[-1], 2 * grid[i] - grid[0])
    glassy = grid <= max(grid[0] + glassy_fraction * (range_end - grid[0]), grid[1])
    glassy_line = tuple(float(v) for v in np.polyfit(grid[glassy], smooth[glassy], 1))
    denominator = glassy_line[0] - inflection_line[0]
    tg_onset = float((inflection_line[1] - glassy_line[1]) / denominator) if denominator else np.nan
    if glassy_line[0] < 0.5 * inflection_line[0] or not grid[0] <= tg_onset <= t_inflection:
        warnings.warn(
            "The glassy tangent is almost as steep as the inflection tangent: the data do "
            "not start on the glassy plateau and the onset Tg is not reliable (NaN if the "
            "tangents do not meet below the inflection). Set the analysis range with "
            "temperature_range / onset_range.",
            stacklevel=2,
        )
        if not grid[0] <= tg_onset <= t_inflection:
            tg_onset = np.nan
    return StorageModulusOnset(
        grid,
        smooth,
        slope,
        (float(grid[0]), float(grid[glassy][-1])),
        glassy_line,
        inflection_line,
        t_inflection,
        tg_onset,
    )


@dataclass
class TemperatureRampResult:
    """Outputs of a temperature-ramp analysis.

    Parameters
    ----------
    dataset : OneDimensionalCompositeDataset or None
        Merged dataset sorted by temperature, with all variables common to
        the selected measurements (names and units unchanged). The default
        independent variable is ``"temperature"``; the default dependent
        variable is ``"storage modulus"`` or, if absent, the first
        non-temperature variable. ``None`` if no measurements matched.
    tg : float or None
        Glass transition temperature from the selected method (the tan-delta
        peak by default). ``None`` when it cannot be determined.
    extracted_dataset : OneDimensionalCompositeDataset or None
        Temperature and the selected properties interpolated at the requested
        temperatures, with the names and units of *dataset*. ``None`` when no
        temperatures were requested.
    tg_values : dict, optional
        Every Tg computed, by label (e.g. ``"Tg (tan(delta) peak)"``).

    Examples
    --------
    >>> result.dataset.data_array_names  # doctest: +SKIP
    >>> result.tg  # doctest: +SKIP
    120.3
    """

    dataset: one_d_composite_dataset.OneDimensionalCompositeDataset | None
    tg: float | None
    extracted_dataset: one_d_composite_dataset.OneDimensionalCompositeDataset | None
    tg_values: dict[str, float] | None = None

    def to_excel(self, path: str) -> str:
        """Write the result to an Excel workbook.

        Sheet ``Temperature Series`` holds the merged dataset (names, units,
        values). Sheet ``Extracted Values`` holds one row per Tg (label, value,
        unit), then, after a blank row, the interpolated dataset.

        Parameters
        ----------
        path : str
            Destination file path (``.xlsx``).

        Returns
        -------
        str
            *path*.
        """
        from openpyxl import Workbook

        wb = Workbook()

        ws1 = wb.active
        ws1.title = "Temperature Series"
        if self.dataset is not None:
            _write_dataset_to_ws(ws1, self.dataset)

        ws2 = wb.create_sheet("Extracted Values")
        extracted_start = 1

        tg_values = self.tg_values or ({"Tg": self.tg} if self.tg is not None else {})
        if tg_values:
            temp_unit = ""
            if self.dataset is not None:
                for name, unit in zip(self.dataset.data_array_names, self.dataset.data_array_units):
                    if name.lower() == "temperature":
                        temp_unit = unit
                        break
            for row, (label, value) in enumerate(tg_values.items(), 1):
                ws2.cell(row=row, column=1, value=label)
                ws2.cell(row=row, column=2, value=value)
                ws2.cell(row=row, column=3, value=temp_unit)
            extracted_start = len(tg_values) + 3

        if self.extracted_dataset is not None:
            _write_dataset_to_ws(ws2, self.extracted_dataset, start_row=extracted_start)

        wb.save(path)
        return path


class TemperatureRampMeasurement(Measurement):
    """Piblin measurement holding the outputs of a temperature-ramp analysis.

    Datasets are the merged temperature series and, when temperatures were
    requested, the values interpolated at those temperatures. Conditions and
    details shared by the selected source measurements are carried over;
    additional details hold the analysis parameters (``steps_to_merge``,
    ``use_temps``, ``properties``, ``tg_method``), the test frequency
    (``angular_frequency``, ``frequency`` and their units), the glass
    transition temperatures (``tg``, ``tg_tan_delta``, ``tg_tan_delta_peaks``,
    ``tg_onset``, ``t_inflection``, ``tg_unit``), the method
    (``analysis_method``) and the software.

    Parameters
    ----------
    dataset, extracted_dataset : OneDimensionalCompositeDataset or None
        Merged and interpolated datasets.
    tg : float or None
        Glass transition temperature from the selected method.
    conditions, details : dict, optional
        Metadata of the source measurements and analysis parameters.
    tan_delta_peaks : TanDeltaPeaks, optional
        All tan(delta) peaks, when the tan(delta) method was used.
    onset : StorageModulusOnset, optional
        The storage-modulus onset construction, when that method was used.
    """

    def __init__(
        self,
        dataset,
        extracted_dataset,
        tg,
        conditions=None,
        details=None,
        tan_delta_peaks: TanDeltaPeaks | None = None,
        onset: StorageModulusOnset | None = None,
    ):
        self.dataset = dataset
        self.extracted_dataset = extracted_dataset
        self.tg = tg
        self.tan_delta_peaks = tan_delta_peaks
        self.onset = onset
        datasets = [d for d in (dataset, extracted_dataset) if d is not None]
        super().__init__(datasets=datasets, conditions=conditions, details=details)

    @property
    def tg_values(self) -> dict[str, float]:
        """Every Tg computed, by label."""
        values = {}
        if self.tan_delta_peaks is not None:
            values["Tg (tan(delta) peak)"] = self.tan_delta_peaks.tg
        if self.onset is not None:
            values["Tg (storage modulus onset)"] = self.onset.tg_onset
            values["Storage modulus inflection"] = self.onset.t_inflection
        return values

    def summary(self) -> str:
        """Test frequency and glass transition temperatures as text."""
        d = self.details
        unit = d.get("tg_unit", "")
        lines = []
        if d.get("angular_frequency") is not None:
            lines.append(
                f"Angular frequency: {d['angular_frequency']:.3g} {d['angular_frequency_unit']}"
            )
        if d.get("frequency") is not None:
            lines.append(f"Frequency: {d['frequency']:.3g} {d['frequency_unit']}")
        lines += [f"{label}: {value:.1f} {unit}" for label, value in self.tg_values.items()]
        if self.tan_delta_peaks is not None and len(self.tan_delta_peaks.temperatures) > 1:
            peaks = ", ".join(f"{t:.1f}" for t in self.tan_delta_peaks.temperatures)
            lines.append(f"tan(delta) peaks: {peaks} {unit}")
        return "\n".join(lines)

    def plot_onset(self, **kwargs):
        """Plot the storage-modulus onset construction.

        Thin wrapper around
        :func:`hermes_rheo.visualization.temperature_ramp.plot_tg_onset`.
        """
        from hermes_rheo.visualization.temperature_ramp import plot_tg_onset

        return plot_tg_onset(self, **kwargs)

    @property
    def result(self) -> TemperatureRampResult:
        """The outputs as a :class:`TemperatureRampResult`."""
        return TemperatureRampResult(
            self.dataset, self.tg, self.extracted_dataset, self.tg_values or None
        )

    def to_excel(self, path: str) -> str:
        """Write the outputs to an Excel workbook; see :meth:`TemperatureRampResult.to_excel`."""
        return self.result.to_excel(path)


class TemperatureRampAnalysis(MeasurementSetTransform):
    """Merge temperature-ramp steps, detect Tg and interpolate properties.

    Steps:

    1. Select measurements whose ``conditions["method"]`` contains any
       *steps_to_merge* substring (case-insensitive), or a temperature
       keyword when *steps_to_merge* is empty.
    2. Concatenate their
       :class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
       objects and sort by ``"temperature"``. Only variables present in all
       datasets are kept.
    3. Determine Tg with *tg_method*: the most prominent tan(delta) peak
       (:func:`find_tan_delta_peaks`, all peaks are reported), the onset of
       the storage-modulus drop (:func:`storage_modulus_onset`), or both.
    4. Read the test frequency from the ``"angular frequency"`` and
       ``"frequency"`` arrays (median values).
    5. Linearly interpolate temperature and the requested *properties* at
       each temperature in *use_temps*.

    Parameters
    ----------
    steps_to_merge : list of str or None, optional
        Substrings matched case-insensitively against the measurement
        ``conditions["method"]`` to identify steps to include.  Typical
        values are step names as they appear in the TRIOS file, e.g.
        ``["Temperature Ramp - 1", "Temperature Ramp - 2"]``.  When
        ``None`` or ``[]``, temperature-ramp/sweep steps are detected
        automatically via keyword matching.
    use_temps : list of float or None, optional
        Temperatures at which to interpolate, in the unit of the data
        (typically °C).
    properties : list of str or None, optional
        Variables to extract (case-insensitive). Temperature is always
        included. When ``None``, all variables are extracted.
    tg_method : {"tan_delta", "storage_modulus", "both"}, default "tan_delta"
        How Tg is determined. With ``"both"``, ``tg`` is the tan(delta) peak
        and the onset is reported alongside it.
    tan_delta_prominence : float, default 0.03
        Minimum prominence of a further tan(delta) peak, as a fraction of the
        tan(delta) range; see :func:`find_tan_delta_peaks`.
    tan_delta_max_width : float, default 1.5
        Maximum width of a further tan(delta) peak relative to the Tg peak.
    onset_range : tuple of float, optional
        Temperature range of the storage-modulus analysis; it must start on
        the glassy plateau and contain the inflection. Default: from the first
        point, symmetric about the inflection.
    glassy_fraction : float, default 0.125
        Fraction of the onset range used for the glassy tangent.
    smoothing : float, default 5.0
        Smoothing window of tan(delta) and log10 G', in temperature units.

    Attributes
    ----------
    result : TemperatureRampResult or None
        View of the last output; the returned object is a
        :class:`TemperatureRampMeasurement`.

    Returns
    -------
    TemperatureRampMeasurement
        From ``apply_to``: a piblin measurement with the merged dataset and
        the dataset interpolated at ``use_temps``; details hold Tg, the
        parameters, the method and software.

    References
    ----------
    .. [1] Whitcomb, K., "Measurement of Glass Transition Temperatures by
       Dynamic Mechanical Analysis and Rheology", TA Instruments application
       note RH100. https://www.tainstruments.com/
    .. [2] Virtanen, P. *et al.*, "SciPy 1.0: fundamental algorithms for
       scientific computing in Python", Nature Methods 17, 261-272 (2020).
       https://doi.org/10.1038/s41592-019-0686-2

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
    >>> from hermes_rheo.transforms.temperature_ramp_analysis import TemperatureRampAnalysis
    >>> from piblin.data import ExperimentSet

    >>> reader = TriosRheoReader()
    >>> mset = reader.data_from_filepath(
    ...     "experiment.txt", create_composite_datasets=True
    ... )  # doctest: +SKIP
    >>> exp = ExperimentSet(mset.measurements)  # doctest: +SKIP
    >>> pipeline = RheoAnalysis() + TemperatureRampAnalysis(
    ...     steps_to_merge=["Temperature Ramp"],
    ...     use_temps=[25.0, 100.0],
    ...     properties=["storage modulus", "loss modulus", "tan(delta)"],
    ... )
    >>> ramp = pipeline(exp)  # doctest: +SKIP
    >>> ramp.tg  # doctest: +SKIP
    >>> ramp.to_excel("sample.xlsx")  # doctest: +SKIP
    """

    def __init__(
        self,
        steps_to_merge: list[str] | None = None,
        use_temps: list[float] | None = None,
        properties: list[str] | None = None,
        *args: Any,
        tg_method: str = "tan_delta",
        tan_delta_prominence: float = 0.03,
        tan_delta_max_width: float = 1.5,
        onset_range: tuple[float, float] | None = None,
        glassy_fraction: float = 0.125,
        smoothing: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if tg_method not in TG_METHODS:
            raise ValueError(f"tg_method must be one of {TG_METHODS}, got {tg_method!r}")
        self.steps_to_merge: list[str] = list(steps_to_merge) if steps_to_merge else []
        self.use_temps: list[float] = sorted(use_temps) if use_temps else []
        self.properties: list[str] | None = list(properties) if properties is not None else None
        self.tg_method = tg_method
        self.tan_delta_prominence = tan_delta_prominence
        self.tan_delta_max_width = tan_delta_max_width
        self.onset_range = onset_range
        self.glassy_fraction = glassy_fraction
        self.smoothing = smoothing
        self.result: TemperatureRampResult | None = None

    def _method_matches(self, method: str) -> bool:
        """Return whether a ``conditions["method"]`` value selects a step."""
        if not method:
            return False
        ml = method.lower()
        if self.steps_to_merge:
            return any(s.lower() in ml for s in self.steps_to_merge)
        return any(kw in ml for kw in _TEMP_METHOD_KEYWORDS)

    @staticmethod
    def _merge_composite_datasets(
        datasets: list[one_d_composite_dataset.OneDimensionalCompositeDataset],
    ) -> one_d_composite_dataset.OneDimensionalCompositeDataset | None:
        """Merge datasets into one, sorted by ascending temperature.

        Only variables present in all datasets (matched case-insensitively)
        are kept, with the names and units of the first dataset.

        Returns
        -------
        OneDimensionalCompositeDataset or None
            ``None`` if *datasets* is empty or has no ``"temperature"``
            variable.
        """
        if not datasets:
            return None

        first_names: list[str] = list(datasets[0].data_array_names)
        first_names_lower = [n.lower() for n in first_names]

        common_lower: list[str] = list(first_names_lower)
        for ds in datasets[1:]:
            ds_lower = {n.lower() for n in ds.data_array_names}
            common_lower = [n for n in common_lower if n in ds_lower]

        common_names: list[str] = [first_names[first_names_lower.index(nl)] for nl in common_lower]
        first_units: list[str] = list(datasets[0].data_array_units)
        common_units: list[str] = [first_units[first_names_lower.index(nl)] for nl in common_lower]

        temp_idx_in_common = next(
            (i for i, n in enumerate(common_lower) if n == "temperature"),
            None,
        )
        if temp_idx_in_common is None:
            return None

        concatenated: list[np.ndarray] = []
        for nl in common_lower:
            parts: list[np.ndarray] = []
            for ds in datasets:
                ds_names_lower = [n.lower() for n in ds.data_array_names]
                idx = ds_names_lower.index(nl)
                parts.append(np.asarray(ds.data_arrays[idx], dtype=float))
            concatenated.append(np.concatenate(parts))

        sort_order = np.argsort(concatenated[temp_idx_in_common])
        concatenated = [arr[sort_order] for arr in concatenated]

        dep_name = next(
            (n for n in common_names if n.lower() == "storage modulus"),
            next(
                (n for n in common_names if n.lower() != "temperature"),
                common_names[0],
            ),
        )

        return one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=concatenated,
            data_array_names=common_names,
            data_array_units=common_units,
            default_independent_name=common_names[temp_idx_in_common],
            default_dependent_name=dep_name,
            source="Merged temperature-ramp data",
        )

    def _analyse(self, ds) -> tuple[TanDeltaPeaks | None, StorageModulusOnset | None, dict]:
        """Tan(delta) peaks, storage-modulus onset and test frequency of *ds*."""
        temperature = _array(ds, ("temperature",))
        peaks = onset = None
        if temperature is not None and self.tg_method in ("tan_delta", "both"):
            tan_delta = _array(ds, _TAN_DELTA_NAMES)
            if tan_delta is None:
                tan_delta = next(
                    (
                        (np.asarray(v, dtype=float), n, u)
                        for n, u, v in zip(ds.data_array_names, ds.data_array_units, ds.data_arrays)
                        if n.lower().startswith("tan(")
                    ),
                    None,
                )
            if tan_delta is not None:
                peaks = find_tan_delta_peaks(
                    temperature[0],
                    tan_delta[0],
                    self.tan_delta_prominence,
                    self.tan_delta_max_width,
                    self.smoothing,
                )
        if temperature is not None and self.tg_method in ("storage_modulus", "both"):
            storage = _array(ds, ("storage modulus",))
            if storage is not None:
                onset = storage_modulus_onset(
                    temperature[0],
                    storage[0],
                    self.onset_range,
                    self.glassy_fraction,
                    self.smoothing,
                )

        frequency: dict[str, Any] = {}
        for key, name in (("angular_frequency", "angular frequency"), ("frequency", "frequency")):
            found = _array(ds, (name,))
            values = None if found is None else found[0][np.isfinite(found[0])]
            if values is not None and values.size:
                frequency[key] = float(np.median(values))
                frequency[f"{key}_unit"] = found[2]
                if np.ptp(values) > 0.01 * abs(frequency[key]):
                    warnings.warn(
                        f"The {name} varies during the ramp; the median is reported.",
                        stacklevel=3,
                    )
        return peaks, onset, frequency

    def _extract_at_temps(
        self,
        ds: one_d_composite_dataset.OneDimensionalCompositeDataset,
    ) -> one_d_composite_dataset.OneDimensionalCompositeDataset | None:
        """Interpolate the selected variables of *ds* at ``use_temps``.

        Returns
        -------
        OneDimensionalCompositeDataset or None
            One row per requested temperature; ``None`` when ``use_temps``
            is empty or *ds* has no temperature variable.

        Raises
        ------
        ValueError
            If ``properties`` is empty or names unknown variables.
        """
        if not self.use_temps:
            return None

        names_lower = [n.lower() for n in ds.data_array_names]
        temp_idx = next((i for i, n in enumerate(names_lower) if n == "temperature"), None)
        if temp_idx is None:
            return None

        use_arr = np.asarray(self.use_temps, dtype=float)
        temp_arr = np.asarray(ds.data_arrays[temp_idx], dtype=float)

        selected_indices = list(range(len(ds.data_arrays)))
        if self.properties is not None:
            if not self.properties:
                raise ValueError("properties must contain at least one variable name")
            indices_by_name = {name.lower(): index for index, name in enumerate(names_lower)}
            unknown_properties = [
                property_name
                for property_name in self.properties
                if property_name.lower() not in indices_by_name
            ]
            if unknown_properties:
                raise ValueError(
                    f"Requested properties are not available: {unknown_properties}. "
                    f"Available properties: {list(ds.data_array_names)}"
                )
            selected_indices = [temp_idx]
            for property_name in self.properties:
                property_index = indices_by_name[property_name.lower()]
                if property_index not in selected_indices:
                    selected_indices.append(property_index)

        extracted = [
            use_arr.copy()
            if index == temp_idx
            else np.interp(use_arr, temp_arr, np.asarray(ds.data_arrays[index], dtype=float))
            for index in selected_indices
        ]

        return one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=extracted,
            data_array_names=[ds.data_array_names[index] for index in selected_indices],
            data_array_units=[ds.data_array_units[index] for index in selected_indices],
            default_independent_name=ds.data_array_names[temp_idx],
            default_dependent_name=(
                ds.dependent_variable_name
                if ds.dependent_variable_name
                in [ds.data_array_names[index] for index in selected_indices]
                else ds.data_array_names[
                    next(index for index in selected_indices if index != temp_idx)
                ]
            ),
            source="Values at selected temperatures",
        )

    @staticmethod
    def _get_measurements(target: Any) -> list[Measurement]:
        """Return the measurements of *target* (a set or an iterable)."""
        if hasattr(target, "measurements"):
            return list(target.measurements)
        return list(target)

    def _apply(self, target: Any, **kwargs: Any) -> TemperatureRampMeasurement:
        """Run the analysis on *target* (typically the output of ``RheoAnalysis``).

        Returns
        -------
        TemperatureRampMeasurement
            Measurement with the merged and extracted datasets; Tg,
            parameters, method and software are in its details. Without
            matching steps it has no datasets and ``tg`` is None.
        """
        all_measurements = self._get_measurements(target)

        selected: list[Measurement] = [
            m for m in all_measurements if self._method_matches(str(m.conditions.get("method", "")))
        ]

        composite_datasets = [
            ds for meas in selected for ds in meas.datasets if hasattr(ds, "data_array_names")
        ]

        merged = self._merge_composite_datasets(composite_datasets)
        peaks, onset, frequency = self._analyse(merged) if merged is not None else (None, None, {})
        tg = (
            onset.tg_onset
            if self.tg_method == "storage_modulus" and onset is not None
            else (peaks.tg if self.tg_method != "storage_modulus" and peaks is not None else None)
        )
        extracted = self._extract_at_temps(merged) if merged is not None else None

        conditions = _shared_items([m.conditions for m in selected])
        details = _shared_items([m.details for m in selected])
        tg_unit = ""
        if merged is not None:
            for name, unit in zip(merged.data_array_names, merged.data_array_units, strict=False):
                if name.lower() == "temperature":
                    tg_unit = unit
        for key in conditions:
            details.pop(key, None)
        details.update(
            {
                "analysis_method": (
                    "Merge of temperature steps; Tg as the most prominent peak of the "
                    f"smoothed tan(delta) ({self.smoothing} degree Savitzky-Golay window; "
                    "further peaks with scipy.signal.find_peaks, prominence "
                    f"{self.tan_delta_prominence} of the range, width up to "
                    f"{self.tan_delta_max_width} times the Tg peak) and/or as the "
                    "storage-modulus onset "
                    "(intersection of the glassy and inflection tangents of log10 G', "
                    "TA Instruments RH100); linear interpolation at use_temps"
                ),
                "software": f"hermes-rheo {__version__}; scipy",
                "steps_to_merge": list(self.steps_to_merge),
                "use_temps": list(self.use_temps),
                "properties": None if self.properties is None else list(self.properties),
                "tg_method": self.tg_method,
                **frequency,
                "tg": tg,
                "tg_tan_delta": None if peaks is None else peaks.tg,
                "tg_tan_delta_peaks": None if peaks is None else peaks.temperatures.tolist(),
                "tg_onset": None if onset is None else onset.tg_onset,
                "t_inflection": None if onset is None else onset.t_inflection,
                "tg_unit": tg_unit,
            }
        )

        result = TemperatureRampMeasurement(
            merged, extracted, tg, conditions, details, tan_delta_peaks=peaks, onset=onset
        )
        self.result = result.result
        return result
