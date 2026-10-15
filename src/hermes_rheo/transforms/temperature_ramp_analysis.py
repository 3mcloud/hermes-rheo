"""
Temperature Ramp Analysis Transform
=====================================

Extracts temperature-dependent viscoelastic properties from an
:class:`~piblin.data.data_collections.experiment_set.ExperimentSet`
containing temperature-ramp or temperature-sweep steps.

Designed to be chained after
:class:`~hermes_rheo.transforms.rheo_analysis.RheoAnalysis`::

    pipeline = RheoAnalysis() + TemperatureRampAnalysis(
        steps_to_merge=["Temperature Sweep - 1", "Temperature Sweep - 2"],
        use_temps=[25.0, 100.0],
    )
    result_exp = pipeline(experiment_set)

    # Access the analysis result via the returned ExperimentSet
    ramp = result_exp.result          # TemperatureRampResult instance
    print(f"Tg = {ramp.tg:.1f} °C")
    ramp.to_excel("output.xlsx")

    # Or via the transform instance itself
    ramp_transform = TemperatureRampAnalysis(steps_to_merge=["Temperature Sweep"])
    pipeline = RheoAnalysis() + ramp_transform
    pipeline(experiment_set)
    print(ramp_transform.result.tg)

Step matching
-------------
``steps_to_merge`` is matched case-insensitively as a substring against the
raw ``conditions["method"]`` value of each
:class:`~piblin.data.data_collections.measurement.Measurement`.  For
TRIOS files this value has the form ``"Step name\\tTemperature Ramp - 1"``,
so passing ``steps_to_merge=["Temperature Ramp - 1"]`` or
``steps_to_merge=["Temperature Ramp"]`` both work.

When ``steps_to_merge`` is ``None`` or empty the transform automatically
selects all measurements whose method contains a temperature keyword
(``"temperature sweep"``, ``"temperature ramp"``, etc.).

Variable names
--------------
All variable names and units are read directly from the
:class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
objects produced by the reader.  No column-name candidates are hardcoded.
The result datasets preserve the original names and units exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from scipy.signal import find_peaks

from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
from piblin.data import MeasurementSet, Measurement
import piblin.data.data_collections.experiment_set as experiment_set_module
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset


# ---------------------------------------------------------------------------
# Step-type keywords used for auto-detection when steps_to_merge is empty
# ---------------------------------------------------------------------------
_TEMP_METHOD_KEYWORDS = (
    "temperature sweep",
    "temperature ramp",
    "temperature sweep dma",
    "temperature sweep (multifrequency)",
)


# ---------------------------------------------------------------------------
# Worksheet helper (no pandas dependency)
# ---------------------------------------------------------------------------

def _write_dataset_to_ws(
    ws,
    ds: one_d_composite_dataset.OneDimensionalCompositeDataset,
    start_row: int = 1,
) -> None:
    """Write *ds* to an openpyxl worksheet.

    Layout (relative to *start_row*):

    * Row 0 : variable names from ``ds.data_array_names``.
    * Row 1 : unit strings from ``ds.data_array_units``.
    * Rows 2+ : one row per data point (numeric values).
    """
    names = list(ds.data_array_names)
    units = list(ds.data_array_units)

    for col, (name, unit) in enumerate(zip(names, units), 1):
        ws.cell(row=start_row,     column=col, value=name)
        ws.cell(row=start_row + 1, column=col, value=unit)

    for row_offset, values in enumerate(zip(*ds.data_arrays), 2):
        for col, val in enumerate(values, 1):
            try:
                ws.cell(row=start_row + row_offset, column=col, value=float(val))
            except (TypeError, ValueError):
                ws.cell(row=start_row + row_offset, column=col, value=val)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TemperatureRampResult:
    """Container for all outputs of a temperature-ramp analysis.

    Parameters
    ----------
    dataset : OneDimensionalCompositeDataset or None
        Merged, temperature-sorted dataset containing **all** variables
        present in the selected measurements.  Variable names and units are
        taken directly from the underlying datasets without renaming.
        The default independent variable is ``"temperature"`` and the default
        dependent variable is ``"storage modulus"`` (or the first
        non-temperature variable when storage modulus is absent).
        ``None`` when no matching measurements are found.
    tg : float or None
        Glass transition temperature detected from the peak of the
        tan-delta variable (identified by searching for ``"tan"`` in
        variable names, case-insensitively).  ``None`` when unavailable.
    extracted_dataset : OneDimensionalCompositeDataset or None
        Dataset containing all variables interpolated at the temperatures
        supplied in *use_temps*.  Has the same variable names and units as
        *dataset*.  ``None`` when no temperatures were requested or no
        matching data was found.

    Examples
    --------
    >>> names = result.dataset.data_array_names
    >>> units = result.dataset.data_array_units
    >>> print(result.tg)
    120.3
    """

    dataset: Optional[one_d_composite_dataset.OneDimensionalCompositeDataset]
    tg: Optional[float]
    extracted_dataset: Optional[one_d_composite_dataset.OneDimensionalCompositeDataset]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_excel(self, path: str) -> str:
        """Export the temperature-ramp result to an Excel workbook.

        Two sheets are written:

        * ``Temperature Series`` – the full merged dataset with variable
          names on row 1 and unit strings on row 2.
        * ``Extracted Values`` – Tg (when detected) on rows 1–2, followed
          by the interpolated dataset starting at row 4.

        Parameters
        ----------
        path : str
            Destination file path (should end in ``.xlsx``).

        Returns
        -------
        str
            The resolved path to the saved workbook (same as *path*).

        Examples
        --------
        >>> result.to_excel("results/sample_A.xlsx")
        'results/sample_A.xlsx'
        """
        from openpyxl import Workbook

        wb = Workbook()

        # --- Sheet 1: full temperature series ---
        ws1 = wb.active
        ws1.title = "Temperature Series"
        if self.dataset is not None:
            _write_dataset_to_ws(ws1, self.dataset)

        # --- Sheet 2: extracted values ---
        ws2 = wb.create_sheet("Extracted Values")
        extracted_start = 1

        if self.tg is not None:
            # Determine the temperature unit from the dataset
            temp_unit = ""
            if self.dataset is not None:
                for name, unit in zip(
                    self.dataset.data_array_names, self.dataset.data_array_units
                ):
                    if name.lower() == "temperature":
                        temp_unit = unit
                        break
            ws2.cell(row=1, column=1, value="Tg")
            ws2.cell(row=1, column=2, value=self.tg)
            ws2.cell(row=1, column=3, value=temp_unit)
            extracted_start = 4   # leave a blank row between Tg and the table

        if self.extracted_dataset is not None:
            _write_dataset_to_ws(ws2, self.extracted_dataset, start_row=extracted_start)

        wb.save(path)
        return path


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TemperatureRampAnalysis(MeasurementSetTransform):
    """Extract temperature-dependent viscoelastic properties from an ExperimentSet.

    This transform is the second stage of the standard temperature-ramp
    analysis pipeline::

        pipeline = RheoAnalysis() + TemperatureRampAnalysis(
            steps_to_merge=["Temperature Ramp - 1"],
            use_temps=[25.0, 100.0],
        )
        result_exp = pipeline(experiment_set)
        ramp = result_exp.result

    The transform performs four steps:

    1. **Step selection** – measurements whose ``conditions["method"]``
       contains any of the *steps_to_merge* substrings (case-insensitive)
       are selected.  When *steps_to_merge* is ``None`` or ``[]``, all
       measurements whose method contains a temperature keyword are included.
    2. **Merging** – all
       :class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
       objects from the selected measurements are concatenated and sorted by
       the ``"temperature"`` variable.  Variable names and units come
       directly from the datasets.
    3. **Tg detection** – the glass transition temperature is located at the
       peak of the variable whose name contains ``"tan"``
       (case-insensitive), using prominence-based peak finding.
    4. **Property extraction** – every variable in the merged dataset is
       interpolated at each temperature in *use_temps*, producing a second
       ``OneDimensionalCompositeDataset``.

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
        Temperatures at which all variables are extracted by interpolation.
        Must be in the same unit as the data (typically °C).

    Attributes
    ----------
    result : TemperatureRampResult or None
        Populated after each call to the transform.

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
    >>> from hermes_rheo.transforms.temperature_ramp_analysis import TemperatureRampAnalysis
    >>> from piblin.data import ExperimentSet

    >>> reader = TriosRheoReader()
    >>> mset = reader.data_from_filepath("experiment.txt", create_composite_datasets=True)
    >>> exp = ExperimentSet(mset.measurements)
    >>> pipeline = RheoAnalysis() + TemperatureRampAnalysis(
    ...     steps_to_merge=["Temperature Ramp"],
    ...     use_temps=[25.0, 100.0],
    ... )
    >>> result_exp = pipeline(exp)
    >>> ramp = result_exp.result
    >>> print(f"Tg = {ramp.tg:.1f} °C")
    >>> ramp.to_excel("sample.xlsx")
    """

    def __init__(
        self,
        steps_to_merge: Optional[List[str]] = None,
        use_temps: Optional[List[float]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.steps_to_merge: List[str] = list(steps_to_merge) if steps_to_merge else []
        self.use_temps: List[float] = sorted(use_temps) if use_temps else []
        self.result: Optional[TemperatureRampResult] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _method_matches(self, method: str) -> bool:
        """Return ``True`` if *method* belongs to a step to be analysed.

        Parameters
        ----------
        method : str
            Raw ``conditions["method"]`` value of a Measurement.

        Returns
        -------
        bool
        """
        if not method:
            return False
        ml = method.lower()
        if self.steps_to_merge:
            return any(s.lower() in ml for s in self.steps_to_merge)
        return any(kw in ml for kw in _TEMP_METHOD_KEYWORDS)

    @staticmethod
    def _merge_composite_datasets(
        datasets: List[one_d_composite_dataset.OneDimensionalCompositeDataset],
    ) -> Optional[one_d_composite_dataset.OneDimensionalCompositeDataset]:
        """Merge composite datasets into one, sorted by the temperature variable.

        Only variable names present in **all** supplied datasets are kept,
        preserving the original names and units without any renaming.

        Parameters
        ----------
        datasets : list of OneDimensionalCompositeDataset
            Datasets to merge (typically one per selected measurement).

        Returns
        -------
        OneDimensionalCompositeDataset or None
            Merged dataset sorted by ascending temperature, or ``None``
            when *datasets* is empty or no ``"temperature"`` variable is
            found.
        """
        if not datasets:
            return None

        # Intersect variable names across all datasets (case-insensitive keys,
        # but preserve the original casing from the first dataset)
        first_names: List[str] = list(datasets[0].data_array_names)
        first_names_lower = [n.lower() for n in first_names]

        common_lower: List[str] = list(first_names_lower)
        for ds in datasets[1:]:
            ds_lower = {n.lower() for n in ds.data_array_names}
            common_lower = [n for n in common_lower if n in ds_lower]

        # Canonical names (original casing from first dataset)
        common_names: List[str] = [
            first_names[first_names_lower.index(nl)] for nl in common_lower
        ]
        # Units from the first dataset
        first_units: List[str] = list(datasets[0].data_array_units)
        common_units: List[str] = [
            first_units[first_names_lower.index(nl)] for nl in common_lower
        ]

        # Identify the temperature variable
        temp_idx_in_common = next(
            (i for i, n in enumerate(common_lower) if n == "temperature"),
            None,
        )
        if temp_idx_in_common is None:
            return None

        # Concatenate arrays for each common variable across all datasets
        concatenated: List[np.ndarray] = []
        for nl in common_lower:
            parts: List[np.ndarray] = []
            for ds in datasets:
                ds_names_lower = [n.lower() for n in ds.data_array_names]
                idx = ds_names_lower.index(nl)
                parts.append(np.asarray(ds.data_arrays[idx], dtype=float))
            concatenated.append(np.concatenate(parts))

        # Sort every array by ascending temperature
        sort_order = np.argsort(concatenated[temp_idx_in_common])
        concatenated = [arr[sort_order] for arr in concatenated]

        # Default dependent: prefer "storage modulus", else first non-temperature var
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

    @staticmethod
    def _detect_tg(
        ds: one_d_composite_dataset.OneDimensionalCompositeDataset,
    ) -> Optional[float]:
        """Detect Tg from the peak of the tan-delta variable in *ds*.

        The tan-delta variable is identified by searching for ``"tan"`` in
        the variable names (case-insensitive).  Uses prominence-based peak
        finding; falls back to ``argmax`` when no prominent peak is found.

        Parameters
        ----------
        ds : OneDimensionalCompositeDataset
            Merged temperature-series dataset.

        Returns
        -------
        float or None
            Temperature at the tan-delta peak, or ``None`` when the
            tan-delta or temperature variables are absent or contain
            insufficient finite values.
        """
        names_lower = [n.lower() for n in ds.data_array_names]

        _TAN_EXACT = frozenset(
            {"tan(delta)", "tan(δ)", "tan delta", "tan δ", "loss factor", "loss tangent"}
        )
        tan_idx = next(
            (i for i, n in enumerate(names_lower) if n in _TAN_EXACT),
            next(
                (i for i, n in enumerate(names_lower) if n.startswith("tan(")),
                None,
            ),
        )
        temp_idx = next(
            (i for i, n in enumerate(names_lower) if n == "temperature"), None
        )
        if tan_idx is None or temp_idx is None:
            return None

        tan_arr  = np.asarray(ds.data_arrays[tan_idx],  dtype=float)
        temp_arr = np.asarray(ds.data_arrays[temp_idx], dtype=float)

        valid = np.isfinite(tan_arr) & np.isfinite(temp_arr)
        if valid.sum() < 3:
            return None

        tan_v  = tan_arr[valid]
        temp_v = temp_arr[valid]

        peaks, _ = find_peaks(tan_v, prominence=0.01)
        if peaks.size:
            vals = tan_v[peaks]
            m = np.max(vals)
            sel = peaks[np.abs(vals - m) <= 0.001]
            peak_idx = int(sel[-1])
        else:
            peak_idx = int(np.argmax(tan_v))

        return float(temp_v[peak_idx])

    def _extract_at_temps(
        self,
        ds: one_d_composite_dataset.OneDimensionalCompositeDataset,
    ) -> Optional[one_d_composite_dataset.OneDimensionalCompositeDataset]:
        """Build a dataset with every variable interpolated at ``use_temps``.

        Parameters
        ----------
        ds : OneDimensionalCompositeDataset
            Merged temperature-series dataset.

        Returns
        -------
        OneDimensionalCompositeDataset or None
            Dataset with ``len(use_temps)`` rows.  Variable names and units
            are identical to *ds*.  Returns ``None`` when ``use_temps`` is
            empty.
        """
        if not self.use_temps:
            return None

        names_lower = [n.lower() for n in ds.data_array_names]
        temp_idx = next(
            (i for i, n in enumerate(names_lower) if n == "temperature"), None
        )
        if temp_idx is None:
            return None

        use_arr  = np.asarray(self.use_temps, dtype=float)
        temp_arr = np.asarray(ds.data_arrays[temp_idx], dtype=float)

        extracted: List[np.ndarray] = [
            np.interp(use_arr, temp_arr, np.asarray(arr, dtype=float))
            for arr in ds.data_arrays
        ]
        # The temperature column should exactly equal use_temps
        extracted[temp_idx] = use_arr.copy()

        return one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=extracted,
            data_array_names=list(ds.data_array_names),
            data_array_units=list(ds.data_array_units),
            default_independent_name=ds.data_array_names[temp_idx],
            default_dependent_name=ds.dependent_variable_name,
            source="Values at selected temperatures",
        )

    @staticmethod
    def _get_measurements(target: Any) -> List[Measurement]:
        """Return a flat list of all measurements from *target*."""
        if isinstance(target, (experiment_set_module.ExperimentSet, MeasurementSet)):
            return list(target.measurements)
        if hasattr(target, "measurements"):
            return list(target.measurements)
        return list(target)

    # ------------------------------------------------------------------
    # MeasurementSetTransform interface
    # ------------------------------------------------------------------

    def _apply(self, target: Any, **kwargs: Any) -> Any:
        """Run the temperature-ramp analysis on *target*.

        Parameters
        ----------
        target : MeasurementSet or ExperimentSet
            Typically the ExperimentSet produced by ``RheoAnalysis()``.

        Returns
        -------
        target
            The same object with ``.result`` and
            ``details["temperature_ramp_result"]`` set.
        """
        all_measurements = self._get_measurements(target)

        # --- 1. Filter to temperature-type steps ---
        selected: List[Measurement] = [
            m for m in all_measurements
            if self._method_matches(str(m.conditions.get("method", "")))
        ]

        # --- 2. Collect composite datasets ---
        composite_datasets = [
            ds
            for meas in selected
            for ds in meas.datasets
            if hasattr(ds, "data_array_names")
        ]

        # --- 3. Merge ---
        merged = self._merge_composite_datasets(composite_datasets)

        # --- 4. Tg detection ---
        tg = self._detect_tg(merged) if merged is not None else None

        # --- 5. Extraction at requested temperatures ---
        extracted = self._extract_at_temps(merged) if merged is not None else None

        # --- Build result ---
        ramp_result = TemperatureRampResult(
            dataset=merged,
            tg=tg,
            extracted_dataset=extracted,
        )

        self.result = ramp_result

        # Expose result attributes directly on the returned experiment so callers can write:
        #   result_exp = pipeline(experiment)
        #   result_exp.tg  /  result_exp.dataset  /  result_exp.extracted_dataset
        try:
            target.result = ramp_result
            target.tg = tg
            target.dataset = merged
            target.extracted_dataset = extracted
        except (AttributeError, TypeError):
            pass
        try:
            if hasattr(target, "details") and isinstance(target.details, dict):
                target.details["temperature_ramp_result"] = ramp_result
        except Exception:
            pass

        return target
