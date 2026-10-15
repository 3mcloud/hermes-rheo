from typing import Dict, List
import numpy as np
import copy
from collections import defaultdict

from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.data import Measurement, MeasurementSet
import piblin.data.data_collections.experiment_set as experiment_set


class SplitTemperatureSweep(MeasurementSetTransform):
    """
    Splits one or more temperature‐sweep measurements into individual frequency‐sweep Measurements.

    Parameters
    ----------
    n_steps : int or None, optional
        Number of original temperature‐sweep segments.  Pass ``None`` (default)
        to infer the count automatically from the number of measurements in the
        target ``MeasurementSet``.
    temp_step_interval : list of float or None, optional
        Nominal temperature step size (°C) for each segment.  Pass ``None``
        (default) to auto‐detect the step size per segment from the data.
        When provided, the list length must equal ``n_steps``.
    temp_tolerance : float, optional
        Threshold (°C) for detecting a block boundary (default 1.0).

    Examples
    --------
    # Fully automatic — works for any number of segments with any step size:
    SplitTemperatureSweep()

    # Explicit segment count, step sizes still auto‐detected:
    SplitTemperatureSweep(n_steps=3)

    # Fully explicit (original API):
    SplitTemperatureSweep(n_steps=2, temp_step_interval=[5.0, 2.5], temp_tolerance=1.0)
    """

    def __init__(
        self,
        n_steps: int | None = None,
        temp_step_interval: List[float] | None = None,
        temp_tolerance: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        if temp_step_interval is not None and n_steps is not None:
            if len(temp_step_interval) != n_steps:
                raise ValueError(
                    f"Expected {n_steps} step‐interval values, got {len(temp_step_interval)}"
                )
        # None means auto‐detect per sweep from the data
        self.temp_step_interval = temp_step_interval
        self.temp_tolerance = temp_tolerance

    @staticmethod
    def _detect_temp_step(
        dataset: one_d_composite_dataset.OneDimensionalCompositeDataset,
        temp_tolerance: float,
    ) -> float:
        """Infer the nominal temperature step size from a dataset.

        Finds all block-boundary temperature jumps (> ``temp_tolerance``),
        takes their median, and rounds to the nearest 0.5 °C so the result
        is always a clean value (e.g. 2.5 or 5.0).

        Falls back to 5.0 °C when no jumps are detected.
        """
        names = dataset.data_array_names
        idx = {n: i for i, n in enumerate(names)}
        if "temperature" not in idx or "step time" not in idx:
            return 5.0

        temps = copy.deepcopy(dataset.data_arrays[idx["temperature"]])
        times = copy.deepcopy(dataset.data_arrays[idx["step time"]])
        order = np.argsort(times)
        temps = temps[order]

        diffs = np.abs(np.diff(temps))
        jumps = diffs[diffs > temp_tolerance]
        if len(jumps) == 0:
            return 5.0

        raw = float(np.median(jumps))
        # Round to nearest 0.5 °C for a clean nominal step
        return float(round(raw * 2) / 2)

    def split_by_temperature(
        self,
        dataset: one_d_composite_dataset.OneDimensionalCompositeDataset,
        temp_step: float,
        temp_tolerance: float,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Core splitting logic for one composite dataset.
        Groups contiguous data by jumps > temp_tolerance, then rounds each block’s
        median temperature to the nearest multiple of temp_step.
        """
        names = dataset.data_array_names
        idx = {n: i for i, n in enumerate(names)}
        for var in ("temperature", "step time"):
            if var not in idx:
                raise ValueError(f"Required variable '{var}' not found")

        # extract + sort by step time
        all_data = {n: copy.deepcopy(dataset.data_arrays[idx[n]]) for n in names}
        order = np.argsort(all_data["step time"])
        for n in names:
            all_data[n] = all_data[n][order]

        temps = all_data["temperature"]
        sweeps: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: {n: [] for n in names})
        start = 0

        # detect block boundaries
        for i in range(1, len(temps)):
            if abs(temps[i] - temps[i - 1]) > temp_tolerance:
                median_T = float(np.median(temps[start:i]))
                nominal = float(np.round(median_T / temp_step) * temp_step)
                for n in names:
                    sweeps[nominal][n].extend(all_data[n][start:i])
                start = i

        # last block
        median_T = float(np.median(temps[start:]))
        nominal = round(round(median_T / temp_step) * temp_step, 6)
        for n in names:
            sweeps[nominal][n].extend(all_data[n][start:])

        # build list of sorted blocks
        sweep_list: List[Dict[str, np.ndarray]] = []
        for nominal_T, block in sweeps.items():
            freq = np.array(block["angular frequency"])
            o = np.argsort(freq)
            entry: Dict[str, np.ndarray] = {}
            for n in names:
                arr = np.array(block[n])[o]
                # Pa → MPa for moduli/stress
                if n in (
                    "storage modulus",
                    "loss modulus",
                    "complex modulus",
                    "oscillation stress",
                ):
                    if dataset.data_array_units[idx[n]] == "Pa":
                        arr = arr / 1e6
                entry[n] = arr
            entry["nominal_temperature"] = nominal_T
            sweep_list.append(entry)

        return sweep_list

    def _apply(self, target: MeasurementSet, **kwargs) -> experiment_set.ExperimentSet:
        """
        Applies splitting to each of the target.measurements.
        Returns a flat ExperimentSet of all resulting frequency sweeps.
        """
        if not target.measurements:
            raise ValueError("No measurements in target MeasurementSet.")

        # Validate n_steps only when the caller specified it explicitly.
        if self.n_steps is not None and len(target.measurements) != self.n_steps:
            raise ValueError(
                f"Expected {self.n_steps} measurements, got {len(target.measurements)}"
            )

        out_measurements: List[Measurement] = []
        for i, meas in enumerate(target.measurements):
            ds = meas.datasets[0]
            if self.temp_step_interval is not None:
                step = self.temp_step_interval[i]
            else:
                step = self._detect_temp_step(ds, self.temp_tolerance)

            # split this block
            blocks = self.split_by_temperature(ds, step, self.temp_tolerance)

            # wrap each into a Measurement
            for blk in blocks:
                arrays = [blk[n] for n in ds.data_array_names]
                names = ds.data_array_names
                units = copy.deepcopy(ds.data_array_units)

                # update Pa → MPa in units
                for j, n in enumerate(names):
                    if (
                        n
                        in (
                            "storage modulus",
                            "loss modulus",
                            "complex modulus",
                            "oscillation stress",
                        )
                        and units[j] == "Pa"
                    ):
                        units[j] = "MPa"

                T = blk["nominal_temperature"]
                new_ds = one_d_composite_dataset.OneDimensionalCompositeDataset(
                    data_arrays=arrays,
                    data_array_names=names,
                    data_array_units=units,
                    default_independent_name="angular frequency",
                    default_dependent_name="storage modulus",
                    source=f"Frequency sweep at {T}°C",
                )

                out_measurements.append(
                    Measurement(
                        datasets=[new_ds],
                        conditions={"method": f"Frequency sweep ({T} °C)"},
                        details={"nominal_temperature": T},
                    )
                )

        return experiment_set.ExperimentSet(out_measurements, merge_redundant=False)
