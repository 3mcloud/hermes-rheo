import copy
from collections import defaultdict

import numpy as np
import piblin.data.data_collections.experiment_set as experiment_set
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.data import Measurement, MeasurementSet
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform

_PA_ARRAYS = ("storage modulus", "loss modulus", "complex modulus", "oscillation stress")


class SplitTemperatureSweep(MeasurementSetTransform):
    """Split temperature-sweep measurements into one frequency sweep per temperature.

    Each input measurement holds consecutive frequency sweeps at stepped
    temperatures. Block boundaries are temperature jumps larger than
    ``temp_tolerance``. Moduli and stress given in Pa are converted to MPa.
    Each output ``Measurement`` carries ``details["nominal_temperature"]``.

    Parameters
    ----------
    n_steps : int or None, optional
        Number of temperature-sweep measurements expected. If None (default),
        any number is accepted.
    temp_step_interval : list of float or None, optional
        Nominal temperature step (degC) for each measurement. If None
        (default), the step is detected from the data. Length must equal
        ``n_steps`` when both are given.
    temp_tolerance : float, optional
        Temperature jump (degC) that marks a block boundary. Default 1.0.

    Examples
    --------
    >>> SplitTemperatureSweep()  # doctest: +SKIP
    >>> SplitTemperatureSweep(n_steps=2, temp_step_interval=[5.0, 2.5])  # doctest: +SKIP
    """

    def __init__(
        self,
        n_steps: int | None = None,
        temp_step_interval: list[float] | None = None,
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
        self.temp_step_interval = temp_step_interval
        self.temp_tolerance = temp_tolerance

    @staticmethod
    def _detect_temp_step(
        dataset: one_d_composite_dataset.OneDimensionalCompositeDataset,
        temp_tolerance: float,
    ) -> float:
        """Estimate the nominal temperature step of a dataset.

        Returns the median of temperature jumps larger than ``temp_tolerance``,
        rounded to the nearest 0.5 degC. Returns 5.0 if no jumps are found
        or the required arrays are missing.
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
        return float(round(raw * 2) / 2)

    def split_by_temperature(
        self,
        dataset: one_d_composite_dataset.OneDimensionalCompositeDataset,
        temp_step: float,
        temp_tolerance: float,
    ) -> list[dict[str, np.ndarray]]:
        """Split one composite dataset into frequency-sweep blocks.

        Data are sorted by step time and grouped at temperature jumps larger
        than ``temp_tolerance``. Each block is labeled with its median
        temperature rounded to a multiple of ``temp_step`` and sorted by
        angular frequency.

        Returns
        -------
        list of dict
            One dict per nominal temperature, mapping array names to arrays,
            plus the key ``"nominal_temperature"``.
        """
        names = dataset.data_array_names
        idx = {n: i for i, n in enumerate(names)}
        for var in ("temperature", "step time"):
            if var not in idx:
                raise ValueError(f"Required variable '{var}' not found")

        all_data = {n: copy.deepcopy(dataset.data_arrays[idx[n]]) for n in names}
        order = np.argsort(all_data["step time"])
        for n in names:
            all_data[n] = all_data[n][order]

        temps = all_data["temperature"]
        sweeps: dict[float, dict[str, list[float]]] = defaultdict(lambda: {n: [] for n in names})
        start = 0

        for i in range(1, len(temps)):
            if abs(temps[i] - temps[i - 1]) > temp_tolerance:
                median_T = float(np.median(temps[start:i]))
                nominal = round(round(median_T / temp_step) * temp_step, 6)
                for n in names:
                    sweeps[nominal][n].extend(all_data[n][start:i])
                start = i

        median_T = float(np.median(temps[start:]))
        nominal = round(round(median_T / temp_step) * temp_step, 6)
        for n in names:
            sweeps[nominal][n].extend(all_data[n][start:])

        sweep_list: list[dict[str, np.ndarray]] = []
        for nominal_T, block in sweeps.items():
            freq = np.array(block["angular frequency"])
            o = np.argsort(freq)
            entry: dict[str, np.ndarray] = {}
            for n in names:
                arr = np.array(block[n])[o]
                if n in _PA_ARRAYS and dataset.data_array_units[idx[n]] == "Pa":
                    arr = arr / 1e6
                entry[n] = arr
            entry["nominal_temperature"] = nominal_T
            sweep_list.append(entry)

        return sweep_list

    def _apply(self, target: MeasurementSet, **kwargs) -> experiment_set.ExperimentSet:
        """Split every measurement; return a flat ``ExperimentSet`` of frequency sweeps."""
        if not target.measurements:
            raise ValueError("No measurements in target MeasurementSet.")

        if self.n_steps is not None and len(target.measurements) != self.n_steps:
            raise ValueError(
                f"Expected {self.n_steps} measurements, got {len(target.measurements)}"
            )

        out_measurements: list[Measurement] = []
        for i, meas in enumerate(target.measurements):
            ds = meas.datasets[0]
            if self.temp_step_interval is not None:
                step = self.temp_step_interval[i]
            else:
                step = self._detect_temp_step(ds, self.temp_tolerance)

            blocks = self.split_by_temperature(ds, step, self.temp_tolerance)

            for blk in blocks:
                arrays = [blk[n] for n in ds.data_array_names]
                names = ds.data_array_names
                units = copy.deepcopy(ds.data_array_units)

                for j, n in enumerate(names):
                    if n in _PA_ARRAYS and units[j] == "Pa":
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
