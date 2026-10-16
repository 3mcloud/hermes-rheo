"""Mutation number of a time-evolving material from successive oscillatory spectra.

The mutation number, ``Mu = T / lambda_Mu``, compares the duration ``T`` of an
observation (a single oscillatory wave or an optimally windowed chirp) with
the mutation time ``lambda_Mu = (d ln g / dt)^-1`` of a viscoelastic quantity
``g`` (usually the complex modulus ``|G*|``). When ``|Mu|`` is small the
sample is quasi-static during the measurement and Fourier-based spectra are
valid; the usual criterion is ``|Mu| <= ~0.15``.

The transform returns one measurement per angular frequency. Each holds a
piblin composite dataset with the columns "time", "mutation number" and
"angular frequency" (and "temperature" for correlated OWCh steps), so the
result can be plotted with :meth:`MutationNumberMeasurementSet.visualize` or
mapped over temperature and frequency with
:class:`~hermes_rheo.transforms.kymograph.Kymograph`.

References
----------
.. [1] Mours, M. and Winter, H. H. (1994). Time-resolved rheometry.
   Rheologica Acta 33(5), 385-397. (Winter's definition of the mutation number.)
.. [2] Jamali, S. and McKinley, G. H. (2022). The Mnemosyne number and the
   rheology of remembrance. Journal of Rheology 66, 1027-1039.
   doi:10.1122/8.0000432.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from piblin.data import Measurement, MeasurementSet, OneDimensionalCompositeDataset
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform

MUTATION_NUMBER = "mutation number"


class MutationNumberMeasurementSet(MeasurementSet):
    """MeasurementSet of mutation-number time series, one measurement per frequency."""

    def visualize(self, show_all_frequencies: bool = False, y_lim=None, **kwargs):
        """Plot the mutation number against time.

        Thin wrapper around
        :func:`hermes_rheo.visualization.mutation.plot_mutation_number`;
        see that function for the keyword arguments (``color_by``,
        ``cmap``, ``threshold``, ...).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes or tuple of matplotlib.axes.Axes
            ``(ax_all, ax_mean)`` if `show_all_frequencies`, else ``ax_mean``.
        """
        from hermes_rheo.visualization.mutation import plot_mutation_number

        return plot_mutation_number(
            self, show_all_frequencies=show_all_frequencies, ylim=y_lim, **kwargs
        )


def _column(dataset: Any, name: str) -> tuple[np.ndarray, str] | None:
    """Return ``(values, unit)`` of the data array *name*, or None if absent."""
    names = list(dataset.data_array_names)
    if name not in names:
        return None
    i = names.index(name)
    return np.asarray(dataset.data_arrays[i], dtype=float), str(dataset.data_array_units[i] or "")


class MutationNumber(MeasurementSetTransform):
    """
    Compute the mutation number of a time-evolving material from a measurement set.

    The mutation number is the ratio of the observation time to the
    mutation time of the modulus, ``Mu = T * d ln|G*| / dt``, evaluated at
    each frequency from successive spectra. Two input layouts are supported:
    "Arbitrary Wave" steps and "Correlated" optimal-waveform (OWCh) steps,
    e.g. a TORCh temperature ramp.

    Parameters
    ----------
    state : str, default "time"
        Name of the state variable stored as a condition (Arbitrary Wave only).
    state_sampling : {"average", "first point", "last point"}, default "first point"
        How `state` is reduced to one value per measurement.
    dependent_variable : str, default "complex modulus"
        Dataset variable whose logarithmic derivative is evaluated.
    measurements : list of int, optional
        Indices of the measurements to use. All are used if None.
    owch_length : float, default 10.0
        Observation time ``T`` of each correlated window, in seconds
        (Correlated steps only).
    absolute : bool, default False
        Return ``|Mu|`` instead of the signed mutation number. The sign
        only tells whether ``g`` grows (e.g. curing) or decays (e.g.
        heating); the quasi-static criterion applies to ``|Mu|``.
    dt_threshold : float, optional
        Skip pairs of consecutive spectra more than `dt_threshold` seconds
        apart (e.g. across an isothermal dwell between two ramps).

    Returns
    -------
    MutationNumberMeasurementSet
        One measurement per angular frequency (condition "Angular
        Frequency", rad/s). Its composite dataset holds "time" (s),
        "mutation number" (dimensionless) and "angular frequency" (rad/s),
        plus "temperature" for correlated steps; ``details`` holds
        ``mutation_number_mean``, ``mutation_number_max``, ``n_points`` and
        the parameters above.

    Notes
    -----
    The derivative is a finite difference between consecutive spectra,
    ``Mu_i = T ln(g_i / g_(i-1)) / (t_i - t_(i-1))``, reported at time
    ``t_i`` (and, for correlated steps, the mean temperature of window
    ``i``). The logarithmic derivative makes ``Mu`` independent of the
    modulus scale.

    References
    ----------
    .. [1] Mours, M. and Winter, H. H. (1994). Time-resolved rheometry.
       Rheologica Acta 33(5), 385-397.
    .. [2] Jamali, S. and McKinley, G. H. (2022). The Mnemosyne number and the
       rheology of remembrance. Journal of Rheology 66, 1027-1039.
    """

    def __init__(
        self,
        state: str = "time",
        state_sampling: str = "first point",
        dependent_variable: str = "complex modulus",
        measurements=None,
        owch_length: float = 10.0,
        absolute: bool = False,
        dt_threshold: float | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state = state
        self.state_sampling = state_sampling
        self.dependent_variable = dependent_variable
        self.measurements = measurements
        self.owch_length = float(owch_length)
        self.absolute = bool(absolute)
        self.dt_threshold = None if dt_threshold is None else float(dt_threshold)

    @staticmethod
    def _get_step_name(measurement) -> str:
        """Return the TRIOS step name of a measurement, or "" if absent."""
        if "Step name" in getattr(measurement, "details", {}):
            return str(measurement.details["Step name"])

        for key in ("StepName", "step name", "Step", "Name"):
            if key in getattr(measurement, "details", {}):
                return str(measurement.details[key])

        for key in ("Step name", "StepName", "step name", "Step", "Name", "method"):
            if key in getattr(measurement, "conditions", {}):
                return str(measurement.conditions[key])

        return ""

    @staticmethod
    def _parse_temperature_from_step_name(step_name: str):
        """Parse a temperature in °C from e.g. ``'Correlated -50.06 °C - 2'``; None if absent."""
        m = re.search(r"(-?\d+\.?\d*)\s*°C", step_name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def _state_to_condition(self, target: MeasurementSet):
        """Reduce the transient `state` dataset of each measurement to a scalar condition."""
        for measurement in target.measurements:
            state_found = False
            for dataset in measurement.datasets:
                if self.state in dataset._data_array_names:
                    current_independent = dataset.independent_variable_name
                    dataset.switch_coordinates(
                        independent_name=current_independent,
                        dependent_name=self.state,
                    )
                    if self.state_sampling == "average":
                        value = np.average(dataset.y_values)
                    elif self.state_sampling == "first point":
                        value = dataset.y_values[0]
                    elif self.state_sampling == "last point":
                        value = dataset.y_values[-1]
                    else:
                        raise ValueError(
                            "Invalid method. Choose 'average', 'first point', or 'last point'"
                        )
                    measurement.add_condition(f"{self.state}", value)
                    state_found = True
                    break
            if not state_found:
                raise ValueError(
                    f"State name {self.state} not found in datasets for one of the measurements"
                )

    def _selected(self, target: MeasurementSet) -> list:
        """Measurements chosen by `measurements`, or all of them."""
        if self.measurements is None:
            return list(target.measurements)
        return [target.measurements[i] for i in self.measurements]

    def _mutation_number(self, ln_ratio, dt, observation_time):
        """``T * dln(g) / dt`` (or its magnitude); NaN where ``g`` is unchanged or `dt` is too long."""
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = observation_time * np.asarray(ln_ratio, float) / dt
        mu = np.where(np.asarray(ln_ratio) == 0, np.nan, mu)
        if self.dt_threshold is not None:
            mu = np.where(np.asarray(dt) > self.dt_threshold, np.nan, mu)
        return np.abs(mu) if self.absolute else mu

    def _make_details(self, values, layout: str) -> dict:
        """Parameters and summary statistics stored with each output measurement."""
        values = np.asarray(values, dtype=float)
        return {
            "mutation_number_mean": float(np.mean(values)),
            "mutation_number_max": float(np.max(values)),
            "n_points": int(values.size),
            "layout": layout,
            "state": self.state,
            "state_sampling": self.state_sampling,
            "dependent_variable": self.dependent_variable,
            "owch_length": self.owch_length,
            "absolute": self.absolute,
            "dt_threshold": self.dt_threshold,
        }

    def _series_measurement(
        self,
        omega: float,
        columns: dict[str, tuple[np.ndarray, str]],
        layout: str,
    ) -> Measurement | None:
        """One output measurement from ``{name: (values, unit)}``; None if all Mu are NaN."""
        valid = np.isfinite(columns[MUTATION_NUMBER][0])
        if not valid.any():
            return None
        dataset = OneDimensionalCompositeDataset(
            data_arrays=[values[valid] for values, _ in columns.values()],
            data_array_names=list(columns),
            data_array_units=[unit for _, unit in columns.values()],
            default_independent_name="time",
            default_dependent_name=MUTATION_NUMBER,
            source=f"mutation number ({layout})",
        )
        return Measurement(
            datasets=[dataset],
            conditions={"Angular Frequency": float(omega)},
            details=self._make_details(columns[MUTATION_NUMBER][0][valid], layout),
        )

    def _apply(self, target: MeasurementSet, **kwargs) -> MutationNumberMeasurementSet:
        """Compute mutation numbers for the Arbitrary Wave or Correlated layout."""
        # The aggregate target.conditions["method"] is empty on multi-step
        # files, so scan per-measurement step names and method conditions.
        step_names_lower: list[str] = []
        for m in getattr(target, "measurements", []) or []:
            step_names_lower.append(self._get_step_name(m).lower())
            method_cond = str(getattr(m, "conditions", {}).get("method", "")).lower()
            if method_cond:
                step_names_lower.append(method_cond)

        has_arbitrary_wave = any("arbitrary wave" in s for s in step_names_lower)
        if "Arbitrary Wave" in str(target.conditions.get("method", "")):
            has_arbitrary_wave = True

        measurements_out = (
            self._arbitrary_wave(target) if has_arbitrary_wave else self._correlated(target)
        )
        if not measurements_out:
            raise ValueError("No valid mutation-number data could be computed.")
        return MutationNumberMeasurementSet(measurements=measurements_out)

    def _arbitrary_wave(self, target: MeasurementSet) -> list[Measurement]:
        """Mutation number between consecutive single-wave spectra."""
        self._state_to_condition(target)
        spectra = []
        for measurement in self._selected(target):
            spectrum = measurement.datasets[0]
            spectrum.switch_coordinates("angular frequency", self.dependent_variable)
            waveform = measurement.datasets[1]
            waveform.switch_coordinates("step time", "time")
            waiting_time = float(measurement.details["waiting_time"])
            spectra.append(
                {
                    "start": float(measurement.conditions["time"]),
                    "waiting": waiting_time,
                    "T": float(np.max(waveform.x_values)) - waiting_time,
                    "omega": np.asarray(spectrum.x_values, float),
                    "g": np.asarray(spectrum.y_values, float),
                }
            )
        omega_unit = _column(target.measurements[0].datasets[0], "angular frequency")[1]

        series: dict[float, dict[str, list]] = defaultdict(lambda: {"time": [], "mu": []})
        for previous, current in zip(spectra, spectra[1:]):
            if current["start"] == previous["start"]:
                continue
            dt = (current["start"] + current["waiting"]) - (previous["start"] + previous["waiting"])
            mu = self._mutation_number(np.log(current["g"] / previous["g"]), dt, current["T"])
            for omega, value in zip(current["omega"], mu):
                series[float(omega)]["time"].append(current["start"])
                series[float(omega)]["mu"].append(value)

        measurements_out = []
        for omega, values in series.items():
            time = np.asarray(values["time"], float)
            measurement = self._series_measurement(
                omega,
                {
                    "time": (time, "s"),
                    MUTATION_NUMBER: (np.asarray(values["mu"], float), ""),
                    "angular frequency": (np.full_like(time, omega), omega_unit),
                },
                "arbitrary wave",
            )
            if measurement is not None:
                measurements_out.append(measurement)
        return measurements_out

    def _correlated(self, target: MeasurementSet) -> list[Measurement]:
        """Mutation number between consecutive correlated OWCh windows (e.g. TORCh)."""
        windows = [m for m in self._selected(target) if "Correlated" in self._get_step_name(m)]
        if len(windows) < 2:
            raise ValueError(
                "Need at least two 'Correlated' measurements (identified by their "
                "'Step name') to compute the mutation number."
            )

        start_times, moduli, temperatures = [], [], []
        temperature_unit = "°C"
        for measurement in windows:
            dataset = measurement.datasets[0]
            time, time_unit = _column(dataset, "time")
            start_times.append(time[0])
            moduli.append(_column(dataset, self.dependent_variable)[0])
            temperature = _column(dataset, "temperature")
            if temperature is not None and temperature[0].size:
                temperatures.append(float(np.mean(temperature[0])))
                temperature_unit = temperature[1]
            else:
                parsed = self._parse_temperature_from_step_name(self._get_step_name(measurement))
                temperatures.append(np.nan if parsed is None else parsed)
        omega, omega_unit = _column(windows[0].datasets[0], "angular frequency")

        start_times = np.asarray(start_times)
        moduli = np.asarray(moduli)
        mu = self._mutation_number(
            np.diff(np.log(moduli), axis=0),
            np.diff(start_times)[:, None],
            self.owch_length,
        )
        time = start_times[1:]
        temperature = np.asarray(temperatures)[1:]

        measurements_out = []
        for j, omega_j in enumerate(omega):
            measurement = self._series_measurement(
                omega_j,
                {
                    "time": (time, time_unit),
                    MUTATION_NUMBER: (mu[:, j], ""),
                    "temperature": (temperature, temperature_unit),
                    "angular frequency": (np.full_like(time, omega_j), omega_unit),
                },
                "correlated",
            )
            if measurement is not None:
                measurements_out.append(measurement)
        return measurements_out
