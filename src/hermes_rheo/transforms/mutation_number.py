from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
from piblin.data import Measurement, MeasurementSet
import numpy as np
import copy
from collections import defaultdict
from matplotlib import pyplot as plt
import re


class MutationNumberMeasurementSet(MeasurementSet):
    """
    Custom MeasurementSet class for handling mutation number data and providing custom visualization.

    Methods
    -------
    visualize(self, show_all_frequencies=False, y_lim=None, **kwargs):
        Visualizes the mutation number data, either showing all frequencies or
        averaging over time with standard deviation.
    """

    def visualize(self, show_all_frequencies: bool = False, y_lim=None, **kwargs):
        """
        Visualizes the mutation number data.

        Parameters
        ----------
        show_all_frequencies : bool
            If True, plot Mutation number vs. Time for all frequencies.
            If False, only plot the average Mutation vs. Time with standard deviation.
        y_lim : tuple, optional
            A tuple specifying the y-axis limits (y_min, y_max). Default is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot(s).
        ax : matplotlib.axes.Axes or tuple
            The axes object(s) of the plot.
        """
        mutation_number_by_frequency = defaultdict(lambda: defaultdict(list))

        # Extract data from the dataset structure
        for measurement in self.measurements:
            # Accept either 'Angular Frequency' (real freq) or 'Frequency Index' (pseudo)
            if 'Angular Frequency' in measurement.conditions:
                freq_key = 'Angular Frequency'
            elif 'Frequency Index' in measurement.conditions:
                freq_key = 'Frequency Index'
            else:
                # skip if no usable frequency-like condition
                continue

            freq = measurement.conditions[freq_key]
            for data in measurement.datasets:
                time_values = data.x_values
                mutation_number_values = data.y_values
                for time, mutation_number in zip(time_values, mutation_number_values):
                    mutation_number_by_frequency[freq][time].append(mutation_number)

        if not mutation_number_by_frequency:
            raise ValueError("No mutation-number data found to visualize.")

        # Calculate average and standard deviation over all frequencies at each time
        all_times = {t for times in mutation_number_by_frequency.values() for t in times}
        average_times = sorted(all_times)

        average_mutation_numbers = []
        mutation_number_stds = []
        for t in average_times:
            all_values_at_time = []
            for freq in mutation_number_by_frequency:
                if t in mutation_number_by_frequency[freq]:
                    values = mutation_number_by_frequency[freq][t]
                    all_values_at_time.extend(values)
            if all_values_at_time:
                average_mutation_numbers.append(np.mean(all_values_at_time))
                mutation_number_stds.append(np.std(all_values_at_time))

        average_times = np.array(average_times)
        average_mutation_numbers = np.array(average_mutation_numbers)
        mutation_number_stds = np.array(mutation_number_stds)

        # Colors for different time steps in the scatter / errorbar plots
        unique_times = sorted(set(average_times))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_times)))

        if show_all_frequencies:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot Mu vs. Time for all frequencies on the left plot
            for color, t in zip(colors, unique_times):
                for freq in mutation_number_by_frequency:
                    if t in mutation_number_by_frequency[freq]:
                        mutation_numbers = mutation_number_by_frequency[freq][t]
                        ax1.scatter(
                            [t] * len(mutation_numbers),
                            mutation_numbers,
                            color=color,
                            s=40
                        )

            ax1.set_xlabel('Time (s)', fontsize=20)
            ax1.set_ylabel('Mutation number', fontsize=20)
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)
            if y_lim:
                ax1.set_ylim(y_lim)
        else:
            fig, ax2 = plt.subplots(figsize=(10, 6))

        # Plot Average Mu vs. Time with error bars on the right (or single) plot
        for i, (t, mean, std) in enumerate(
            zip(average_times, average_mutation_numbers, mutation_number_stds)
        ):
            color = colors[i % len(colors)]
            ax2.errorbar(
                t,
                mean,
                yerr=std,
                fmt='o',
                color=color,
                ecolor='black',
                markersize=8,
                elinewidth=3,
                capsize=4
            )

        ax2.set_xlabel('Time (s)', fontsize=20)
        ax2.set_ylabel('Mutation number (average)', fontsize=20)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        if y_lim:
            ax2.set_ylim(y_lim)

        plt.tight_layout()

        if show_all_frequencies:
            return fig, (ax1, ax2)
        else:
            return fig, ax2


class MutationNumber(MeasurementSetTransform):
    """
    A transform class to calculate and visualize the mutation number from measurement sets.

    Parameters
    ----------
    state : str
        The state variable to be used (default is 'time').
    state_sampling : str
        The method to determine the state value ('average', 'first point', 'last point').
    dependent_variable : str
        The dependent variable to be used for mutation number calculation (default 'complex modulus').
    measurements : list[int] or None
        Optional list of measurement indices to use (e.g. list(range(1, 126, 2))).
    owch_length : float
        Duration of each OWCh / correlated window, used in the mutation formula.
    """

    def __init__(
        self,
        state: str = 'time',
        state_sampling: str = 'first point',
        dependent_variable: str = 'complex modulus',
        measurements=None,
        owch_length: float = 10.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.state = state
        self.state_sampling = state_sampling
        self.dependent_variable = dependent_variable
        self.measurements = measurements  # list of indices to use, or None for all
        self.owch_length = float(owch_length)

    # ----------------- helpers -----------------

    @staticmethod
    def _get_step_name(measurement) -> str:
        """
        Robustly extract the TRIOS 'Step name' for this measurement.

        Checks details first (most likely), then conditions with a few common keys.
        """
        # 1) details['Step name'] – this is what TRIOS usually produces
        if 'Step name' in getattr(measurement, 'details', {}):
            return str(measurement.details['Step name'])

        # 2) fallback: other possible keys in details
        for key in ('StepName', 'step name', 'Step', 'Name'):
            if key in getattr(measurement, 'details', {}):
                return str(measurement.details[key])

        # 3) fallback: check conditions
        for key in ('Step name', 'StepName', 'step name', 'Step', 'Name', 'method'):
            if key in getattr(measurement, 'conditions', {}):
                return str(measurement.conditions[key])

        return ""

    @staticmethod
    def _parse_temperature_from_step_name(step_name: str):
        """
        Try to parse a temperature (°C) from a string like:
            'Correlated -50.06 °C - 2'
        Returns float or None.
        """
        m = re.search(r'(-?\d+\.?\d*)\s*°C', step_name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    # ----------------- state to condition (Arbitrary Wave helper) -----------------

    def _state_to_condition(self, target: MeasurementSet):
        """
        Applies the transformation to compute the specified state's value and adds it as a condition.
        Used when a physical property is collected transiently and needs to be converted to a condition.
        """
        for measurement in target.measurements:
            state_found = False
            for dataset in measurement.datasets:
                if self.state in dataset._data_array_names:
                    # Preserve the dataset's currently-active independent
                    # variable — previous code hardcoded ``'temperature'``
                    # here, which raised on any dataset that did not have
                    # a temperature column.
                    current_independent = dataset.independent_variable_name
                    dataset.switch_coordinates(
                        independent_name=current_independent,
                        dependent_name=self.state,
                    )
                    if self.state_sampling == 'average':
                        value = np.average(dataset.y_values)
                    elif self.state_sampling == 'first point':
                        value = dataset.y_values[0]
                    elif self.state_sampling == 'last point':
                        value = dataset.y_values[-1]
                    else:
                        raise ValueError(
                            "Invalid method. Choose 'average', 'first point', or 'last point'"
                        )
                    measurement.add_condition(f'{self.state}', value)
                    state_found = True
                    break
            if not state_found:
                raise ValueError(
                    f"State name {self.state} not found in datasets for one of the measurements"
                )

    # ----------------- main apply -----------------

    def _apply(self, target: MeasurementSet, **kwargs) -> MutationNumberMeasurementSet:
        """
        Applies the mutation number calculation to the target dataset and returns a MutationNumberMeasurementSet.
        """
        # Branch selection: the aggregate ``target.conditions['method']``
        # is empty on multi-step files (different measurements have
        # different step names, so the intersection is empty), so we
        # instead scan per-measurement step names / method conditions and
        # pick whichever category dominates.
        step_names_lower: list[str] = []
        for m in getattr(target, "measurements", []) or []:
            step_names_lower.append(self._get_step_name(m).lower())
            # Also consider the ``method`` condition set by TriosRheoReader
            # (e.g. ``"Step name\\tArbitrary Wave - 1"``).
            method_cond = str(getattr(m, "conditions", {}).get("method", "")).lower()
            if method_cond:
                step_names_lower.append(method_cond)

        has_arbitrary_wave = any("arbitrary wave" in s for s in step_names_lower)
        has_correlated = any("correlated" in s for s in step_names_lower)

        # Preserve backwards compatibility: an explicit top-level method
        # condition (older calling code / tests) still wins.
        top_level_method = str(target.conditions.get("method", ""))
        if "Arbitrary Wave" in top_level_method:
            has_arbitrary_wave = True
        if "Correlated" in top_level_method:
            has_correlated = True

        # ------------------------------------------------------------------
        # Branch 1: Arbitrary Wave logic (kept intact)
        # ------------------------------------------------------------------
        if has_arbitrary_wave:
            self._state_to_condition(target)
            moduli = []
            frequencies = []
            wave_start_times = []
            waiting_times = []
            wave_durations = []

            measurements_to_use = target.measurements
            if self.measurements is not None:
                measurements_to_use = [target.measurements[i] for i in self.measurements]

            # Extract data and prepare for mutation number calculation
            for measurement in measurements_to_use:
                dataset_frequency = measurement.datasets[0]
                dataset_frequency.switch_coordinates(
                    independent_name='angular frequency',
                    dependent_name=self.dependent_variable
                )
                frequency = copy.deepcopy(dataset_frequency.x_values)
                modulus = copy.deepcopy(dataset_frequency.y_values)

                wave_start_time = measurement.conditions['time']
                waiting_time = measurement.details['waiting_time']

                dataset_time = measurement.datasets[1]
                dataset_time.switch_coordinates('step time', 'time')
                step_time = copy.deepcopy(dataset_time.x_values)

                wave_duration = float(np.max(step_time))

                moduli.append(modulus)
                frequencies.append(frequency)
                wave_start_times.append(float(wave_start_time))
                waiting_times.append(float(waiting_time))
                wave_durations.append(wave_duration)

            moduli_by_frequency = defaultdict(lambda: defaultdict(list))
            for i, freq_list in enumerate(frequencies):
                for j, freq in enumerate(freq_list):
                    moduli_by_frequency[freq][wave_start_times[i]].append(moduli[i][j])

            mutation_number_by_frequency = defaultdict(lambda: defaultdict(list))
            T_values = [wave_duration - waiting_time
                        for wave_duration, waiting_time in zip(wave_durations, waiting_times)]
            measurements_out = []
            for i, freq_list in enumerate(frequencies):
                if i > 0:
                    time_diff = (wave_start_times[i] + waiting_times[i]
                                 - (wave_start_times[i - 1] + waiting_times[i - 1]))
                    T = T_values[i]
                    for j, freq in enumerate(freq_list):
                        if wave_start_times[i] != wave_start_times[i - 1]:
                            ln_modulus_curr = np.log(moduli[i][j])
                            ln_modulus_prev = np.log(moduli[i - 1][j])
                            derivative_ln_modulus = (ln_modulus_curr - ln_modulus_prev) / time_diff
                            if derivative_ln_modulus != 0:
                                # your original formula T / (1 / derivative) simplifies to T * derivative
                                mutation_number = T * derivative_ln_modulus
                                mutation_number_by_frequency[freq][wave_start_times[i]].append(
                                    mutation_number
                                )

            # Create Measurement objects with mutation number data
            for freq, time_dict in mutation_number_by_frequency.items():
                all_times = []
                all_mutation_numbers = []
                for t, mutation_numbers in time_dict.items():
                    all_times.extend([t] * len(mutation_numbers))
                    all_mutation_numbers.extend(mutation_numbers)

                mutation_number_dataset = one_dimensional_dataset.OneDimensionalDataset(
                    dependent_variable_data=np.array(all_mutation_numbers),
                    dependent_variable_names=['mutation number'],
                    dependent_variable_units=[' '],
                    independent_variable_data=[np.array(all_times)],
                    independent_variable_names=['time'],
                    independent_variable_units=['s'],
                    source='datasets in time and frequency domain'
                )

                measurements_out.append(
                    Measurement(
                        datasets=[mutation_number_dataset],
                        conditions={'Angular Frequency': freq},
                        details={}
                    )
                )

            return MutationNumberMeasurementSet(measurements=measurements_out)

        # ------------------------------------------------------------------
        # Branch 2: "Correlated" OWCh-like steps (your new experiment)
        # ------------------------------------------------------------------
        # For now, we focus ONLY on the correlated signal, using the logic
        # you provided (OWCh_length, dt/dlnG*, etc.). Frequency Chirp steps
        # are ignored.
        # ------------------------------------------------------------------

        # Decide which measurements to use (indices)
        if self.measurements is not None:
            indices = list(self.measurements)
        else:
            indices = list(range(len(target.measurements)))

        # Collect correlated measurements in index order
        correlated_measurements = []
        for idx in indices:
            m = target.measurements[idx]
            step_name = self._get_step_name(m)
            if 'Correlated' in step_name:
                correlated_measurements.append((idx, m, step_name))

        if not correlated_measurements:
            raise ValueError(
                "No 'Correlated' measurements found in the selected measurement indices. "
                "Check that 'Step name' is present in measurement.details/conditions and "
                "that it contains 'Correlated'."
            )

        OWCh_length = self.owch_length
        OWCh_time = []
        OWCh_complex_modulus = []
        OWCh_temperature = []

        # ---- Extract time, temperature and complex modulus arrays per correlated step ----
        for idx, measurement, step_name in correlated_measurements:
            dataset = measurement.datasets[0]

            # First: time vs complex modulus
            dataset.switch_coordinates(independent_name='time',
                                       dependent_name=self.dependent_variable)
            # Take the starting time of this window (as in your script)
            time_values = copy.deepcopy(dataset.x_values)
            if len(time_values) == 0:
                raise ValueError(f"No time values found in correlated measurement index {idx}.")
            OWCh_time.append(float(time_values[0]))

            # G*_t(omega): array of complex modulus values at this OWCh
            G_values = copy.deepcopy(dataset.y_values)
            G_values = np.asarray(G_values, dtype=float)
            OWCh_complex_modulus.append(G_values)

            # Second: temperature vs complex modulus
            dataset.switch_coordinates(independent_name='temperature',
                                       dependent_name=self.dependent_variable)
            temp_values = copy.deepcopy(dataset.x_values)
            temp_values = np.asarray(temp_values, dtype=float)
            if temp_values.size == 0:
                # Fallback: parse from step name if needed
                t_parsed = self._parse_temperature_from_step_name(step_name)
                if t_parsed is None:
                    raise ValueError(
                        f"No temperature values found and cannot parse temperature "
                        f"from step name: {step_name}"
                    )
                OWCh_temperature.append(float(t_parsed))
            else:
                # average in case multiple temperatures are present
                OWCh_temperature.append(float(np.average(temp_values)))

        OWCh_time = np.array(OWCh_time, dtype=float)          # shape: (n_steps,)
        OWCh_temperature = np.array(OWCh_temperature, dtype=float)
        OWCh_complex_modulus = np.array(OWCh_complex_modulus)  # shape: (n_steps, n_freq)

        n_steps, n_freq = OWCh_complex_modulus.shape

        if n_steps < 2:
            raise ValueError("Need at least two correlated OWCh steps to compute mutation.")

        # ---- Your mutation number logic (vectorized) ----

        # Time difference between consecutive OWCh
        dt = np.diff(OWCh_time)           # shape: (n_steps - 1,)
        # Temperature difference between consecutive OWCh
        dT = np.diff(OWCh_temperature)    # shape: (n_steps - 1,)

        # ln G*_t(omega) for each OWCh
        lnG = np.log(OWCh_complex_modulus)  # shape: (n_steps, n_freq)

        # differentiation of lnG* between consecutive OWCh
        # dlnG_star[i] = lnG[i+1] - lnG[i]
        dlnG_star = lnG[1:, :] - lnG[:-1, :]      # shape: (n_steps - 1, n_freq)

        # Avoid division by zero: mask where dlnG_star == 0
        zero_mask = (dlnG_star == 0)

        # dt / dlnG* (broadcast dt over frequency dimension)
        dt_expanded = dt[:, None]                  # shape: (n_steps - 1, 1)
        dT_expanded = dT[:, None]                  # shape: (n_steps - 1, 1)

        dt_dlnG = np.empty_like(dlnG_star)
        dT_dlnG = np.empty_like(dlnG_star)

        with np.errstate(divide='ignore', invalid='ignore'):
            dt_dlnG = dt_expanded / dlnG_star
            dT_dlnG = dT_expanded / dlnG_star

        # Assign NaN where we had zero in denominator
        dt_dlnG[zero_mask] = np.nan
        dT_dlnG[zero_mask] = np.nan

        # Mu vs time, Mu vs temperature
        Mu_vs_time = OWCh_length / dt_dlnG           # shape: (n_steps - 1, n_freq)
        Mu_vs_temperature = OWCh_length / dT_dlnG    # same shape; not used downstream yet

        # Time and temperature arrays corresponding to these mutation values:
        # They start at OWCh 2 (index 1), as in your script
        time_Mu = OWCh_time[1:]          # shape: (n_steps - 1,)
        temperature_Mu = OWCh_temperature[1:]   # shape: (n_steps - 1,)

        # ---- Build MeasurementSet: one Measurement per frequency index ----
        measurements_out = []

        for j in range(n_freq):
            mu_series = Mu_vs_time[:, j]          # (n_steps - 1,) for this frequency index
            t_series = time_Mu.copy()

            # Filter out NaNs if any
            valid = ~np.isnan(mu_series)
            if not np.any(valid):
                continue

            mu_series = mu_series[valid]
            t_series = t_series[valid]

            mutation_number_dataset = one_dimensional_dataset.OneDimensionalDataset(
                dependent_variable_data=np.array(mu_series),
                dependent_variable_names=['mutation number'],
                dependent_variable_units=['a.u.'],
                independent_variable_data=[np.array(t_series)],
                independent_variable_names=['time'],
                independent_variable_units=['s'],
                source='correlated OWCh steps'
            )

            # Use a pseudo-frequency condition so visualize can group/average
            measurements_out.append(
                Measurement(
                    datasets=[mutation_number_dataset],
                    conditions={'Frequency Index': j},
                    details={}
                )
            )

        if not measurements_out:
            raise ValueError("No valid mutation-number data constructed for correlated steps.")

        return MutationNumberMeasurementSet(measurements=measurements_out)
