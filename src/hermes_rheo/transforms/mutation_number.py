from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
from piblin.data import Measurement, MeasurementSet
import numpy as np
import copy
from collections import defaultdict
from matplotlib import pyplot as plt


class MutationNumberMeasurementSet(MeasurementSet):
    """
    Custom MeasurementSet class for handling mutation number data and providing custom visualization.

    Methods:
    --------
    visualize(self, show_all_frequencies=False, y_lim=None, **kwargs):
        Visualizes the mutation number data, either showing all frequencies or averaging over time with standard
        deviation.
    """

    def visualize(self, show_all_frequencies=False, y_lim=None, **kwargs):
        """
        Visualizes the mutation number data.

        Parameters:
        -----------
        show_all_frequencies : bool
            If True, plot Mutation number vs. Time for all frequencies. If False, only plot the average Mutation
            vs. Time with standard deviation.
        y_lim : tuple, optional
            A tuple specifying the y-axis limits (y_min, y_max). Default is None, which automatically sets the limits.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot(s).
        ax : matplotlib.axes._subplots.AxesSubplot or tuple
            The axes object(s) of the plot.
        """
        mutation_number_by_frequency = defaultdict(lambda: defaultdict(list))

        # Extract data from the dataset structure
        for measurement in self.measurements:
            freq = measurement.conditions['Angular Frequency']
            for data in measurement.datasets:
                time_values = data.x_values
                mutation_number_values = data.y_values
                for time, mutation_number in zip(time_values, mutation_number_values):
                    mutation_number_by_frequency[freq][time].append(mutation_number)

        # Calculate average and standard deviation
        average_times = sorted({time for times in mutation_number_by_frequency.values() for time in times})
        average_mutation_numbers = []
        mutation_number_stds = []
        for time in average_times:
            all_values_at_time = []
            for freq in mutation_number_by_frequency:
                if time in mutation_number_by_frequency[freq]:
                    values = mutation_number_by_frequency[freq][time]
                    all_values_at_time.extend(values)
            if all_values_at_time:
                average_mutation_numbers.append(np.mean(all_values_at_time))
                mutation_number_stds.append(np.std(all_values_at_time))

        average_times = np.array(average_times)
        average_mutation_numbers = np.array(average_mutation_numbers)
        mutation_number_stds = np.array(mutation_number_stds)

        # Generate colors for different time steps
        unique_times = sorted(set(average_times))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_times)))

        if show_all_frequencies:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot Mu vs. Time for all frequencies on the left plot
            for color, time in zip(colors, unique_times):
                for freq in mutation_number_by_frequency:
                    if time in mutation_number_by_frequency[freq]:
                        mutation_numbers = mutation_number_by_frequency[freq][time]
                        ax1.scatter([time] * len(mutation_numbers), mutation_numbers, color=color, s=40)

            ax1.set_xlabel('Time (s)', fontsize=20)
            ax1.set_ylabel('Mutation number', fontsize=20)
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)
            if y_lim:
                ax1.set_ylim(y_lim)
        else:
            fig, ax2 = plt.subplots(figsize=(10, 6))

        # Plot Average Mu vs. Time with error bars on the right plot
        for i, (time, mean, std) in enumerate(zip(average_times, average_mutation_numbers, mutation_number_stds)):
            color = colors[i % len(colors)]
            ax2.errorbar(time, mean, yerr=std, fmt='o', color=color, ecolor='black', markersize=8, elinewidth=3,
                         capsize=4)

        ax2.set_xlabel('Time (s)', fontsize=20)
        ax2.set_ylabel('Mutation number (average)', fontsize=20)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        if y_lim:
            ax2.set_ylim(y_lim)

        plt.tight_layout()

        return fig, (ax1, ax2) if show_all_frequencies else (fig, ax2)


class MutationNumber(MeasurementSetTransform):
    """
    A transform class to calculate and visualize the mutation number from measurement sets.

    Args:
    -----
    state : str
        The state variable to be used (default is 'time').
    state_sampling : str
        The method to determine the state value ('average', 'first point', 'last point').
    dependent_variable : str
        The dependent variable to be used for mutation number calculation (default is 'complex modulus').

    Methods:
    --------
    _state_to_condition(self, target):
        Computes the specified state's value using the defined method and adds it as a condition in the dataset.

    _apply(self, target, **kwargs):
        Applies the mutation number calculation to the dataset and returns a MutationNumberMeasurementSet object.
    """

    def __init__(self, state='time', state_sampling='first point', dependent_variable='complex modulus', *args, **kwargs):
        """
        Initializes the MutationNumber transform.

        Parameters:
        -----------
        state : str
            The state variable to be used (default is 'time').
        state_mode : str
            The method to determine the state value ('average', 'first point', 'last point').
        dependent_variable : str
            The dependent variable to be used for mutation number calculation (default is 'complex modulus').
        """
        super().__init__(*args, **kwargs)
        self.state = state
        self.state_sampling = state_sampling
        self.dependent_variable = dependent_variable

    def _state_to_condition(self, target):
        """
        Applies the transformation to compute the specified state's value and adds it as a condition. This is used when
        a physical property is collected transiently and needs to be converted to a condition at a specific time.
        For example, the average temperature at which each measurement was taken. The state_sampling value allows to choose
        between 'average', 'first point', or 'last point' of the state variable to be used as the condition.

        Parameters:
        -----------
        target : MeasurementSet
            The target dataset to which the transformation is applied.

        Raises:
        -------
        ValueError: If the state name is not found in the dataset or if the state sampling method is invalid.
        """
        for measurement in target.measurements:
            state_found = False
            for dataset in measurement.datasets:
                if self.state in dataset._data_array_names:
                    dataset.switch_coordinates(independent_name='temperature', dependent_name=self.state)
                    if self.state_sampling == 'average':
                        value = np.average(dataset.y_values)
                    elif self.state_sampling == 'first point':
                        value = dataset.y_values[0]
                    elif self.state_sampling == 'last point':
                        value = dataset.y_values[-1]
                    else:
                        raise ValueError("Invalid method. Choose 'average', 'first point', or 'last point'")
                    measurement.add_condition(f'{self.state}', value)
                    state_found = True
                    break
            if not state_found:
                raise ValueError(f"State name {self.state} not found in datasets for one of the measurements")

    def _apply(self, target, **kwargs):
        """
        Applies the mutation number calculation to the target dataset and returns a MutationNumberMeasurementSet.

        Parameters:
        -----------
        target : MeasurementSet
            The target dataset containing the measurements to be processed.

        Returns:
        --------
        MutationNumberMeasurementSet : The dataset containing mutation numbers for different frequencies.

        Raises:
        -------
        ValueError: If state conditions or mutation data are missing or invalid.
        """
        self._state_to_condition(target)
        moduli = []
        frequencies = []
        wave_start_times = []
        waiting_times = []
        wave_durations = []

        # Extract data and prepare for mutation number calculation
        for i, measurement in enumerate(target.measurements):
            dataset_frequency = measurement.datasets[0]
            dataset_frequency.switch_coordinates(independent_name='angular frequency',
                                                 dependent_name=self.dependent_variable)
            frequency = copy.deepcopy(dataset_frequency.x_values)
            modulus = copy.deepcopy(dataset_frequency.y_values)

            wave_start_time = measurement.conditions['time']
            waiting_time = measurement.details['waiting_time']


            dataset_time = measurement.datasets[1]
            dataset_time.switch_coordinates('step time', 'time')
            step_time = copy.deepcopy(dataset_time.x_values)

            wave_duration = max(step_time)

            moduli.append(modulus)
            frequencies.append(frequency)
            wave_start_times.append(wave_start_time)
            waiting_times.append(waiting_time)
            wave_durations.append(wave_duration)

        moduli_by_frequency = defaultdict(lambda: defaultdict(list))
        for i, freq_list in enumerate(frequencies):
            for j, freq in enumerate(freq_list):
                moduli_by_frequency[freq][wave_start_times[i]].append(moduli[i][j])

        mutation_number_by_frequency = defaultdict(lambda: defaultdict(list))
        T_values = [wave_duration - waiting_time for wave_duration, waiting_time in zip(wave_durations, waiting_times)]
        measurements = []
        for i, freq_list in enumerate(frequencies):
            if i > 0:
                # Use wave_start_times to calculate time difference (delta_t)
                time_diff = wave_start_times[i] + waiting_times[i] - (wave_start_times[i - 1] + waiting_times[i -1])
                T = T_values[i]
                for j, freq in enumerate(freq_list):
                    if wave_start_times[i] != wave_start_times[i - 1]:
                        ln_modulus_curr = np.log(moduli[i][j])
                        ln_modulus_prev = np.log(moduli[i - 1][j])
                        derivative_ln_modulus = (ln_modulus_curr - ln_modulus_prev) / time_diff  # Updated time difference
                        if derivative_ln_modulus != 0:
                            mutation_number = T / (1 / derivative_ln_modulus)
                            mutation_number_by_frequency[freq][wave_start_times[i]].append(mutation_number)

        # Create Measurement objects with mutation number data
        for freq, time_dict in mutation_number_by_frequency.items():
            all_times = []
            all_mutation_numbers = []
            for time, mutation_numbers in time_dict.items():
                all_times.extend([time] * len(mutation_numbers))
                all_mutation_numbers.extend(mutation_numbers)

            # Create the mutation number dataset
            mutation_number_dataset = one_dimensional_dataset.OneDimensionalDataset(
                dependent_variable_data=np.array(all_mutation_numbers),
                dependent_variable_names=['mutation number'],
                dependent_variable_units=['a.u.'],
                independent_variable_data=[np.array(all_times)],
                independent_variable_names=['time'],
                independent_variable_units=['s'],
                source='datasets in time and frequency domain')

            # Create a Measurement object
            measurements.append(
                Measurement(datasets=[mutation_number_dataset], conditions={'Angular Frequency': freq}, details={}))

        return MutationNumberMeasurementSet(measurements=measurements)
