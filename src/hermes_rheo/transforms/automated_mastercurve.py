from cralds_base.transform.abc.measurement_set_transform import MeasurementSetTransform
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
import numpy as np


class AutomatedMasterCurve(MeasurementSetTransform):
    """
    AutomatedMasterCurve is a class for constructing master curves using a data-driven  algorithm which
    employs Gaussian process regression to learn statistical models that describe the data, and then uses maximum a
    posteriori estimation to optimally superpose the data sets . It supports customization of the state and property
    for the master curve creation and allows for optional vertical shift if needed.

    The algorithm was developed by the Swan and McKinley lab at MIT. For more information, see:
    "A data-driven method for automated data superposition with applications in soft matter science"
    Data-Centric Engineering , Volume 4 , 2023 , e13
    DOI: https://doi.org/10.1017/dce.2023.3

    The implementation in this class is based on an open-source Python package developed by Kyle Lennon. For
    source code, visit: https://github.com/krlennon/mastercurves. Documentation for the package can be found at:
    https://krlennon-mastercurves.readthedocs.io.

    Note: The mastercurves package is released under the GNU General Public License v3.0. This class, developed
    for use within 3M, complies with these licensing terms and is intended for internal use only.

    Args:
        state (str): The state based on which the master curve is constructed (e.g., 'time', 'temperature'). Defaults to 'temperature'.
        method (str): The method to compute the state value ('average', 'first point', 'last point'). Only used if state is a dataset variable. Defaults to 'average'.
        x (str): The x-axis data for the master curve (e.g., 'angular frequency'). Defaults to 'angular frequency'.
        y (str): The y-axis data for the master curve (e.g., 'storage modulus'). Defaults to 'storage modulus'.
        vertical_shift (bool): Whether to apply a vertical shift. Defaults to False.
        reverse_data (bool): Whether to reverse the data order. Defaults to False.
        *args, **kwargs: Additional arguments and keyword arguments for superclass initialization.

    Methods:
        _apply(self, target, **kwargs):
            Applies the algorithm to the target dataset to create a master curve. This method integrates the data
            processing, state conditioning, and master curve creation into a single automated process.

        _states_to_condition(self, target):
            Computes the specified state's value using the defined method ('average', 'first point', 'last point')
            and adds it as a condition to the target dataset. This method is used internally to condition the data
            before applying the master curve algorithm.
    """

    def __init__(self, state='temperature', state_mode='average', x='angular frequency', y='storage modulus',
                 vertical_shift=False, reverse_data=False, measurements=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.state_mode = state_mode
        self.x = x
        self.y = y
        self.vertical_shift = vertical_shift
        self.reverse_data = reverse_data
        self.measurements = measurements

    def _state_to_condition(self, target):
        """
        Applies the transformation to compute the specified state's value and adds it as a condition.

        Args:
            target (object): The target dataset to which the transformation is applied.

        Returns:
            object: The target with the new condition added.

        Raises:
            ValueError: If the state name is not a string or if the specified method is invalid.
        """
        for measurement in target.measurements:
            state_found = False
            for dataset in measurement.datasets:
                if self.state in dataset._data_array_names:
                    dataset.switch_coordinates(independent_name='temperature', dependent_name=self.state)
                    if self.state_mode == 'average':
                        value = np.average(dataset.y_values)
                    elif self.state_mode == 'first point':
                        value = dataset.y_values[0]
                    elif self.state_mode == 'last point':
                        value = dataset.y_values[-1]
                    else:
                        raise ValueError("Invalid method. Choose 'average', 'first point', or 'last point'")
                    measurement.add_condition(f'{self.state}', value)
                    state_found = True
                    break  # Break after adding the condition to the current measurement

            if not state_found:
                raise ValueError(f"State name {self.state} not found in datasets for one of the measurements")

    def _apply(self, target, **kwargs):
        self._state_to_condition(target)
        states = []
        x_data = []
        y_data = []

        measurements_to_use = target.measurements
        if self.measurements is not None:
            measurements_to_use = [target.measurements[i] for i in self.measurements]

        for i, measurement in enumerate(measurements_to_use):
            dataset = measurement.datasets[0]
            dataset.switch_coordinates(independent_name=self.x, dependent_name=self.y)
            x_values = dataset.x_values
            y_values = dataset.y_values

            # Handling external array of state values
            if isinstance(self.state, (list, np.ndarray)):
                if i < len(self.state):
                    state_value = self.state[i]
                else:
                    raise ValueError("Length of the 'state' array does not match the number of measurements.")
            else:
                # Handling state as a condition name within the dataset
                state_value = measurement.conditions.get(f'{self.state}')
                if state_value is None:
                    raise ValueError(f"The specified state '{self.state}' is not found in the conditions.")

            # Compute logarithms and filter out NaNs
            log_x_values = np.log(x_values)
            log_y_values = np.log(y_values)

            # Create a mask that keeps only non-NaN values in y_data
            valid_indices = ~np.isnan(log_y_values)

            # Apply the mask to both x_values and y_values
            filtered_x_values = log_x_values[valid_indices]
            filtered_y_values = log_y_values[valid_indices]

            states.append(state_value)
            x_data.append(filtered_x_values)
            y_data.append(filtered_y_values)

        # Reverse the data if the reverse_data flag is True
        if self.reverse_data:
            states.reverse()
            x_data.reverse()
            y_data.reverse()

        master_curve = MasterCurve()
        master_curve.add_data(x_data, y_data, states)
        master_curve.add_htransform(Multiply())

        if self.vertical_shift:
            master_curve.add_vtransform(Multiply())

        master_curve.superpose()

        return master_curve

