import numpy as np
import pandas as pd
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform


class MasterCurveExtended(MasterCurve):
    """
    Extends the MasterCurve class with additional functionality such as exporting to Excel.
    """

    def to_excel(self, filename):
        """
        Exports the master curve data to an Excel file.

        Args:
            filename (str): The path to the Excel file to save the data.
        """
        if (
            not hasattr(self, "states")
            or not hasattr(self, "xtransformed")
            or not hasattr(self, "ytransformed")
        ):
            raise ValueError(
                "Master curve data is missing. Ensure the master curve is properly constructed."
            )

        # Extract the data from the master curve
        states = self.states
        xtransformed = self.xtransformed
        ytransformed = self.ytransformed
        xdata = self.xdata
        ydata = self.ydata

        # Flatten the arrays
        xtransformed_flat = [val for sublist in xtransformed for val in sublist]
        ytransformed_flat = [val for sublist in ytransformed for val in sublist]
        xdata_flat = [val for sublist in xdata for val in sublist]
        ydata_flat = [val for sublist in ydata for val in sublist]

        # Repeat the states for each corresponding row in xtransformed and ytransformed
        states_flat = np.repeat(states, [len(sublist) for sublist in xtransformed])

        # Compute the exponent of the transformed values to reverse the log transformation
        xtransformed_exp = np.exp(xtransformed_flat)
        ytransformed_exp = np.exp(ytransformed_flat)
        xdata_exp = np.exp(xdata_flat)
        ydata_exp = np.exp(ydata_flat)

        # Create a dictionary for the long format data
        data_long = {
            "State": states_flat,
            "X_Transformed": xtransformed_exp,
            "Y_Transformed": ytransformed_exp,
            "X_Data": xdata_exp,
            "Y_Data": ydata_exp,
        }

        # Convert to DataFrame
        df_long = pd.DataFrame(data_long)

        # Export to Excel
        df_long.to_excel(filename, index=False)
        print(f"Data has been exported to {filename}.")


class AutomatedMasterCurve(MeasurementSetTransform):
    """
    AutomatedMasterCurve is a class for constructing master curves using a data-driven algorithm which
    employs Gaussian process regression to learn statistical models that describe the data, and then uses maximum a
    posteriori estimation to optimally superpose the data sets. It supports customization of the state and property
    for the master curve creation and allows for optional vertical shift if needed.

    The algorithm was developed by the Swan and McKinley lab at MIT. For more information, see:
    Lennon, K. R., McKinley, G. H. and Swan, J. W.,
    "A data-driven method for automated data superposition with applications in soft matter science",
    Data-Centric Engineering, Volume 4, 2023, e13.
    DOI: https://doi.org/10.1017/dce.2023.3 —
    preprint: https://arxiv.org/abs/2204.09521.

    The implementation in this class is based on an open-source Python package developed by Kyle Lennon. For
    source code, visit: https://github.com/krlennon/mastercurves. Documentation for the package can be found at:
    https://krlennon-mastercurves.readthedocs.io.

    Note: The mastercurves package is released under the GNU General Public License v3.0.

    Args:
        state (str): The state based on which the master curve is constructed (e.g., 'time', 'temperature').
                     Defaults to 'temperature'.
        state_mode (str): The method to compute the state value ('average', 'first point', 'last point'). Only used
                          if the state is a dataset variable. Defaults to 'average'.
        x (str): The x-axis data for the master curve (e.g., 'angular frequency'). Defaults to 'angular frequency'.
        y (str): The y-axis data for the master curve (e.g., 'storage modulus'). Defaults to 'storage modulus'.
        vertical_shift (bool): Whether to apply a vertical shift. Defaults to False.
        reverse_data (bool): Whether to reverse the data order. Defaults to False.
        measurements (list, optional): A list of specific measurement indices to use. If not provided, all measurements
                                       in the target dataset will be used. Defaults to None.
        *args, **kwargs: Additional arguments and keyword arguments for superclass initialization.

    Methods:
        _apply(self, target, **kwargs):
            Applies the algorithm to the target dataset to create a master curve. This method integrates the data
            processing, state conditioning, and master curve creation into a single automated process.

        _state_to_condition(self, target):
            Computes the specified state's value using the defined method ('average', 'first point', 'last point')
            and adds it as a condition to the target dataset. This method is used internally to condition the data
            before applying the master curve algorithm.
    """

    def __init__(
        self,
        state="temperature",
        state_mode="average",
        x="angular frequency",
        y="storage modulus",
        vertical_shift=False,
        reverse_data=False,
        measurements=None,
        sort_states=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state = state
        self.state_mode = state_mode
        self.x = x
        self.y = y
        self.vertical_shift = vertical_shift
        self.reverse_data = reverse_data
        self.measurements = measurements
        self.sort_states = sort_states

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
                    dataset.switch_coordinates(
                        independent_name=self.state, dependent_name=self.state
                    )
                    if self.state_mode == "average":
                        value = np.average(dataset.y_values)
                    elif self.state_mode == "first point":
                        value = dataset.y_values[0]
                    elif self.state_mode == "last point":
                        value = dataset.y_values[-1]
                    else:
                        raise ValueError(
                            "Invalid method. Choose 'average', 'first point', or 'last point'"
                        )
                    measurement.add_condition(f"{self.state}", value)
                    state_found = True
                    break  # Break after adding the condition to the current measurement

            if not state_found:
                raise ValueError(
                    f"State name {self.state} not found in datasets for one of the measurements"
                )

    def _apply(self, target, **kwargs):
        """
        Applies the mastercurve algorithm to create a master curve from the target dataset.

        Args:
            target (object): The target dataset containing the data to be used for master curve construction.

        Returns:
            MasterCurve: The generated master curve based on the specified state and data.

        Raises:
            ValueError: If the state array's length does not match the number of measurements.
                        If the specified state is not found in the dataset conditions.
        """
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
                    raise ValueError(
                        "Length of the 'state' array does not match the number of measurements."
                    )
            else:
                # Handling state as a condition name within the dataset
                state_value = measurement.conditions.get(f"{self.state}")
                if state_value is None:
                    raise ValueError(
                        f"The specified state '{self.state}' is not found in the conditions."
                    )

            # Compute logarithms and filter out NaNs
            log_x_values = np.log(x_values)
            log_y_values = np.log(y_values)

            valid_indices = ~np.isnan(log_y_values)
            filtered_x_values = log_x_values[valid_indices]
            filtered_y_values = log_y_values[valid_indices]

            states.append(state_value)
            x_data.append(filtered_x_values)
            y_data.append(filtered_y_values)

        # sort by state (e.g. temperature) in ascending order
        if self.sort_states:
            # argsort gives indices that would sort 'states'
            order = np.argsort(states)
            states = [states[i] for i in order]
            x_data = [x_data[i] for i in order]
            y_data = [y_data[i] for i in order]

        # Optional: reverse final order after sorting
        if self.reverse_data:
            states = states[::-1]
            x_data = x_data[::-1]
            y_data = y_data[::-1]

        master_curve = MasterCurveExtended()
        master_curve.add_data(x_data, y_data, states)
        master_curve.add_htransform(Multiply())

        if self.vertical_shift:
            master_curve.add_vtransform(Multiply())

        master_curve.superpose()

        return master_curve


class CompositeMasterCurve(MeasurementSetTransform):
    """
    Pipeline step that generates master curves for multiple y-quantities and returns itself,
    so downstream steps (FitModel) receive an object exposing:
      - available_quantities
      - get_mastercurve(quantity)
    """

    def __init__(
        self,
        n_curves=2,
        y_quantities=None,
        quantities=None,
        x="angular frequency",
        state="temperature",
        state_mode="average",
        vertical_shift=False,
        reverse_data=False,
        measurements=None,
        sort_states=True,
        ref_temp_idx=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if y_quantities is None:
            y_quantities = quantities

        if y_quantities is None or len(y_quantities) != n_curves:
            raise ValueError("y_quantities must be a list of length n_curves")

        self.n_curves = n_curves
        self.y_quantities = y_quantities
        self.x = x
        self.state = state
        self.state_mode = state_mode
        self.vertical_shift = vertical_shift
        self.reverse_data = reverse_data
        self.measurements = measurements
        self.sort_states = sort_states
        self.ref_temp_idx = ref_temp_idx

        self.mastercurves = {}  # {quantity: mastercurve_obj}

    def _apply(self, target, **kwargs):
        # Build mastercurves dict
        self.mastercurves = {}

        for yq in self.y_quantities:
            mc = AutomatedMasterCurve(
                state=self.state,
                state_mode=self.state_mode,
                x=self.x,
                y=yq,
                vertical_shift=self.vertical_shift,
                reverse_data=self.reverse_data,
                measurements=self.measurements,
                sort_states=self.sort_states,
            ).apply_to(target, make_copy=False, **kwargs)

            if self.ref_temp_idx is not None:
                mc.change_ref(mc.states[self.ref_temp_idx])

            self.mastercurves[yq] = mc

        # IMPORTANT: return self so FitModel sees CompositeMasterCurve-like API
        return self

    def get_mastercurve(self, quantity):
        # FitModel seems to ask for lowercase keys like 'storage modulus'
        # so normalize lookup.
        if quantity in self.mastercurves:
            return self.mastercurves[quantity]
        q_norm = str(quantity).strip().lower()
        for k, v in self.mastercurves.items():
            if str(k).strip().lower() == q_norm:
                return v
        return None

    @property
    def available_quantities(self):
        return [str(k).strip().lower() for k in self.mastercurves.keys()]
