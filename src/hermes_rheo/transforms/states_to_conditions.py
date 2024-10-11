from piblin.transform.abc.measurement_transform import MeasurementTransform
#from cralds.transform.new_transforms.abstract_base_classes.measurement_transform import MeasurementTransform



import numpy as np


class StatesToConditions(MeasurementTransform):
    def __init__(self, state: str = 'temperature', method: str = 'average', *args, **kwargs):
        """
        Initializes the StatesToCondition object with the specified state, method, and chirp flag.

        Args:
            state (str): The state for which the value is to be computed. Defaults to 'temperature'.
            method (str): The method to compute the value ('average', 'first point', 'last point'). Defaults to 'average'.
            chirp (bool): Flag to determine which dataset to use. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.state = state
        self.method = method

    def _apply(self, target, **kwargs) -> object:
        """
        Applies the transformation to compute the specified value of the state and adds it as a condition.

        Args:
            target (object): The target dataset to which the transformation is applied.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The modified target with the new condition added.

        Raises:
            ValueError: If the state name is not a string.
        """
        state_name = self.state

        if not isinstance(state_name, str):
            raise ValueError("state_name must be a string")

        # Select dataset based on chirp value
        value = None
        for dataset in target.datasets:
            if state_name in dataset._data_array_names:
                dataset.switch_coordinates(independent_name='temperature', dependent_name=state_name)
                # Choose method to compute the value
                if self.method == 'average':
                    value = np.average(dataset.y_values)
                elif self.method == 'first point':
                    value = dataset.y_values[0]
                elif self.method == 'last point':
                    value = dataset.y_values[-1]
                else:
                    raise ValueError("Invalid method. Choose 'average', 'first point', or 'last point'")
                break  # Exit the loop after finding and setting the value

        if value is None:
            raise ValueError("State name not found in datasets", state_name)

        target.add_condition(f'{state_name}', value)

        return target
