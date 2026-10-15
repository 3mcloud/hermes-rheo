from piblin.transform.abc.measurement_transform import MeasurementTransform

import numpy as np


class StatesToConditions(MeasurementTransform):
    """Compute a summary value of a per-measurement state and store it as a condition.

    Iterates over ``target.datasets`` and, for the first dataset whose
    ``data_array_names`` contain *state*, computes the mean / first / last
    point of that state's values while keeping the currently-active
    independent coordinate in place, then attaches the value to
    ``target.conditions[state]``.

    Parameters
    ----------
    state : str, default "temperature"
        Name of the data array to summarise.
    method : {"average", "first point", "last point"}, default "average"
        Aggregation applied to the state's values.

    Examples
    --------
    >>> from hermes_rheo.transforms.states_to_conditions import StatesToConditions
    >>> StatesToConditions(state="temperature", method="average")  # doctest: +SKIP
    """

    def __init__(self, state: str = "temperature", method: str = "average", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.method = method

    def _apply(self, target, **kwargs) -> object:
        """Compute the aggregated state value and add it as a condition on *target*.

        Raises
        ------
        ValueError
            If ``state`` is not a string, is not found in any of the target's
            datasets, or ``method`` is not one of the accepted values.
        """
        state_name = self.state

        if not isinstance(state_name, str):
            raise ValueError("state_name must be a string")

        value = None
        for dataset in target.datasets:
            if state_name in dataset._data_array_names:
                # Preserve the dataset's currently-active independent
                # variable — previous code hardcoded ``'temperature'`` here
                # which broke on any dataset that lacked a temperature
                # column.
                current_independent = dataset.independent_variable_name
                dataset.switch_coordinates(
                    independent_name=current_independent,
                    dependent_name=state_name,
                )
                if self.method == "average":
                    value = np.average(dataset.y_values)
                elif self.method == "first point":
                    value = dataset.y_values[0]
                elif self.method == "last point":
                    value = dataset.y_values[-1]
                else:
                    raise ValueError(
                        "Invalid method. Choose 'average', 'first point', or 'last point'"
                    )
                break

        if value is None:
            raise ValueError(f"State name not found in datasets: {state_name!r}")

        target.add_condition(f"{state_name}", value)

        return target
