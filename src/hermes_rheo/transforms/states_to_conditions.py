import numpy as np
from piblin.transform.abc.measurement_transform import MeasurementTransform

STATE_METHODS = ("average", "first point", "last point")


def summarize_state(measurement, state: str, method: str = "average") -> float:
    """Summarize a measured state array of a measurement as one value.

    The first dataset containing the data array *state* is used; the
    dataset is not modified.

    Parameters
    ----------
    measurement : piblin.data.Measurement
        Measurement holding the datasets.
    state : str
        Name of the data array to summarize.
    method : {"average", "first point", "last point"}, default "average"
        Aggregation applied to the state's values.

    Returns
    -------
    float
        The mean, first or last value of the array.

    Raises
    ------
    ValueError
        If *state* is not a string, *method* is invalid, or *state* is not
        found in any dataset of *measurement*.
    """
    if not isinstance(state, str):
        raise ValueError("state_name must be a string")
    if method not in STATE_METHODS:
        raise ValueError("Invalid method. Choose 'average', 'first point', or 'last point'")

    for dataset in measurement.datasets:
        names = list(dataset.data_array_names)
        if state in names:
            values = np.asarray(dataset.data_arrays[names.index(state)])
            if method == "average":
                return np.average(values)
            return values[0] if method == "first point" else values[-1]

    raise ValueError(f"State name not found in datasets: {state!r}")


class StatesToConditions(MeasurementTransform):
    """Store a summary of a measured state as a measurement condition.

    The first dataset containing the array *state* is used. The summary
    (mean, first or last point) is added to ``target.conditions[state]``.
    The dataset's independent variable is left unchanged.

    Parameters
    ----------
    state : str, default "temperature"
        Name of the data array to summarize.
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
        """Add the summarized state value as a condition on *target*.

        Raises
        ------
        ValueError
            If ``state`` is not a string, is not found in any of the target's
            datasets, or ``method`` is not one of the accepted values.
        """
        target.add_condition(self.state, summarize_state(target, self.state, self.method))
        return target
