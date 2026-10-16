import numpy as np
import pandas as pd
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
from piblin.data import MeasurementSet, OneDimensionalCompositeDataset
from piblin.data.data_collections.measurement import Measurement
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform

from hermes_rheo import __version__
from hermes_rheo.transforms.states_to_conditions import summarize_state

METHOD = "mastercurves (Gaussian-process superposition; Lennon et al. 2023, doi:10.1017/dce.2023.3)"


class MasterCurveExtended(MasterCurve):
    """:class:`mastercurves.MasterCurve` with an Excel export."""

    def to_excel(self, filename):
        """Write original and shifted data to an Excel file in long format.

        Columns are ``State``, ``X_Transformed``, ``Y_Transformed``,
        ``X_Data`` and ``Y_Data``. Values are returned to linear scale
        (the master curve works on log-transformed data).

        Parameters
        ----------
        filename : str or path-like
            Output ``.xlsx`` path.

        Raises
        ------
        ValueError
            If the master curve has not been constructed.
        """
        if (
            not hasattr(self, "states")
            or not hasattr(self, "xtransformed")
            or not hasattr(self, "ytransformed")
        ):
            raise ValueError(
                "Master curve data is missing. Ensure the master curve is properly constructed."
            )

        def flat_exp(blocks):
            return np.exp([val for block in blocks for val in block])

        df_long = pd.DataFrame(
            {
                "State": np.repeat(self.states, [len(block) for block in self.xtransformed]),
                "X_Transformed": flat_exp(self.xtransformed),
                "Y_Transformed": flat_exp(self.ytransformed),
                "X_Data": flat_exp(self.xdata),
                "Y_Data": flat_exp(self.ydata),
            }
        )
        df_long.to_excel(filename, index=False)
        print(f"Data has been exported to {filename}.")


def _units_by_name(measurement):
    """Map data-array names to units over all datasets of *measurement*."""
    units = {}
    for dataset in measurement.datasets:
        for name, unit in zip(dataset.data_array_names, dataset.data_array_units, strict=False):
            units.setdefault(name, unit)
    return units


def _shared_items(dicts):
    """Return the key/value pairs identical across all *dicts*."""
    if not dicts:
        return {}
    shared = dict(dicts[0])
    for other in dicts[1:]:
        shared = {k: v for k, v in shared.items() if k in other and np.array_equal(other[k], v)}
    return shared


class MasterCurveMeasurement(Measurement):
    """Piblin measurement holding a superposed master curve.

    Datasets
    --------
    0. Master curve: one-dimensional composite dataset with the shifted
       abscissa (``x``), shifted ordinate (``y``) and the ``state`` of each
       point, in linear scale and sorted by ``x``.
    1. Shift factors: horizontal shift factor (and vertical shift factor if
       fitted) against the state.

    Details hold the method, reference state, shift parameters and
    uncertainties, and software. The fitted :class:`mastercurves.MasterCurve`
    is available as :attr:`master_curve`.

    Parameters
    ----------
    master_curve : MasterCurveExtended
        The fitted master curve (log-transformed data).
    state, x, y : str
        Names of the shift variable, abscissa and ordinate arrays.
    units : dict of str to str
        Units by array name, as found on the source datasets.
    conditions, details : dict, optional
        Metadata copied from the source measurements.
    reference_state : float, optional
        Reference state. Defaults to the state whose shift factor is closest
        to one.
    """

    def __init__(
        self, master_curve, state, x, y, units, conditions=None, details=None, reference_state=None
    ):
        self.master_curve = master_curve
        self._build_args = (state, x, y, units, conditions, details)
        x_all = np.exp(np.concatenate([np.asarray(b, float) for b in master_curve.xtransformed]))
        y_all = np.exp(np.concatenate([np.asarray(b, float) for b in master_curve.ytransformed]))
        s_all = np.repeat(
            np.asarray(master_curve.states, float),
            [len(b) for b in master_curve.xtransformed],
        )
        order = np.argsort(x_all, kind="stable")
        curve = OneDimensionalCompositeDataset(
            data_arrays=[x_all[order], y_all[order], s_all[order]],
            data_array_names=[x, y, state],
            data_array_units=[units.get(x, ""), units.get(y, ""), units.get(state, "")],
            default_independent_name=x,
            default_dependent_name=y,
        )

        arrays = [
            np.asarray(master_curve.states, float),
            np.asarray(master_curve.hparams[0], float),
        ]
        names = [state, "horizontal shift factor"]
        shift_units = [units.get(state, ""), ""]
        if master_curve.vparams:
            arrays.append(np.asarray(master_curve.vparams[0], float))
            names.append("vertical shift factor")
            shift_units.append("")
        shifts = OneDimensionalCompositeDataset(
            data_arrays=arrays,
            data_array_names=names,
            data_array_units=shift_units,
            default_independent_name=state,
            default_dependent_name="horizontal shift factor",
        )

        a_t = np.asarray(master_curve.hparams[0], float)
        states = np.asarray(master_curve.states, float)
        meta = dict(details or {})
        meta.update(
            {
                "method": METHOD,
                "reference": "https://doi.org/10.1017/dce.2023.3",
                "software": f"hermes-rheo {__version__}; mastercurves",
                "state": state,
                "x": x,
                "y": y,
                "reference_state": float(
                    states[np.argmin(np.abs(np.log(a_t)))]
                    if reference_state is None
                    else reference_state
                ),
                "horizontal_shift_factors": a_t.tolist(),
                "horizontal_shift_uncertainties": np.asarray(
                    master_curve.huncertainties[0], float
                ).tolist(),
                "vertical_shift": bool(master_curve.vparams),
            }
        )
        if master_curve.vparams:
            meta["vertical_shift_factors"] = np.asarray(master_curve.vparams[0], float).tolist()
            meta["vertical_shift_uncertainties"] = np.asarray(
                master_curve.vuncertainties[0], float
            ).tolist()
        super().__init__(datasets=[curve, shifts], conditions=conditions, details=meta)

    @property
    def states(self):
        """States of the superposed data, in superposition order."""
        return self.master_curve.states

    @property
    def hparams(self):
        """Horizontal shift parameters (``hparams[0]`` holds the shift factors)."""
        return self.master_curve.hparams

    @property
    def vparams(self):
        """Vertical shift parameters."""
        return self.master_curve.vparams

    @property
    def xtransformed(self):
        """Log-scale shifted abscissa blocks, one per state."""
        return self.master_curve.xtransformed

    @property
    def ytransformed(self):
        """Log-scale shifted ordinate blocks, one per state."""
        return self.master_curve.ytransformed

    def change_ref(self, state):
        """Change the reference state and refresh the piblin datasets and details."""
        self.master_curve.change_ref(state)
        self.__init__(self.master_curve, *self._build_args, reference_state=state)

    def plot(self, *args, **kwargs):
        """Plot the master curve; see :meth:`mastercurves.MasterCurve.plot`."""
        return self.master_curve.plot(*args, **kwargs)

    def to_excel(self, filename):
        """Export original and shifted data; see :meth:`MasterCurveExtended.to_excel`."""
        self.master_curve.to_excel(filename)


class MasterCurveSet(MeasurementSet):
    """Piblin measurement set of master curves, one per quantity.

    Provides :meth:`get_mastercurve` and :attr:`available_quantities` for
    downstream steps such as :class:`~hermes_rheo.transforms.fit_model.FitModel`.
    """

    def __init__(self, measurements=None, quantities=None):
        super().__init__(measurements=measurements, merge_redundant=False)
        self.quantities = list(quantities or [])

    def get_mastercurve(self, quantity):
        """Return the master-curve measurement for *quantity* (case-insensitive), or None."""
        q_norm = str(quantity).strip().lower()
        for name, measurement in zip(self.quantities, self.measurements, strict=False):
            if str(name).strip().lower() == q_norm:
                return measurement
        return None

    @property
    def available_quantities(self):
        """Lower-cased names of the quantities with a master curve."""
        return [str(q).strip().lower() for q in self.quantities]


class AutomatedMasterCurve(MeasurementSetTransform):
    """Superpose a measurement set into a master curve (Gaussian-process method).

    Wraps the ``mastercurves`` package, which learns a Gaussian-process model
    of the data and finds shift factors by maximum a posteriori estimation
    [1]_. Data are log-transformed; horizontal shifts (and optionally
    vertical shifts) are multiplicative.

    Parameters
    ----------
    state : str, default "temperature"
        Name of the data array used as the shift variable. Its value per
        measurement is summarized with
        :func:`~hermes_rheo.transforms.states_to_conditions.summarize_state`
        and stored as a condition.
    state_mode : {"average", "first point", "last point"}, default "average"
        How each measurement's state value is computed from its array.
    x : str, default "angular frequency"
        Name of the abscissa array.
    y : str, default "storage modulus"
        Name of the ordinate array.
    vertical_shift : bool, default False
        Also fit multiplicative vertical shift factors.
    reverse_data : bool, default False
        Reverse the (sorted) data order before superposition.
    measurements : list of int, optional
        Indices of measurements to use. All are used if None.
    sort_states : bool, default True
        Sort the measurements by ascending state value.
    ref_state_idx : int, optional
        Index into the ordered states of the reference state; if given, the
        shift factors are re-referenced to it.

    Returns
    -------
    MasterCurveMeasurement
        From ``apply_to``: a piblin measurement with the master-curve
        dataset (shifted ``x``, ``y`` and ``state``) and a shift-factor
        dataset (state against horizontal/vertical shift factors). Details
        hold the method, reference state, shift factors and uncertainties,
        and software. The ``mastercurves`` object is ``result.master_curve``.

    References
    ----------
    .. [1] Lennon, K. R., McKinley, G. H. and Swan, J. W. "A data-driven
       method for automated data superposition with applications in soft
       matter science", Data-Centric Engineering 4, e13 (2023).
       https://doi.org/10.1017/dce.2023.3
       Code: https://github.com/krlennon/mastercurves
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
        ref_state_idx=None,
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
        self.ref_state_idx = ref_state_idx

    def _apply(self, target, **kwargs):
        """Superpose the selected measurements.

        Returns
        -------
        MasterCurveMeasurement
            The superposed master curve.

        Raises
        ------
        ValueError
            If the state is missing from a measurement or ``state_mode`` is invalid.
        """
        for measurement in target.measurements:
            measurement.add_condition(
                self.state, summarize_state(measurement, self.state, self.state_mode)
            )

        states = []
        x_data = []
        y_data = []

        measurements_to_use = target.measurements
        if self.measurements is not None:
            measurements_to_use = [target.measurements[i] for i in self.measurements]

        for measurement in measurements_to_use:
            dataset = measurement.datasets[0]
            dataset.switch_coordinates(independent_name=self.x, dependent_name=self.y)
            x_values = dataset.x_values
            y_values = dataset.y_values

            # Only NaNs in log(y) (non-positive y) are dropped.
            log_x_values = np.log(x_values)
            log_y_values = np.log(y_values)

            valid_indices = ~np.isnan(log_y_values)

            states.append(measurement.conditions[self.state])
            x_data.append(log_x_values[valid_indices])
            y_data.append(log_y_values[valid_indices])

        if self.sort_states:
            order = np.argsort(states)
            states = [states[i] for i in order]
            x_data = [x_data[i] for i in order]
            y_data = [y_data[i] for i in order]

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

        if self.ref_state_idx is not None:
            master_curve.change_ref(master_curve.states[self.ref_state_idx])

        units = _units_by_name(measurements_to_use[0])
        conditions = _shared_items([m.conditions for m in measurements_to_use])
        conditions.pop(self.state, None)
        details = _shared_items([m.details for m in measurements_to_use])
        return MasterCurveMeasurement(
            master_curve, self.state, self.x, self.y, units, conditions, details
        )


class CompositeMasterCurve(MeasurementSetTransform):
    """Build one master curve per y-quantity.

    Returns a :class:`MasterCurveSet` (a piblin measurement set) holding one
    :class:`MasterCurveMeasurement` per quantity, usable by downstream steps
    such as ``FitModel`` through :attr:`MasterCurveSet.available_quantities`
    and :meth:`MasterCurveSet.get_mastercurve`.

    Parameters
    ----------
    n_curves : int, default 2
        Number of quantities; must equal ``len(y_quantities)``.
    y_quantities : list of str
        Ordinate arrays to superpose (e.g. ``["storage modulus", "loss modulus"]``).
    quantities : list of str, optional
        Alias for ``y_quantities``, used when that is None.
    x, state, state_mode, vertical_shift, reverse_data, measurements, sort_states
        See :class:`AutomatedMasterCurve`.
    ref_temp_idx : int, optional
        Index into the sorted states of the reference state; if given, each
        master curve is re-referenced to it.

    References
    ----------
    .. [1] Lennon, K. R., McKinley, G. H. and Swan, J. W. "A data-driven
       method for automated data superposition with applications in soft
       matter science", Data-Centric Engineering 4, e13 (2023).
       https://doi.org/10.1017/dce.2023.3
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

    def _apply(self, target, **kwargs):
        curves = [
            AutomatedMasterCurve(
                state=self.state,
                state_mode=self.state_mode,
                x=self.x,
                y=yq,
                vertical_shift=self.vertical_shift,
                reverse_data=self.reverse_data,
                measurements=self.measurements,
                sort_states=self.sort_states,
                ref_state_idx=self.ref_temp_idx,
            ).apply_to(target, make_copy=False, **kwargs)
            for yq in self.y_quantities
        ]
        return MasterCurveSet(curves, quantities=self.y_quantities)
