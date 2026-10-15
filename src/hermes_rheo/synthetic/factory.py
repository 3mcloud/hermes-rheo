"""Factory for classical LVE synthetic-data models.

:class:`SyntheticRheometry` is a thin factory with one class method per
constitutive model (Maxwell, multi-mode Maxwell, Kelvin--Voigt,
Jeffreys).  Each method returns a
:class:`~hermes_rheo.synthetic.models.RheologyModel` instance that
exposes :meth:`~.RheologyModel.stress_relaxation`,
:meth:`~.RheologyModel.frequency_sweep`,
:meth:`~.RheologyModel.creep` and
:meth:`~.RheologyModel.to_measurement`.

The class also exposes :meth:`experiment_set` for composing several
synthetic experiments on the same model into a single
:class:`~piblin.data.data_collections.experiment_set.ExperimentSet`
that can be fed straight into a hermes-rheo pipeline.

Examples
--------
>>> import numpy as np
>>> from hermes_rheo.synthetic import SyntheticRheometry
>>> model = SyntheticRheometry.maxwell(G0=1e5, tau=1.0)
>>> t = np.geomspace(1e-3, 1e2, 200)
>>> meas = model.to_measurement(experiment="stress_relaxation", x=t)
>>> exp = SyntheticRheometry.experiment_set(
...     model,
...     experiments=[
...         {"experiment": "stress_relaxation", "x": t},
...         {"experiment": "frequency_sweep", "x": np.geomspace(1e-2, 1e2, 100)},
...     ],
... )
>>> len(exp.measurements)
2
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from piblin.data.data_collections import experiment_set as _experiment_set
from piblin.data.data_collections import measurement_set as _measurement_set

from hermes_rheo.synthetic.models import (
    JeffreysModel,
    KelvinVoigtModel,
    MaxwellModel,
    MultiModeMaxwellModel,
    RheologyModel,
)

__all__ = ["SyntheticRheometry"]


class SyntheticRheometry:
    """Factory for classical linear-viscoelastic constitutive models.

    All methods are ``@classmethod`` and return concrete
    :class:`~hermes_rheo.synthetic.models.RheologyModel` instances.
    Instantiating :class:`SyntheticRheometry` itself is unnecessary
    (though allowed); the class is a namespace.

    See Also
    --------
    hermes_rheo.synthetic.models.RheologyModel
        Base class describing the API exposed by every returned model.
    """

    @classmethod
    def maxwell(cls, G0: float, tau: float) -> MaxwellModel:
        """Build a single-mode :class:`MaxwellModel`.

        Parameters
        ----------
        G0 : float
            Instantaneous modulus (Pa).
        tau : float
            Relaxation time (s).

        Returns
        -------
        MaxwellModel

        Examples
        --------
        >>> m = SyntheticRheometry.maxwell(G0=1e5, tau=1.0)
        >>> type(m).__name__
        'MaxwellModel'
        """
        return MaxwellModel(G0=G0, tau=tau)

    @classmethod
    def multi_mode_maxwell(
        cls,
        g: Sequence[float],
        tau: Sequence[float],
        G_inf: float = 0.0,
    ) -> MultiModeMaxwellModel:
        """Build a :class:`MultiModeMaxwellModel` (Prony series).

        Parameters
        ----------
        g : sequence of float
            Mode weights :math:`g_i` (Pa).
        tau : sequence of float
            Relaxation times :math:`\\tau_i` (s).
        G_inf : float, optional
            Plateau modulus (Pa).  Default 0 (viscoelastic fluid).

        Returns
        -------
        MultiModeMaxwellModel

        Examples
        --------
        >>> m = SyntheticRheometry.multi_mode_maxwell(
        ...     g=[1e5, 1e4], tau=[0.1, 1.0]
        ... )
        >>> len(m.tau)
        2
        """
        return MultiModeMaxwellModel(g=g, tau=tau, G_inf=G_inf)

    @classmethod
    def kelvin_voigt(cls, G: float, eta: float) -> KelvinVoigtModel:
        """Build a :class:`KelvinVoigtModel` (spring \\+ dashpot in parallel).

        Parameters
        ----------
        G : float
            Elastic modulus (Pa).
        eta : float
            Dashpot viscosity (Pa*s).

        Returns
        -------
        KelvinVoigtModel
        """
        return KelvinVoigtModel(G=G, eta=eta)

    @classmethod
    def jeffreys(
        cls,
        G0: float,
        tau: float,
        eta_s: float,
    ) -> JeffreysModel:
        """Build a :class:`JeffreysModel`.

        Parameters
        ----------
        G0 : float
            Maxwell-branch elastic modulus (Pa).
        tau : float
            Maxwell-branch relaxation time (s).
        eta_s : float
            Solvent viscosity of the parallel dashpot (Pa*s).

        Returns
        -------
        JeffreysModel
        """
        return JeffreysModel(G0=G0, tau=tau, eta_s=eta_s)

    # -----------------------------------------------------------------
    # Composition helper
    # -----------------------------------------------------------------

    @classmethod
    def experiment_set(
        cls,
        model: RheologyModel,
        experiments: Sequence[Dict[str, Any]],
        *,
        merge_redundant: bool = False,
    ) -> _experiment_set.ExperimentSet:
        """Compose several synthetic experiments on *model* into an ``ExperimentSet``.

        Each entry in *experiments* is a dict of keyword arguments
        forwarded to :meth:`RheologyModel.to_measurement`.  The required
        keys are ``"experiment"`` (one of ``"stress_relaxation"``,
        ``"frequency_sweep"``, ``"creep"``) and ``"x"`` (observation
        grid); optional keys ``"conditions"``, ``"details"``, ``"source"``
        pass through unchanged.

        Parameters
        ----------
        model : RheologyModel
        experiments : sequence of dict
            One dict per synthetic experiment.
        merge_redundant : bool, optional
            Forwarded to
            :class:`~piblin.data.data_collections.experiment_set.ExperimentSet`.
            Default ``False`` because different experiments carry
            different ``conditions["method"]`` values and should NOT be
            merged.

        Returns
        -------
        piblin.data.data_collections.experiment_set.ExperimentSet

        Raises
        ------
        ValueError
            If *experiments* is empty or any entry is missing required
            keys.
        """
        if not experiments:
            raise ValueError("experiments must contain at least one entry")

        measurements = []
        for i, entry in enumerate(experiments):
            if "experiment" not in entry or "x" not in entry:
                raise ValueError(
                    f"experiments[{i}] must contain 'experiment' and 'x' keys, got {entry!r}"
                )
            measurements.append(model.to_measurement(**entry))

        try:
            return _experiment_set.ExperimentSet(
                measurements=measurements,
                merge_redundant=merge_redundant,
            )
        except Exception:
            # Fall back to a plain MeasurementSet when experiments carry
            # different columns/units (e.g. mixing stress_relaxation +
            # frequency_sweep on the same model).
            return _measurement_set.MeasurementSet(measurements=measurements)
