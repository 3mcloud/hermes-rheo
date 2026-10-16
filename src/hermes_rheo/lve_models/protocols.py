"""Simulated rheometer protocols.

:class:`Protocols` runs SAOS, creep and stress relaxation on any
:class:`~hermes_rheo.lve_models.base.LVEModel` and returns piblin objects that
feed directly into hermes-rheo transforms.

Examples
--------
>>> import numpy as np
>>> from hermes_rheo.lve_models import MaxwellModel, Protocols
>>> model = MaxwellModel(G=1e5, tau=1.0)
>>> meas = Protocols.saos(model, np.geomspace(1e-2, 1e2, 50))
>>> meas.datasets[0].data_array_names[:3]
['angular frequency', 'storage modulus', 'loss modulus']
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from piblin.data.data_collections import experiment_set as _experiment_set
from piblin.data.data_collections import measurement as _measurement
from piblin.data.data_collections import measurement_set as _measurement_set

from hermes_rheo.lve_models.base import LVEModel

__all__ = ["Protocols"]


class Protocols:
    """Simulated experiments on an LVE model. All methods are static."""

    @staticmethod
    def saos(model: LVEModel, omega: np.ndarray, **kwargs: Any) -> _measurement.Measurement:
        """Small-amplitude oscillatory shear. *omega* in rad/s."""
        return model.to_measurement("saos", omega, **kwargs)

    @staticmethod
    def creep(model: LVEModel, t: np.ndarray, **kwargs: Any) -> _measurement.Measurement:
        """Creep compliance. *t* in s."""
        return model.to_measurement("creep", t, **kwargs)

    @staticmethod
    def stress_relaxation(
        model: LVEModel, t: np.ndarray, **kwargs: Any
    ) -> _measurement.Measurement:
        """Stress relaxation modulus. *t* in s."""
        return model.to_measurement("stress_relaxation", t, **kwargs)

    @staticmethod
    def experiment_set(
        model: LVEModel,
        protocols: Sequence[dict[str, Any]],
    ) -> _experiment_set.ExperimentSet | _measurement_set.MeasurementSet:
        """Run several protocols on one model.

        Parameters
        ----------
        model : LVEModel
        protocols : sequence of dict
            Each dict has ``"protocol"`` (``"saos"``, ``"creep"`` or
            ``"stress_relaxation"``) and ``"x"`` (the grid), plus optional
            ``"conditions"`` and ``"details"``.

        Returns
        -------
        ExperimentSet or MeasurementSet
            An ``ExperimentSet`` when piblin can group the measurements,
            otherwise a ``MeasurementSet``.
        """
        if not protocols:
            raise ValueError("protocols must contain at least one entry")
        measurements = []
        for i, entry in enumerate(protocols):
            if "protocol" not in entry or "x" not in entry:
                raise ValueError(f"protocols[{i}] needs 'protocol' and 'x', got {entry!r}")
            measurements.append(model.to_measurement(**entry))
        try:
            return _experiment_set.ExperimentSet(measurements=measurements, merge_redundant=False)
        except Exception:
            return _measurement_set.MeasurementSet(measurements=measurements)
