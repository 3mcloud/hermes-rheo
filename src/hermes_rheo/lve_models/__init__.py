"""Simulated linear viscoelastic experiments.

Run SAOS (small-amplitude oscillatory shear), creep and stress relaxation on
classical and fractional models. Each experiment returns a piblin
:class:`~piblin.data.data_collections.measurement.Measurement` with one
composite one-dimensional dataset and metadata describing the model, its
parameters and the simulated protocol, so it passes through the same
hermes-rheo transforms as measured data.

Models
------
Classical: :class:`MaxwellModel`, :class:`MultiModeMaxwellModel`,
:class:`KelvinVoigtModel`, :class:`JeffreysModel`.

Fractional: :class:`SpringPotModel`, :class:`FractionalMaxwellGelModel`,
:class:`FractionalMaxwellLiquidModel`, :class:`FractionalMaxwellModel`,
:class:`FractionalKelvinVoigtModel`, :class:`FractionalZenerModel`.

Examples
--------
>>> import numpy as np
>>> from hermes_rheo.lve_models import MultiModeMaxwellModel, Protocols
>>> model = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
>>> meas = Protocols.stress_relaxation(model, np.geomspace(1e-3, 1e2, 200))
>>> meas.datasets[0].data_array_names
['time', 'relaxation modulus']
>>> meas.conditions["simulated"], meas.details["protocol"]
(True, 'stress_relaxation')
"""

from __future__ import annotations

from hermes_rheo.lve_models.base import LVEModel, invert_laplace
from hermes_rheo.lve_models.classical import (
    JeffreysModel,
    KelvinVoigtModel,
    MaxwellModel,
    MultiModeMaxwellModel,
)
from hermes_rheo.lve_models.fractional import (
    FractionalKelvinVoigtModel,
    FractionalMaxwellGelModel,
    FractionalMaxwellLiquidModel,
    FractionalMaxwellModel,
    FractionalZenerModel,
    SpringPotModel,
)
from hermes_rheo.lve_models.protocols import Protocols

__all__ = [
    "Protocols",
    "LVEModel",
    "invert_laplace",
    "MaxwellModel",
    "MultiModeMaxwellModel",
    "KelvinVoigtModel",
    "JeffreysModel",
    "SpringPotModel",
    "FractionalMaxwellGelModel",
    "FractionalMaxwellLiquidModel",
    "FractionalMaxwellModel",
    "FractionalKelvinVoigtModel",
    "FractionalZenerModel",
]
