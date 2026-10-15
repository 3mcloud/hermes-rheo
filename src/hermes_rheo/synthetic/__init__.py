"""Synthetic linear-viscoelastic data for hermes-rheo pipelines.

This subpackage generates piblin
:class:`~piblin.data.data_collections.measurement.Measurement` and
:class:`~piblin.data.data_collections.experiment_set.ExperimentSet`
objects from classical linear viscoelastic (LVE) constitutive equations
(Maxwell, multi-mode Maxwell, Kelvin--Voigt, Jeffreys).  The generated
measurements plug directly into every hermes-rheo transform pipeline
without any special-casing, so the same code paths that run on
experimental data can be validated against analytical ground truth.

The API is intentionally two-tier:

* :class:`~hermes_rheo.synthetic.factory.SyntheticRheometry` --- factory
  class with one method per constitutive model (:meth:`.maxwell`,
  :meth:`.multi_mode_maxwell`, :meth:`.kelvin_voigt`, :meth:`.jeffreys`),
  each returning a :class:`RheologyModel` instance.
* :class:`~hermes_rheo.synthetic.models.RheologyModel` --- lightweight
  model object exposing :meth:`.stress_relaxation`,
  :meth:`.frequency_sweep` and :meth:`.creep` for direct array output,
  plus :meth:`.to_measurement` for building a piblin
  ``Measurement`` with the correct column names, units, conditions and
  details.

Future work
-----------
The base class is designed to accommodate fractional LVE kernels
(Mittag--Leffler relaxation) and nonlinear differential constitutive
equations (Giesekus, PTT, FENE-P).  Those are intentionally out of
scope for the initial release but referenced in the class docstrings
as future subclasses.

References
----------
.. [Ferry1980] Ferry, J. D., *Viscoelastic Properties of Polymers*, 3rd
   ed., John Wiley & Sons (1980).  ISBN 978-0471048947.  The
   canonical reference for classical LVE constitutive equations.
.. [Bird1987] Bird, R. B., Armstrong, R. C. and Hassager, O., *Dynamics
   of Polymeric Liquids, Volume 1: Fluid Mechanics*, 2nd ed., Wiley
   (1987).  ISBN 978-0471802457.

Examples
--------
>>> import numpy as np
>>> from hermes_rheo.synthetic import SyntheticRheometry
>>> model = SyntheticRheometry.multi_mode_maxwell(g=[1e5, 1e4], tau=[0.1, 1.0])
>>> t = np.geomspace(1e-3, 1e2, 200)
>>> meas = model.to_measurement(experiment="stress_relaxation", x=t)
>>> meas.datasets[0].data_array_names
['time', 'relaxation modulus']
"""
from __future__ import annotations

from hermes_rheo.synthetic.factory import SyntheticRheometry
from hermes_rheo.synthetic.models import (
    JeffreysModel,
    KelvinVoigtModel,
    MaxwellModel,
    MultiModeMaxwellModel,
    RheologyModel,
)

__all__ = [
    "SyntheticRheometry",
    "RheologyModel",
    "MaxwellModel",
    "MultiModeMaxwellModel",
    "KelvinVoigtModel",
    "JeffreysModel",
]
