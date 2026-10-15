Synthetic constitutive-model data
=================================

The :mod:`hermes_rheo.synthetic` subpackage provides analytical
synthetic-data generators for the classical linear-viscoelastic
constitutive models.  The generated
:class:`~piblin.data.data_collections.measurement.Measurement`
objects carry the same column names, units and
``conditions["method"]`` / ``details`` metadata as real TRIOS / OWChirp
files, so they compose directly with every hermes-rheo transform.

.. note::

   Synthetic measurements are marked with ``conditions["synthetic"] =
   True`` and carry both ``details["model"]`` (short label) and
   ``details["parameters"]`` (constitutive parameter snapshot).  This
   lets downstream code distinguish them from real data.

Overview
--------

* :class:`~hermes_rheo.synthetic.SyntheticRheometry` --- factory with
  one ``@classmethod`` per constitutive model.
* :class:`~hermes_rheo.synthetic.models.MaxwellModel` --- single
  spring + dashpot in series (fluid).
* :class:`~hermes_rheo.synthetic.models.MultiModeMaxwellModel` ---
  generalized Maxwell / Prony series with optional
  :math:`G_\infty`.
* :class:`~hermes_rheo.synthetic.models.KelvinVoigtModel` ---
  spring + dashpot in parallel (viscoelastic solid).
* :class:`~hermes_rheo.synthetic.models.JeffreysModel` ---
  Maxwell branch + purely viscous solvent (Jeffreys / Oldroyd model).

Each model exposes three canonical experiments as pure NumPy functions:

* ``stress_relaxation(t) -> G(t)`` in Pa
* ``frequency_sweep(omega) -> (G'(omega), G''(omega))`` in Pa
* ``creep(t) -> J(t)`` in 1/Pa

plus a convenience ``to_measurement(experiment, x)`` that wraps the
result in a piblin ``Measurement``.

Quickstart
----------

.. code-block:: python

   import numpy as np
   from hermes_rheo.synthetic import SyntheticRheometry

   # Two-mode Maxwell fluid
   model = SyntheticRheometry.multi_mode_maxwell(
       g=[1e5, 1e4],
       tau=[0.1, 1.0],
   )

   # Frequency sweep on a log grid
   omega = np.geomspace(1e-2, 1e3, 60)
   meas = model.to_measurement(experiment="frequency_sweep", x=omega)

   # Compose several experiments into a single ExperimentSet
   exp = SyntheticRheometry.experiment_set(
       model,
       experiments=[
           {"experiment": "stress_relaxation", "x": np.geomspace(1e-3, 1e2, 200)},
           {"experiment": "frequency_sweep",   "x": omega},
       ],
   )

Downstream, the produced measurements can be fed to any
hermes-rheo transform (e.g. ``RelaxationSpectrum``, ``FitModel``,
``FrequencyToTime``) exactly like real experimental data.

API reference
-------------

Factory
~~~~~~~

.. automodule:: hermes_rheo.synthetic.factory
   :members: SyntheticRheometry
   :undoc-members:
   :show-inheritance:

Constitutive models
~~~~~~~~~~~~~~~~~~~

.. automodule:: hermes_rheo.synthetic.models
   :members:
       RheologyModel,
       MaxwellModel,
       MultiModeMaxwellModel,
       KelvinVoigtModel,
       JeffreysModel
   :undoc-members:
   :show-inheritance:

References
----------

The classical LVE models implemented here are covered in every rheology
textbook; the reference set below is a minimal one.

* Ferry, J. D., *Viscoelastic Properties of Polymers*, 3rd ed., John
  Wiley & Sons (1980). ISBN 978-0471048947.
* Bird, R. B., Armstrong, R. C. and Hassager, O., *Dynamics of
  Polymeric Liquids, Volume 1: Fluid Mechanics*, 2nd ed., Wiley (1987).
  ISBN 978-0471802457.
* Jeffreys, H., *The Earth: Its Origin, History and Physical
  Constitution*, 2nd ed., Cambridge University Press (1929).
