Quickstart
==========

.. _installation:

Installation
------------

``hermes-rheo`` requires Python 3.10–3.13 and is installed from PyPI:

.. code-block:: console

   pip install hermes-rheo

or with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

   uv add hermes-rheo            # add to a project
   uv pip install hermes-rheo    # install into an existing environment

Optional extras:

.. code-block:: console

   pip install "hermes-rheo[notebooks]"   # SciencePlots and OpenCV for the tutorials
   uv add "hermes-rheo[notebooks]"        # same extras with uv

Model fitting with pyRheo models requires pyRheo, which is not on PyPI:

.. code-block:: console

   pip install git+https://github.com/mirandi1/pyRheo.git

.. _importing:

Importing the API
-----------------

All commonly used file readers and transforms are re-exported from the
top-level package:

.. code-block:: python

   from hermes_rheo import (
       # file readers
       TriosRheoReader, RheoCompassReader,

       # analysis transforms
       OWChirpAnalysis,          # chirp / OWChirp FFT moduli extraction
       AutomatedMasterCurve,
       CompositeMasterCurve,
       TemperatureRampAnalysis,
       RecoveryRheology,
       MutationNumber,
       FitModel,
       MultiModeMaxwellModel,
       FrequencyToTime,
       RelaxationSpectrum,

       # signal design helpers
       OWChirpGeneration,

       # utility transforms
       SplitTemperatureSweep,
       StatesToConditions,
       Kymograph,
   )

.. note::

   ``RheoAnalysis`` still works but emits a :class:`DeprecationWarning`.
   Prefer :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`
   for new code.

Reading a file
--------------

.. code-block:: python

   from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
   from piblin.data import ExperimentSet

   mset = TriosRheoReader().data_from_filepath(
       "creep_experiment.txt", create_composite_datasets=True
   )
   print(mset)
   # -> shows e.g. `strain(unitless)=f(time[s]) (483 points)` for each Creep step,
   # instead of the legacy `Temperature(°C)=f(Temperature[°C])`.

   ds = mset.measurements[0].datasets[0]
   ds.independent_variable_name, ds.dependent_variable_name
   # ('time', 'strain')

**Legacy consistency mode.** Mixed-method files (e.g. a Stress-Control
equilibration step followed by several Creep steps) will have
per-measurement coordinates that differ across measurements, so wrapping
the resulting ``MeasurementSet`` directly in an ``ExperimentSet`` may
raise ``InconsistentMeasurementsError``.  Users who need the historical
behaviour can opt in:

.. code-block:: python

   mset = TriosRheoReader().data_from_filepath(
       "mixed.txt",
       create_composite_datasets=True,
       default_coords="uniform",   # all measurements get X=Y="Temperature"
   )
   ExperimentSet(mset.measurements)   # works, but display labels are ugly

Running a chirp-analysis pipeline
---------------------------------

.. code-block:: python

   from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
   from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
   from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve

   data = TriosRheoReader().data_from_filepath(
       "owchirp_tts.txt", create_composite_datasets=True
   )

   pipeline = OWChirpAnalysis() + AutomatedMasterCurve(
       state="temperature",
       y="storage modulus",
       vertical_shift=False,
   )

   master_curve = pipeline(data)

Learn more
----------

* :doc:`background` — background on hermes-rheo and piblin.
* :doc:`file_readers` — full reader API reference.
* :doc:`transforms` — full transform API reference (with paper
  citations).
* :doc:`tutorials` — end-to-end Jupyter notebooks.
