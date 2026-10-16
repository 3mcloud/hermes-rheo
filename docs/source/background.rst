Background
==========

``hermes-rheo`` builds on `piblin <https://github.com/3mcloud/piblin>`_,
which provides the generic data model (``Measurement``,
``MeasurementSet``, ``ExperimentSet``, ``OneDimensionalCompositeDataset``)
together with the ``FileReader`` and transform base classes.
``hermes-rheo`` adds instrument-specific file readers for TA Instruments
TRIOS and Anton Paar RheoCompass exports, and rheology-specific transforms.

The package supports:

- Reading TA Instruments TRIOS ``.txt`` exports
  (see `TA Instruments <https://www.tainstruments.com/>`_) and
  Anton Paar RheoCompass ``.csv`` exports
  (see `Anton Paar <https://www.anton-paar.com/us-en/>`_).
- Analysing chirp-based oscillatory tests via the
  `Optimal Windowed Chirp (OWChirp) <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.041042>`_
  technique (Geri *et al.* 2018), including the strain-controlled
  variant on TA ARES-G2 rheometers described in Perego (2025,
  `Rheologica Acta <https://doi.org/10.1007/s00397-025-01511-0>`_).
- Data-driven time–temperature superposition via the
  `automated master-curve algorithm <https://www.cambridge.org/core/journals/data-centric-engineering/article/datadriven-method-for-automated-data-superposition-with-applications-in-soft-matter-science/DA44C868EE1128DD79798653A1376594>`_
  of Lennon *et al.* (2023).
- Time–cure superposition, temperature-ramp analysis, mutation number,
  recovery rheology, and 2-D kymograph construction from TORCh
  sequences (Vadillo, Perego *et al.* 2026).


Citing
------

If you use this software, please cite the JOSS paper:

    Perego *et al.* (2025), *hermes-rheo: A Python package for
    rheological data analysis*, Journal of Open Source Software,
    `10.21105/joss.08753 <https://joss.theoj.org/papers/10.21105/joss.08753>`_.

The Zenodo release DOI is also available:

    .. image:: https://zenodo.org/badge/851879885.svg
       :target: https://doi.org/10.5281/zenodo.14182224
       :alt: DOI

Installation
------------

``hermes-rheo`` is available on PyPI:

.. code-block:: console

   $ pip install hermes-rheo

You can also use pip to download updates:

.. code-block:: console

   $ pip install hermes-rheo --upgrade

Contributing
------------

Questions, comments, or suggestions can be raised as issues on `GitHub <https://github.com/3mcloud/hermes-rheo>`_
or emailed directly to aperego[at]mmm.com.