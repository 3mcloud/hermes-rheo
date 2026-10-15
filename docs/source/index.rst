hermes-rheo
===========

`hermes-rheo <https://github.com/3mcloud/hermes-rheo>`_ is a Python
package developed at
`3M <https://www.3m.com/3M/en_US/company-us/about-3m/research-development/>`_
that layers rheology-specific file readers and transforms on top of the
generic data abstractions of
`piblin <https://github.com/3mcloud/piblin>`_.

Highlights
----------

* **File readers** for
  `TA Instruments <https://www.tainstruments.com/>`_ TRIOS ``.txt``
  (via *Export to LIMS* on TRIOS ≤5 or *Export to .txt (Full)* on
  TRIOS 6.0+; the plain *Export to .txt* option omits step metadata
  and is unsupported),
  and `Anton Paar <https://www.anton-paar.com/us-en/>`_ RheoCompass
  ``.csv``.
* **Reader-driven per-method default coordinates**: every measurement
  arrives with its scientifically-natural ``(independent, dependent)``
  pair (e.g. ``strain=f(time)`` for Creep, ``storage modulus=f(angular
  frequency)`` for a Frequency Sweep).
* **OWChirp analysis** — time-resolved viscoelastic-moduli extraction
  from Optimally Windowed Chirp signals (Geri *et al.* 2018,
  DOI `10.1103/PhysRevX.8.041042
  <https://doi.org/10.1103/PhysRevX.8.041042>`_).
* **Automated master curves** using the data-driven Gaussian-process
  method of Lennon, McKinley and Swan (2023,
  DOI `10.1017/dce.2023.3 <https://doi.org/10.1017/dce.2023.3>`_).
* **Recovery rheology**, **mutation number**, **temperature-ramp
  analysis**, **model fitting**, **kymographs** and a variety of
  utility transforms.
* Composable pipelines via the ``+`` operator inherited from
  :mod:`piblin`.

.. note::

   This project is under active development.  See :doc:`quickstart` to
   get started or :doc:`transforms` for the full API.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   background
   readers
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   file_readers
   transforms
   synthetic
   visualization
