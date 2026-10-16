File Readers Overview
=====================

The :mod:`hermes_rheo.file_readers` subpackage provides
instrument-specific readers that convert exported instrument files into
:class:`~piblin.data.data_collections.measurement_set.MeasurementSet`
objects.  See :doc:`file_readers` for the full API reference.

Currently supported instruments and formats:

+---------------------------------+----------------------------------------+
| Instrument / software           | Reader                                 |
+=================================+========================================+
| TA Instruments TRIOS (``.txt``, | :class:`~hermes_rheo.file_readers.     |
| via *Export to LIMS*)           | trios_rheo_txt.TriosRheoReader`        |
+---------------------------------+----------------------------------------+
| Anton Paar RheoCompass          | :class:`~hermes_rheo.file_readers.     |
| (``.csv``)                      | rheo_compass_csv.RheoCompassReader`    |
+---------------------------------+----------------------------------------+

.. important::

   **TRIOS ``.txt`` export path.** ``TriosRheoReader`` needs the full
   step-header metadata that only two TRIOS export paths write:

   * TRIOS ≤5 → *Export* → *Export to LIMS*.
   * TRIOS 6.0+ → *Export* → *Export to .txt (Full)*.

   The plain *Export to .txt* option strips the geometry, calibration
   and wave parameters that ``hermes-rheo`` needs and is unsupported.
   TRIOS is a free download from
   `TA Instruments <https://www.tainstruments.com/support/software-downloads-support/downloads/>`_.
   RheoCompass exports are produced by
   `Anton Paar <https://www.anton-paar.com/us-en/>`_ instrument
   software.

For TRIOS V5 ``.json`` exports a reader is planned but not yet released;
see the project issue tracker on GitHub.

For custom formats, users can develop their own reader while still using
the transforms in :mod:`hermes_rheo.transforms`.  See the *Writing a new
reader* section in :doc:`file_readers`.
