File Readers
============

Every reader in :mod:`hermes_rheo.file_readers` subclasses
:class:`piblin.dataio.fileio.read.file_reader.FileReader` and converts
an exported instrument file into a
:class:`~piblin.data.data_collections.measurement_set.MeasurementSet`.

**Reader-driven default coordinates.**  As of the 2.x → 3.x refactor the
readers install the *natural* per-method default coordinates on each
resulting composite dataset (e.g. ``strain=f(time)`` for a Creep step,
``storage modulus=f(angular frequency)`` for a frequency sweep).  Users
who need the historical consistency-preserving
``Temperature=f(Temperature)`` behaviour can opt in with
``default_coords="legacy-temperature"`` on
:class:`~hermes_rheo.file_readers.trios_rheo_txt.TriosRheoReader`.

TA Instruments TRIOS ``.txt``
-----------------------------
Reads ``.txt`` files produced by
`TA Instruments <https://www.tainstruments.com/>`_ TRIOS.  Use the
*Export to LIMS* command (TRIOS ≤5) or, on TRIOS 6.0+,
*Export to .txt (Full)* — the plain *Export to .txt* option strips the
step metadata this reader needs.  TRIOS is available free of charge
from
`TA Instruments <https://www.tainstruments.com/support/software-downloads-support/downloads/>`_.

.. automodule:: hermes_rheo.file_readers.trios_rheo_txt
   :members: TriosRheoReader
   :undoc-members:
   :show-inheritance:

Anton Paar RheoCompass ``.csv``
-------------------------------
Reads ``.csv`` files exported from
`Anton Paar <https://www.anton-paar.com/us-en/>`_ RheoCompass.
Per-measurement default coordinates are chosen from the column set
(frequency sweep, flow sweep, or transient fallback).

.. automodule:: hermes_rheo.file_readers.rheo_compass_csv
   :members: RheoCompassReader
   :undoc-members:
   :show-inheritance:

Writing a new reader
--------------------
Checklist:

1. Subclass
   :class:`piblin.dataio.fileio.read.file_reader.FileReader` (or
   ``GenericFileReader``).
2. Declare ``supported_extensions``.
3. Implement ``_data_from_file_contents(cls, file_contents, ...)`` returning
   a :class:`~piblin.data.MeasurementSet`.
4. Register per-method default coordinates using the same pattern as
   ``_METHOD_DEFAULTS`` /
   :func:`hermes_rheo.file_readers.trios_rheo_txt._lookup_method_defaults`.
5. Add a fixture in ``tests/fileio/test_data/`` and a test.
6. Export the reader from ``src/hermes_rheo/file_readers/__init__.py``.
