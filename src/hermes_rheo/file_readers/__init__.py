"""File readers for rheological instrument data.

Instrument-specific readers convert exported data files into a
:class:`~piblin.data.data_collections.measurement_set.MeasurementSet` of
:class:`~piblin.data.data_collections.measurement.Measurement` objects,
each carrying a
:class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
with per-method default coordinates (see each reader's
``_METHOD_DEFAULTS`` / ``_DEFAULT_COORD_RULES``).

Available readers
-----------------
:class:`~hermes_rheo.file_readers.trios_rheo_txt.TriosRheoReader`
    TA Instruments TRIOS ``.txt`` export: use *Export to LIMS*
    (TRIOS <= 5) or *Export to .txt (Full)* (TRIOS >= 6.0).  The plain
    *Export to .txt* option strips step metadata and will not parse.
:class:`~hermes_rheo.file_readers.rheo_compass_csv.RheoCompassReader`
    Anton Paar RheoCompass ``.csv``.

:func:`~hermes_rheo.file_readers.read.read_file` reads a file with a given
reader and returns the same collection types as ``piblin.dataio.read_file``.
"""

from __future__ import annotations

from hermes_rheo.file_readers.read import read_file
from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader

__all__ = [
    "TriosRheoReader",
    "RheoCompassReader",
    "read_file",
]
