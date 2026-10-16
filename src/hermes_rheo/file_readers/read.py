"""Read a file with an explicitly chosen reader.

``piblin.dataio.read_file`` selects a reader from the file extension and
only considers readers shipped with piblin.  :func:`read_file` lets the
caller pass a hermes-rheo reader directly while returning the same data
collection types as piblin.
"""

from __future__ import annotations

from pathlib import Path

from piblin.data.data_collections.data_collection_factory import DataCollectionFactory
from piblin.dataio.fileio.read.file_reader import FileReader

from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader

_READERS_BY_EXTENSION: dict[str, type[FileReader]] = {
    ".txt": TriosRheoReader,
    ".csv": RheoCompassReader,
}


def read_file(
    filepath: str | Path,
    file_reader: type[FileReader] | FileReader | None = None,
    to_experiments: bool = True,
    minimize_hierarchy: bool = False,
    merge_redundant: bool = True,
    **read_kwargs,
):
    """Read a rheometer export into a piblin data collection.

    Parameters
    ----------
    filepath : str or Path
        Path to the exported file.
    file_reader : FileReader subclass or instance, optional
        Reader to use.  If omitted, it is chosen from the file extension:
        ``.txt`` -> :class:`TriosRheoReader`, ``.csv`` ->
        :class:`RheoCompassReader`.
    to_experiments : bool, default True
        Convert to an ``ExperimentSet`` when the measurements are consistent.
    minimize_hierarchy : bool, default False
        Return the least nested collection possible.
    merge_redundant : bool, default True
        Merge measurements with identical conditions.
    **read_kwargs
        Passed to the reader, e.g. ``create_composite_datasets=True``.

    Returns
    -------
    MeasurementSet or ExperimentSet
        The most structured collection supported by the measurements, as
        returned by ``piblin.dataio.read_file``.

    Examples
    --------
    >>> from hermes_rheo import TriosRheoReader, read_file
    >>> experiment = read_file(  # doctest: +SKIP
    ...     "frequency_sweep_tts.txt",
    ...     file_reader=TriosRheoReader,
    ...     create_composite_datasets=True,
    ... )
    """
    filepath = Path(filepath)
    if file_reader is None:
        try:
            file_reader = _READERS_BY_EXTENSION[filepath.suffix.lower()]
        except KeyError:
            raise ValueError(
                f"No hermes-rheo reader for extension {filepath.suffix!r}; "
                "pass file_reader explicitly."
            ) from None
    reader = file_reader() if isinstance(file_reader, type) else file_reader
    measurement_set = reader.data_from_filepath(str(filepath), **read_kwargs)
    return DataCollectionFactory.from_measurements(
        measurement_set.measurements,
        to_experiments=to_experiments,
        minimize_hierarchy=minimize_hierarchy,
        merge_redundant=merge_redundant,
    )
