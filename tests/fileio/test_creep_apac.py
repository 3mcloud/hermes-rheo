from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def test_creep_apac(datadir):
    """OWChirpAnalysis processes a mixed Stress-Control + Creep file end-to-end.

    The four measurements in the file (one Stress-Control equilibration
    step, three Creep steps) use different natural (independent, dependent)
    coordinate pairs after the reader-defaults change in 2.0.  This
    means the file can no longer be wrapped directly in an
    :class:`~piblin.data.ExperimentSet` (which enforces consistency
    across measurements), so the test now consumes the underlying
    :class:`~piblin.data.MeasurementSet` returned by the reader.
    Callers who need ``ExperimentSet`` semantics for mixed-method files
    should pass ``default_coords="legacy-temperature"`` to the reader.
    """
    filepath = datadir.join("test_creep_apac.txt")
    mset = TriosRheoReader().data_from_filepath(filepath=filepath, create_composite_datasets=True)

    processed = OWChirpAnalysis()(mset)

    assert len(processed.measurements) == 4

    assert processed.measurements[0].conditions["method"] == "Stress Control - 1"
    assert processed.measurements[1].conditions["method"] == "Creep - 2"
    assert processed.measurements[2].conditions["method"] == "Creep - 3"
    assert processed.measurements[3].conditions["method"] == "Creep - 4"
