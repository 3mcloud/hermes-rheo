from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader


def test_trios_lims_export_file_single_step(datadir):
    filepath = datadir.join("TRIOS_LIMS_export_single_step.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 3
    assert len(data.measurements[0].datasets) == 27


def test_trios_lims_export_file_multi_step(datadir):
    filepath = datadir.join("TRIOS_LIMS_export_multiple_step.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 8
    assert len(data.measurements[0].datasets) == 27

def test_DMA(datadir):
    filepath = datadir.join("468mp compression owchirp - freq sweep_strain 1%_20230221.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 6
    assert len(data.measurements[0].datasets) == 10
