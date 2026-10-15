from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader


def test_rheocompass_frequency_sweep_tts(datadir):
    filepath = datadir.join("test_antonpaar_702e_tts.csv")
    data = RheoCompassReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 33


def test_rheocompass_stress_relaxation(datadir):
    filepath = datadir.join("test_antonpaar_htr_stress_relax.csv")
    data = RheoCompassReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 1
