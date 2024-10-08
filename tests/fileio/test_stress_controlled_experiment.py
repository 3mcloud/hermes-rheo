from cralds.dataio.fileio.read.specific.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis

def test_DHR_serial(datadir):
    filepath = datadir.join("test_owchirp_DHR.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline = RheoAnalysis()

    processed_data = pipeline(data)
    assert processed_data.details['instrument_serial_number'][:4] == '5343'