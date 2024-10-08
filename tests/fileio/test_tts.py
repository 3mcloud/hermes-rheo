from cralds.dataio.fileio.read.specific.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve


def test_frequency_sweep_tts(datadir):
    filepath = datadir.join("test_frequency_sweep_tts_ARES.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline_storage_modulus = RheoAnalysis() + AutomatedMasterCurve(state='temperature',
                                                                     y='storage modulus',
                                                                     vertical_shift=False,
                                                                     reverse_data=False,
                                                                     measurements=None)

    master_curve = pipeline_storage_modulus(data)
    assert len(master_curve.hparams[0]) == 20


def test_owchirp_tts(datadir):
    filepath = datadir.join("test_owchirp_tts_ARES.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline_storage_modulus = RheoAnalysis() + AutomatedMasterCurve(state='temperature',
                                                                     y='storage modulus',
                                                                     vertical_shift=False,
                                                                     reverse_data=False,
                                                                     measurements=None)


    master_curve = pipeline_storage_modulus(data)
    assert len(master_curve.hparams[0]) == 10
