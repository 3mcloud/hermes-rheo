from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep


def test_frequency_sweep_tts(datadir):
    filepath = datadir.join("test_frequency_sweep_tts_ARES.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline_storage_modulus = OWChirpAnalysis() + AutomatedMasterCurve(
        state="temperature",
        y="storage modulus",
        vertical_shift=False,
        reverse_data=False,
        measurements=None,
    )

    master_curve = pipeline_storage_modulus(data)
    assert len(master_curve.hparams[0]) == 20


def test_owchirp_tts(datadir):
    filepath = datadir.join("test_owchirp_tts_ARES.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline_storage_modulus = OWChirpAnalysis() + AutomatedMasterCurve(
        state="temperature",
        y="storage modulus",
        vertical_shift=False,
        reverse_data=False,
        measurements=None,
    )

    master_curve = pipeline_storage_modulus(data)
    assert len(master_curve.hparams[0]) == 10


def test_temp_sweep_tts_temp_not_sorted(datadir):
    filepath = datadir.join("test_temp_sweep_tts.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline_complex_modulus = (
        OWChirpAnalysis()
        + SplitTemperatureSweep(n_steps=5)
        + AutomatedMasterCurve(
            state="Temperature",
            state_mode="average",
            y="complex modulus",
            vertical_shift=False,
            reverse_data=False,
            sort_states=True,
        )
    )

    master_curve = pipeline_complex_modulus(data)
    assert len(master_curve.hparams[0]) == 26
