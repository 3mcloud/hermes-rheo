import pytest

pytest.importorskip("pyRheo")

from piblin.data import ExperimentSet  # noqa: E402
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve, CompositeMasterCurve
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.fit_model import FitModel
from pyRheo.oscillation_model import SAOSModel  # noqa: E402
import warnings

warnings.filterwarnings("ignore")


def test_frequency_sweep_tts(datadir):
    filepath = datadir.join("test_fitting.txt")
    data = ExperimentSet(
        TriosRheoReader()
        .data_from_filepath(filepath=filepath, create_composite_datasets=True)
        .measurements
    )
    measurements_range = tuple(range(0, 39))  # Pipeline will be applied to the following meas.
    pipeline = (
        OWChirpAnalysis()
        + SplitTemperatureSweep()
        + CompositeMasterCurve(
            n_curves=2,
            quantities=["storage modulus", "loss modulus"],
            state="temperature",
            state_mode="average",
            vertical_shift=False,
            reverse_data=False,
            measurements=measurements_range,
            ref_temp_idx=19,  # index of reference temperature (Tref)
        )
        + FitModel(
            model=SAOSModel,
            model_name="FractionalZener",
            initial_guesses="random",
            cost_function="BIC",
            bounds="auto",
            num_initial_guesses=500,
            minimization_algorithm="Powell",
        )
    )

    result = pipeline(data)
    assert (
        isinstance(result.fitted_model, SAOSModel)
        and getattr(result.fitted_model, "model", None) == "FractionalZener"
    )
