import pytest

pytest.importorskip("pyRheo")

from piblin.data import ExperimentSet
from piblin.data.data_collections.measurement import Measurement
from pyRheo.oscillation_model import SAOSModel

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.automated_mastercurve import CompositeMasterCurve
from hermes_rheo.transforms.fit_model import FitModel
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep


def test_frequency_sweep_model_fit(datadir):
    """Fit a Fractional Zener model to a temperature sweep."""
    measurements = TriosRheoReader().data_from_filepath(
        filepath=datadir.join("test_rsag2_trios_temp_sweep_fitting.txt"),
        create_composite_datasets=True,
    )
    data = ExperimentSet(measurements.measurements)
    fit_model = FitModel(
        model=SAOSModel,
        model_name="FractionalZener",
        initial_guesses="random",
        cost_function="BIC",
        bounds="auto",
        num_initial_guesses=500,
        minimization_algorithm="Powell",
    )
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
            measurements=tuple(range(39)),
            ref_temp_idx=19,
        )
        + fit_model
    )

    result = pipeline(data)

    assert isinstance(result, Measurement)
    assert isinstance(fit_model.fitted_model, SAOSModel)
    assert fit_model.fitted_model.model == "FractionalZener"
    assert result.details["model"] == "FractionalZener"
    assert result.details["goodness_of_fit"]["cost_function"] == "BIC"
    assert "fitted storage modulus" in result.datasets[0].data_array_names
