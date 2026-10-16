import numpy as np
from piblin.data import ExperimentSet, OneDimensionalCompositeDataset

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.temperature_ramp_analysis import (
    TemperatureRampAnalysis,
    TemperatureRampMeasurement,
    TemperatureRampResult,
)

_USE_TEMPS = [0.0, 50.0, 100.0, 150.0]


def _run_analysis(datadir, **kwargs):
    measurements = TriosRheoReader().data_from_filepath(
        filepath=datadir.join("test_aresg2_trios_temp_ramp.txt"),
        create_composite_datasets=True,
    )
    analysis = TemperatureRampAnalysis(use_temps=kwargs.pop("use_temps", _USE_TEMPS), **kwargs)
    result = (OWChirpAnalysis() + analysis)(ExperimentSet(measurements.measurements))
    return analysis, result


def test_temperature_ramp_analysis_returns_temperature_series(datadir):
    """Return sorted data, Tg, and extracted temperatures."""
    analysis, measurement = _run_analysis(datadir, steps_to_merge=["Temperature Ramp"])
    result = analysis.result
    temperature_index = result.dataset.data_array_names.index("temperature")
    temperatures = np.asarray(result.dataset.data_arrays[temperature_index], dtype=float)

    assert isinstance(result, TemperatureRampResult)
    assert isinstance(result.dataset, OneDimensionalCompositeDataset)
    assert np.all(np.diff(temperatures) >= 0)
    assert temperatures.min() <= result.tg <= temperatures.max()
    assert len(result.extracted_dataset.data_arrays[0]) == len(_USE_TEMPS)
    assert isinstance(measurement, TemperatureRampMeasurement)
    assert measurement.datasets == [result.dataset, result.extracted_dataset]


def test_temperature_ramp_analysis_selects_requested_properties(datadir):
    """Extract only the requested temperature-series properties."""
    analysis, _ = _run_analysis(datadir, steps_to_merge=["Temperature Ramp"])
    names = {name.lower(): name for name in analysis.result.dataset.data_array_names}
    properties = [names["storage modulus"], names["loss modulus"]]

    analysis, _ = _run_analysis(
        datadir,
        steps_to_merge=["Temperature Ramp"],
        properties=properties,
    )

    assert analysis.result.extracted_dataset.data_array_names == [
        names["temperature"],
        *properties,
    ]


def test_temperature_ramp_analysis_handles_missing_steps(datadir):
    """Return empty results when no configured steps exist."""
    analysis, _ = _run_analysis(datadir, steps_to_merge=["missing"], use_temps=[25.0])

    assert analysis.result.tg is None
    assert analysis.result.dataset is None
    assert analysis.result.extracted_dataset is None
