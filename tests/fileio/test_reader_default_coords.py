import pytest
from piblin.data import ExperimentSet

from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader


@pytest.mark.parametrize(
    ("filename", "measurement_index", "expected_coordinates"),
    [
        ("test_aresg2_trios_creep.txt", 1, ("time", "strain")),
        ("test_aresg2_trios_temp_ramp.txt", 0, ("temperature", "storage modulus")),
        (
            "test_aresg2_trios_amplitude_sweep.txt",
            0,
            ("oscillation strain", "storage modulus"),
        ),
        (
            "test_aresg2_trios_frequency_sweep_tts.txt",
            0,
            ("angular frequency", "storage modulus"),
        ),
        ("test_dhr30_trios_owchirp.txt", 0, ("step time", "strain")),
    ],
)
def test_trios_reader_uses_natural_default_coordinates(
    datadir, filename, measurement_index, expected_coordinates
):
    """Use method-appropriate coordinates for TRIOS data."""
    measurements = TriosRheoReader().data_from_filepath(
        filepath=datadir.join(filename), create_composite_datasets=True
    )
    dataset = measurements.measurements[measurement_index].datasets[0]

    assert (
        dataset.independent_variable_name,
        dataset.dependent_variable_name,
    ) == expected_coordinates


@pytest.mark.parametrize("default_coords", ["uniform", "legacy-temperature"])
def test_trios_uniform_coordinates_support_mixed_methods(datadir, default_coords):
    """Allow mixed-method data to use uniform coordinates."""
    measurements = TriosRheoReader().data_from_filepath(
        filepath=datadir.join("test_aresg2_trios_creep.txt"),
        create_composite_datasets=True,
        default_coords=default_coords,
    )

    assert ExperimentSet(measurements.measurements).num_experiments == 4
    assert all(
        dataset.independent_variable_name == dataset.dependent_variable_name == "Temperature"
        for measurement in measurements.measurements
        for dataset in measurement.datasets
    )


@pytest.mark.parametrize(
    ("filename", "expected_coordinates"),
    [
        (
            "test_mcr702e_rheocompass_frequency_sweep_tts.csv",
            ("angular frequency", "storage modulus"),
        ),
        ("test_mcr702e_rheocompass_stress_relaxation.csv", ("time", "shear stress")),
    ],
)
def test_rheocompass_reader_uses_natural_default_coordinates(
    datadir, filename, expected_coordinates
):
    """Use method-appropriate coordinates for RheoCompass data."""
    measurements = RheoCompassReader().data_from_filepath(str(datadir.join(filename)))
    dataset = measurements.measurements[0].datasets[0]

    assert (
        dataset.independent_variable_name,
        dataset.dependent_variable_name,
    ) == expected_coordinates
