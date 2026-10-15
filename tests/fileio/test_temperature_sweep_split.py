import numpy as np
import pytest
from piblin.data import ExperimentSet
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep


# ---------------------------------------------------------------------------
# Single-step fixture (uniform 5 °C steps, -40 → 50 °C)
# ---------------------------------------------------------------------------

def test_temperature_sweep_split(datadir):
    """Existing single-step sweep: auto-detect still produces 19 freq-sweeps."""
    filepath = datadir.join("test_temperature_sweep_RSAG2.txt")
    experiment = ExperimentSet(
        TriosRheoReader().data_from_filepath(filepath=filepath, create_composite_datasets=True).measurements
    )
    pipeline = OWChirpAnalysis() + SplitTemperatureSweep()
    processed_experiment = pipeline(experiment)
    assert len(processed_experiment.measurements) == 19
    assert len(processed_experiment.measurements[0].datasets) == 1


def test_temperature_sweep_split_explicit_n_steps(datadir):
    """Explicit n_steps=1 with auto-detected interval gives the same result."""
    filepath = datadir.join("test_temperature_sweep_RSAG2.txt")
    experiment = ExperimentSet(
        TriosRheoReader().data_from_filepath(filepath=filepath, create_composite_datasets=True).measurements
    )
    pipeline = OWChirpAnalysis() + SplitTemperatureSweep(n_steps=1)
    processed_experiment = pipeline(experiment)
    assert len(processed_experiment.measurements) == 19


def test_temperature_sweep_split_wrong_n_steps_raises(datadir):
    """Passing a wrong n_steps raises ValueError."""
    filepath = datadir.join("test_temperature_sweep_RSAG2.txt")
    experiment = ExperimentSet(
        TriosRheoReader().data_from_filepath(filepath=filepath, create_composite_datasets=True).measurements
    )
    pipeline = OWChirpAnalysis() + SplitTemperatureSweep(n_steps=3)
    with pytest.raises(ValueError, match="Expected 3 measurements, got 1"):
        pipeline(experiment)


# ---------------------------------------------------------------------------
# Multi-step fixture (DF-17A: three sweeps, step sizes 5 / 2.5 / 5 °C)
# ---------------------------------------------------------------------------

def _load_multi_step(datadir):
    filepath = datadir.join("DF-17A_multi_step_temp_sweep.txt")
    experiment = ExperimentSet(
        TriosRheoReader().data_from_filepath(filepath=filepath, create_composite_datasets=True).measurements
    )
    return experiment


def test_multi_step_auto_detect_count(datadir):
    """Three-sweep file: auto-detect produces 31 individual freq-sweeps."""
    experiment = _load_multi_step(datadir)
    processed = (OWChirpAnalysis() + SplitTemperatureSweep())(experiment)
    # Sweep 1 (-40→20 °C, 5 °C steps)  → 13 temperatures
    # Sweep 2 (22.5→50 °C, 2.5 °C steps) → 12 temperatures
    # Sweep 3 (55→80 °C, 5 °C steps)    →  6 temperatures
    assert len(processed.measurements) == 31


def test_multi_step_explicit_n_steps(datadir):
    """n_steps=3 explicit gives the same 31 freq-sweeps as auto-detection."""
    experiment = _load_multi_step(datadir)
    processed = (OWChirpAnalysis() + SplitTemperatureSweep(n_steps=3))(experiment)
    assert len(processed.measurements) == 31


def test_multi_step_nominal_temperatures_no_spurious_merging(datadir):
    """Each frequency sweep should contain exactly 6 data points (no merging of
    adjacent 2.5 °C temperatures into a 5 °C bin)."""
    experiment = _load_multi_step(datadir)
    processed = (OWChirpAnalysis() + SplitTemperatureSweep())(experiment)
    for meas in processed.measurements:
        ds = meas.datasets[0]
        n_points = len(ds.data_arrays[0])
        assert n_points == 6, (
            f"Expected 6 freq points per sweep, got {n_points} "
            f"for {meas.conditions}"
        )


def test_multi_step_state_values_near_nominal(datadir):
    """State (average temperature) of each sweep should be close to its nominal
    value, confirming temperatures are not spuriously merged across sweeps."""
    from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve

    experiment = _load_multi_step(datadir)
    processed = (OWChirpAnalysis() + SplitTemperatureSweep())(experiment)
    mc = AutomatedMasterCurve()(processed)
    states = np.array(mc.states, dtype=float)
    # Nominal temperatures (°C)
    nominal = np.array(
        [-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20,   # sweep 1
         22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50,  # sweep 2
         55, 60, 65, 70, 75, 80],                                        # sweep 3
        dtype=float,
    )
    assert len(states) == len(nominal)
    # Each state should be within 1 °C of its nominal temperature
    assert np.all(np.abs(np.sort(states) - np.sort(nominal)) < 1.0), (
        f"States deviate from nominal by more than 1 °C:\n"
        f"sorted states:   {np.sort(states)}\n"
        f"sorted nominal:  {np.sort(nominal)}"
    )


# ---------------------------------------------------------------------------
# Unit tests for _detect_temp_step
# ---------------------------------------------------------------------------

def test_detect_temp_step_5deg(datadir):
    """Single-step fixture: auto-detected step should be 5.0 °C."""
    filepath = datadir.join("test_temperature_sweep_RSAG2.txt")
    experiment = ExperimentSet(
        TriosRheoReader().data_from_filepath(filepath=filepath, create_composite_datasets=True).measurements
    )
    from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep
    processed = OWChirpAnalysis()(experiment)
    ds = processed.measurements[0].datasets[0]
    step = SplitTemperatureSweep._detect_temp_step(ds, temp_tolerance=1.0)
    assert step == pytest.approx(5.0, abs=0.5)


def test_detect_temp_step_25deg(datadir):
    """Sweep 2 of the multi-step fixture: auto-detected step should be 2.5 °C."""
    experiment = _load_multi_step(datadir)
    processed = OWChirpAnalysis()(experiment)
    # Second measurement = sweep 2 (2.5 °C steps)
    ds = processed.measurements[1].datasets[0]
    step = SplitTemperatureSweep._detect_temp_step(ds, temp_tolerance=1.0)
    assert step == pytest.approx(2.5, abs=0.5)

