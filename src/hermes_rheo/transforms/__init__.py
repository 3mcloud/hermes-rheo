"""Rheological analysis transforms built on :mod:`piblin` base classes.

Available transforms
--------------------

Chirp / OWChirp:
    :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`
        Extract time-resolved moduli from Arbitrary Wave / Frequency Chirp
        steps (Geri et al. 2018).
    :class:`~hermes_rheo.transforms.rheo_analysis.RheoAnalysis`
        *Deprecated* alias for :class:`OWChirpAnalysis`.
    :class:`~hermes_rheo.transforms.owchirp_generation.OWChirpGeneration`
        Signal design for TRIOS Arbitrary Wave macros.

Master curves and fitting:
    :class:`~hermes_rheo.transforms.automated_mastercurve.AutomatedMasterCurve`,
    :class:`~hermes_rheo.transforms.automated_mastercurve.CompositeMasterCurve`
        Data-driven master-curve construction (via ``mastercurves``).
    :class:`~hermes_rheo.transforms.fit_model.FitModel`,
    :class:`~hermes_rheo.transforms.fit_model.MultiModeMaxwellModel`
        Constitutive-model fitting (built-in generalized Maxwell, or pyRheo
        models when pyRheo is installed).
    :class:`~hermes_rheo.transforms.shift_factor_models.ShiftFactorModel`,
    :class:`~hermes_rheo.transforms.shift_factor_models.WLFModel`,
    :class:`~hermes_rheo.transforms.shift_factor_models.ArrheniusModel`
        Temperature dependence of TTS shift factors.
    :class:`~hermes_rheo.transforms.frequency_to_time.FrequencyToTime`
        Relaxation modulus G(t) from G*(omega) (i-Rheo Tempo method).
    :class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`
        Continuous and discrete relaxation spectra (pyReSpect, vendored).

Temperature, recovery and mutation:
    :class:`~hermes_rheo.transforms.temperature_ramp_analysis.TemperatureRampAnalysis`,
    :class:`~hermes_rheo.transforms.recovery_rheology.RecoveryRheology`,
    :class:`~hermes_rheo.transforms.mutation_number.MutationNumber`.

Utilities:
    :class:`~hermes_rheo.transforms.split_temperature_sweep.SplitTemperatureSweep`,
    :class:`~hermes_rheo.transforms.states_to_conditions.StatesToConditions`,
    :class:`~hermes_rheo.transforms.kymograph.Kymograph`.
"""
from __future__ import annotations

from hermes_rheo.transforms.automated_mastercurve import (
    AutomatedMasterCurve,
    CompositeMasterCurve,
)
from hermes_rheo.transforms.fit_model import FitModel, MultiModeMaxwellModel
from hermes_rheo.transforms.frequency_to_time import (
    FrequencyToTime,
    FrequencyToTimeResult,
)
from hermes_rheo.transforms.kymograph import Kymograph, KymographResult
from hermes_rheo.transforms.mutation_number import (
    MutationNumber,
    MutationNumberMeasurementSet,
)
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.owchirp_generation import OWChirpGeneration
from hermes_rheo.transforms.recovery_rheology import (
    RecoveryRheology,
    RecoveryRheologyResult,
)
from hermes_rheo.transforms.relaxation_spectrum import (
    RelaxationSpectrum,
    RelaxationSpectrumResult,
)
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
from hermes_rheo.transforms.shift_factor_models import (
    ArrheniusModel,
    ShiftFactorModel,
    WLFModel,
)
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep
from hermes_rheo.transforms.states_to_conditions import StatesToConditions
from hermes_rheo.transforms.temperature_ramp_analysis import (
    TemperatureRampAnalysis,
    TemperatureRampResult,
)

__all__ = [
    "OWChirpAnalysis",
    "RheoAnalysis",
    "OWChirpGeneration",
    "AutomatedMasterCurve",
    "CompositeMasterCurve",
    "FitModel",
    "MultiModeMaxwellModel",
    "ShiftFactorModel",
    "WLFModel",
    "ArrheniusModel",
    "FrequencyToTime",
    "FrequencyToTimeResult",
    "RelaxationSpectrum",
    "RelaxationSpectrumResult",
    "TemperatureRampAnalysis",
    "TemperatureRampResult",
    "RecoveryRheology",
    "RecoveryRheologyResult",
    "MutationNumber",
    "MutationNumberMeasurementSet",
    "SplitTemperatureSweep",
    "StatesToConditions",
    "Kymograph",
    "KymographResult",
]
