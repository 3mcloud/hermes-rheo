"""hermes-rheo: rheological data analysis built on piblin.

This package layers domain-specific rheology transforms and file readers on
top of the generic data abstractions from :mod:`piblin`.

Import commonly-used symbols directly from the top-level package::

    from hermes_rheo import (
        # file readers
        TriosRheoReader, RheoCompassReader,
        # analysis transforms
        OWChirpAnalysis, AutomatedMasterCurve, TemperatureRampAnalysis,
        RecoveryRheology, MutationNumber, FitModel,
        # signal design
        OWChirpGeneration,
    )
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hermes-rheo")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

# --- File readers --------------------------------------------------------
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader

# --- Transforms ----------------------------------------------------------
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis  # deprecated alias
from hermes_rheo.transforms.automated_mastercurve import (
    AutomatedMasterCurve,
    CompositeMasterCurve,
)
from hermes_rheo.transforms.fit_model import FitModel, MultiModeMaxwellModel
from hermes_rheo.transforms.temperature_ramp_analysis import (
    TemperatureRampAnalysis,
    TemperatureRampResult,
)
from hermes_rheo.transforms.recovery_rheology import (
    RecoveryRheology,
    RecoveryRheologyResult,
)
from hermes_rheo.transforms.mutation_number import (
    MutationNumber,
    MutationNumberMeasurementSet,
)
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep
from hermes_rheo.transforms.states_to_conditions import StatesToConditions
from hermes_rheo.transforms.kymograph import Kymograph, KymographResult
from hermes_rheo.transforms.owchirp_generation import OWChirpGeneration
from hermes_rheo.transforms.frequency_to_time import (
    FrequencyToTime,
    FrequencyToTimeResult,
)
from hermes_rheo.transforms.relaxation_spectrum import (
    RelaxationSpectrum,
    RelaxationSpectrumResult,
)

__all__ = [
    "__version__",
    # file readers
    "TriosRheoReader",
    "RheoCompassReader",
    # transforms — chirp analysis
    "OWChirpAnalysis",
    "RheoAnalysis",  # deprecated alias
    # transforms — master curves & fitting
    "AutomatedMasterCurve",
    "CompositeMasterCurve",
    "FitModel",
    "MultiModeMaxwellModel",
    # transforms — temperature / mutation / recovery
    "TemperatureRampAnalysis",
    "TemperatureRampResult",
    "RecoveryRheology",
    "RecoveryRheologyResult",
    "MutationNumber",
    "MutationNumberMeasurementSet",
    # transforms — utilities
    "SplitTemperatureSweep",
    "StatesToConditions",
    "Kymograph",
    "KymographResult",
    "OWChirpGeneration",
    "FrequencyToTime",
    "FrequencyToTimeResult",
    "RelaxationSpectrum",
    "RelaxationSpectrumResult",
]
