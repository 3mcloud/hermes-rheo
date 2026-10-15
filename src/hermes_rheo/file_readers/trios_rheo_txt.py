import logging
import re
from typing import Any

import numpy as np
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset
import piblin.dataio.fileio.read.file_reader as file_reader
from piblin.dataio.fileio.read import FileParsingException

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-method default (independent, dependent) coordinates
# ---------------------------------------------------------------------------
#
# Every rheological method has a natural view — e.g. a Creep test is best
# plotted as ``strain vs time`` and a Frequency Sweep as
# ``storage modulus vs angular frequency``.  This module records that choice
# so the reader can build each ``OneDimensionalCompositeDataset`` with
# sensible defaults instead of the historical ``Temperature=f(Temperature)``
# fallback.
#
# Defaults are selected by canonicalising the ``[Step]`` name (the string
# stored in ``conditions["method"]``) and matching substrings in the order
# below.  This is more reliable than matching by column-schema because
# several TRIOS instrument profiles share column sets — e.g. a DMA
# Temperature Ramp file has every column expected by an Amplitude sweep, so
# schema-based matching would silently mis-classify the file.  The rules
# below reproduce the ``switch_coordinates()`` calls that used to live in
# the ``RheoAnalysis._apply`` elif cascade, so migrating pipelines from
# ``RheoAnalysis`` to the reader-driven defaults is a no-op.


def _canonicalise_method(method_line: str) -> str:
    """Return the canonical, lower-cased form of a TRIOS step-name line.

    TRIOS exports the step name as ``"Step name\\t<name>"`` and often
    appends a repetition tag (``" Repeat"``) or a trailing ``- N`` counter.
    All three variants collapse to the same canonical form here.

    Parameters
    ----------
    method_line : str
        The raw value stored in ``measurement.conditions["method"]``.

    Returns
    -------
    str
        The lowercase, prefix-stripped, tag-stripped canonical method name.

    Examples
    --------
    >>> _canonicalise_method("Step name\\tTemperature Ramp - 1")
    'temperature ramp'
    >>> _canonicalise_method("Frequency sweep Repeat - 3")
    'frequency sweep'
    """
    s = str(method_line).strip()
    if s.lower().startswith("step name"):
        parts = s.split("\t", 1)
        if len(parts) == 2:
            s = parts[1]
    s = s.replace(" Repeat", "")
    s = re.sub(r"\s*-\s*\d+\s*$", "", s)
    return s.strip().lower()


def _lookup_method_defaults(
    method_line: str,
    variable_names: list[str],
    fallback_independent: str,
) -> tuple[str, str]:
    """Return the natural ``(default_independent, default_dependent)`` pair.

    Matches the canonicalised method string against an ordered set of
    substring rules and returns the corresponding coordinates.  When the
    chosen names are not both present in *variable_names* (some TRIOS
    exports omit columns), a safe fallback is returned:

    * *default_independent* → *fallback_independent* (usually the appended
      ``"Temperature"`` column).
    * *default_dependent*   → the first entry of *variable_names* that is
      neither the fallback nor equal to the chosen independent.

    Parameters
    ----------
    method_line : str
        The raw method-line string from
        ``measurement.conditions["method"]``.
    variable_names : list of str
        Names of the data arrays that will make up the composite dataset,
        in construction order.
    fallback_independent : str
        Column name to use as the independent variable when the natural
        default is not present in *variable_names*.

    Returns
    -------
    default_independent : str
    default_dependent : str
    """
    m = _canonicalise_method(method_line)

    # Order matches RheoAnalysis._apply's elif cascade.
    if "arbitrary wave" in m:
        default_x, default_y = "step time", "strain"
    elif "frequency chirp" in m:
        default_x, default_y = "step time", "strain"
    elif "creep" in m:
        default_x, default_y = "time", "strain"
    elif "amplitude sweep" in m:
        default_x, default_y = "oscillation strain (cmd)", "storage modulus"
    elif (
        ("frequency sweep" in m)
        or ("frequenz" in m)
        or (m == "frequency")
        or (m.startswith("frequency") and "chirp" not in m)
    ):
        default_x, default_y = "angular frequency", "storage modulus"
    elif "multiwave" in m:
        default_x, default_y = "angular frequency", "storage modulus"
    elif "stress relaxation" in m:
        default_x, default_y = "step time", "modulus"
    elif "temperature ramp iso force" in m:
        default_x, default_y = "temperature", "modulus"
    elif ("temperature ramp" in m) or ("temperaturrampe" in m):
        default_x, default_y = "temperature", "storage modulus"
    elif "temperature sweep" in m:
        default_x, default_y = "temperature", "storage modulus"
    elif "frequency dma" in m:
        default_x, default_y = "angular frequency", "storage modulus"
    elif ("shift factors" in m) and ("master curve" not in m):
        default_x, default_y = "1/temperature", "aT"
    elif "master curve - shift factors" in m:
        default_x, default_y = "1/temperature", "aT"
    elif "master curve - master curve" in m:
        default_x, default_y = "angular frequency", "storage modulus"
    elif ("master curve" in m) and ("master curve -" not in m):
        default_x, default_y = "angular frequency", "storage modulus"
    elif "fast sampling" in m:
        default_x, default_y = "step time", "storage modulus"
    elif "recovery" in m:
        default_x, default_y = "step time", "modulus"
    elif "nanodma" in m:
        default_x, default_y = "frequency", "storage modulus"
    elif "axial" in m:
        default_x, default_y = "step time", "axial force"
    elif "correlated" in m:
        default_x, default_y = "angular frequency", "storage modulus"
    elif "stress control" in m:
        default_x, default_y = "time", "storage modulus"
    elif "amplitude" in m:
        default_x, default_y = "oscillation strain", "storage modulus"
    elif "flow sweep" in m or "flow ramp" in m:
        default_x, default_y = "shear rate", "viscosity"
    elif "sine strain" in m:
        default_x, default_y = "stress", "strain"
    elif "time" in m:
        default_x, default_y = "step time", "storage modulus"
    else:
        _logger.debug(
            "No default-coordinate rule matched method %r; falling back to (%r, first-dependent).",
            method_line,
            fallback_independent,
        )
        default_x = fallback_independent
        default_y = _first_non(variable_names, {fallback_independent})

    # Safety fallbacks when a chosen name is not in this file's columns.
    if default_x not in variable_names:
        _logger.debug(
            "Default independent %r for method %r not in columns; falling back to %r.",
            default_x,
            method_line,
            fallback_independent,
        )
        default_x = fallback_independent
    if default_y not in variable_names:
        _logger.debug(
            "Default dependent %r for method %r not in columns; "
            "falling back to first available dependent.",
            default_y,
            method_line,
        )
        default_y = _first_non(variable_names, {fallback_independent, default_x})

    return default_x, default_y


def _first_non(names: list[str], excluded: set) -> str:
    """Return the first entry of *names* not in *excluded* (or names[0])."""
    for name in names:
        if name not in excluded:
            return name
    return names[0]


class TriosRheoReader(file_reader.FileReader):
    """Reader for TA rheometer ``.txt`` exports produced by TRIOS.

    Two TRIOS export paths write the full step-header metadata this
    reader needs:

    * **TRIOS ≤5** — *Export* → *Export to LIMS*.
    * **TRIOS 6.0+** — *Export* → *Export to .txt (Full)*.  The plain
      *Export to .txt* option strips the geometry, calibration and
      wave parameters and will raise a
      :class:`~piblin.dataio.fileio.read.FileParsingException`.

    The latest TRIOS is available free of charge from
    `TA Instruments <https://www.tainstruments.com/support/software-downloads-support/downloads/>`_.

    Each ``[Step]`` in the file becomes one
    :class:`~piblin.data.data_collections.measurement.Measurement`.  When
    ``create_composite_datasets=True`` (the default) each measurement holds a
    single
    :class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
    containing every column mapped by
    :attr:`Y_VALUES_COLUMN_LABELS` plus the appended ``"Temperature"``
    coordinate.

    The active (independent, dependent) coordinate pair for each measurement
    is controlled by the ``default_coords`` read-time keyword argument:

    ``"natural"`` (default)
        Use the scientifically meaningful pair for each method — see
        :data:`_METHOD_DEFAULTS`.  For example a *Creep* step becomes
        ``strain=f(time)`` and a *Frequency Sweep* becomes
        ``storage modulus=f(angular frequency)``.  This yields clean
        ``print(measurement_set)`` output but *individual measurements from a
        mixed-method file will have different active coordinates*, so wrapping
        the resulting ``MeasurementSet`` directly in an ``ExperimentSet``
        (which enforces consistency across measurements) can raise
        ``InconsistentMeasurementsError``.

    ``"legacy-temperature"``
        Reproduces the historical behaviour: every measurement gets
        ``Temperature=f(Temperature)`` regardless of method.  Ugly but
        consistent, so mixed-method files can be wrapped directly in
        ``ExperimentSet``.  Provided for backwards compatibility with code
        written against hermes-rheo 1.x.

    Users can always override the active coordinates on any measurement with
    :meth:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset.switch_coordinates`.

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> mset = TriosRheoReader().data_from_filepath(
    ...     "creep.txt", create_composite_datasets=True
    ... )  # doctest: +SKIP
    >>> ds = mset.measurements[0].datasets[0]  # doctest: +SKIP
    >>> ds.independent_variable_name, ds.dependent_variable_name  # doctest: +SKIP
    ('time', 'strain')

    Legacy behaviour for mixed-method files intended for ``ExperimentSet``:

    >>> mset = TriosRheoReader().data_from_filepath(
    ...     "mixed.txt",
    ...     create_composite_datasets=True,
    ...     default_coords="legacy-temperature",
    ... )  # doctest: +SKIP
    """

    supported_extensions = {"txt"}

    __TOP_LEVEL_PARAMETER_REGEXES: dict[str, str] = {
        "document_id": r"Document ID\s+(.*)",
        "file_path": r"File path\s+(.*)",
        "instrument_serial_number": r"Instrument serial number\s+(.*)",
        "instrument_name": r"Instrument name\s+(.*)",
        "logged_in_user": r"Logged-in user at run start\s+(.*)",
        "run_date": r"Run date\s+(.*)",
    }
    """Regular expressions for extracting top-level header parameters from the file."""

    __SAMPLE_REGEXES: dict[str, str] = {
        "sample_name": r"Name\s+(.*)",
        "operator": r"Operator\s+(.*)",
        "sample_file_name": r"File name\s+(.*)",
        "sample_project": r"Project\s+(.*)",
        "sample_notes_line": r"Notes\s+(.*)",
    }

    __PROCEDURE_REGEXES: dict[str, str] = {
        "procedure_name": r"Procedure name\s+(.*)",
        "procedure_segment": r"proceduresegments\s+(.*)",
    }

    __FILE_PARAMETER_REGEXES: dict[str, str] = {
        "trios_version": r"TA Instruments LIMS export format version\s+(.*)",
        "run_date": r"Run date\s+(.*)",  # may overwrite earlier run_date if both dicts are used
    }

    __GEOMETRY_REGEXES: dict[str, str] = {
        "geometry_type": r"Geometry type\s+(.*)",  # may not exist in all files, harmless if not found
        "geometry_name": r"Name\s+(.*)",
        "geometry_diameter": r"Diameter\s+(.*)",
        "geometry_gap": r"Gap\s+(.*)",
        "geometry_load_gap": r"Loading gap\s+(.*)",
        "geometry_trim_gap_off": r"Trim gap offset\s+(.*)",
        "geometry_material": r"Material\s+(.*)",
    }

    __GEOMETRY_EXT_REGEXES: dict[str, str] = {
        "geometry_exp_coefficient": r"Expansion coefficient\s+(.*)",
        "geometry_exp_coefficient_enabled": r"Expansion coefficient enabled\s+(.*)",
        "geometry_compliance": r"Compliance\s+(.*)",
        "geometry_inertia": r"Geometry Inertia\s+(.*)",
        "geometry_friction": r"Friction\s+(.*)",
        "geometry_friction_enabled": r"Friction correction enabled\s+(.*)",
        "geometry_stress_constant": r"Stress constant\s+(.*)",
        "geometry_strain_constant": r"Strain constant\s+(.*)",
        "geometry_normal_stress_constant": r"Normal Stress Constant\s+(.*)",
        "geometry_fluid_density_constant": r"Fluid Density Constant\s+(.*)",
        "geometry_upper_compliance": r"Upper Compliance\s+(.*)",
        "geometry_lower_compliance": r"Lower Compliance\s+(.*)",
        "geometry_friction_enabled_legacy": r"Friction Enabled\s+(.*)",
    }

    __CALIBRATION_PARAM_REGEXES: dict[str, str] = {
        "SPFrtScaleLow": r"SPFrtScaleLow\s+(.*)",
        "SPFrtScaleMid": r"SPFrtScaleMid\s+(.*)",
        "transducer_full_scale": r"Transducer full scale\s+(.*)",
        "SPNrmScale": r"SPNrmScale\s+(.*)",
        "SPXducerMass": r"SPXducerMass\s+(.*)",
        "SPXducerNormDamping": r"SPXducerNormDamping\s+(.*)",
        "SPXducerNormSpringConst": r"SPXducerNormSpringConst\s+(.*)",
        "instrument_inertia": r"Instrument inertia\s+(.*)",
    }

    __REGEX_DICTS = [
        __PROCEDURE_REGEXES,
        __FILE_PARAMETER_REGEXES,
        __GEOMETRY_REGEXES,
        __GEOMETRY_EXT_REGEXES,
        __CALIBRATION_PARAM_REGEXES,
    ]

    """All of the regular expression dictionaries defined for this file reader."""

    __ANY_HEADER_REGEX = r"\[.*\]"
    """Regular expression for locating any section header in the file."""

    __GEOMETRY_HEADER_REGEX = r"\[Geometry Parameters]"
    """Regular expression for locating calibration sections in the file header."""

    __GEOMETRY_EXT_HEADER_REGEX = r"\[Geometry Extended Parameters]"
    """Regular expression for locating calibration sections in the file header."""

    __CALIBRATION_HEADER_REGEX = r"\[Calibration Parameters]"
    """Regular expression for locating calibration sections in the file header."""

    __STEP_HEADER_REGEX = r"\[Step\]"
    """Regular expression for locating step headers in the file."""

    __NUMERICAL_SUFFIXES = {
        0: "th",
        1: "st",
        2: "nd",
        3: "rd",
        4: "th",
        5: "th",
        6: "th",
        7: "th",
        8: "th",
        9: "th",
    }
    DATA_HEADER_REGEX = r"\[step\]"

    X_VALUES_COLUMN_LABEL = "Temperature"
    Y_VALUES_COLUMN_LABEL = "Temperature"

    """The label for the x-values column."""

    Y_VALUES_COLUMN_LABELS = {  # EDIT as needed
        "Amplitude sweep": {
            "Angular frequency": "angular frequency",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Raw phase": "raw phase",
            "Torque": "torque",
            "Velocity": "velocity",
            "Oscillation torque": "oscillation torque",
            "Oscillation displacement": "oscillation displacement",
            "Run time": "run time",
            "Time": "time",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Oscillation stress": "oscillation stress",
            "Oscillation torque (sample)": "oscillation torque (sample)",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress (cmd)": "oscillation stress (cmd)",
            "Loss modulus": "loss modulus",
            "Storage modulus": "storage modulus",
            "Normal stress": "normal stress",
            "Shear rate": "shear rate",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
            "Phase angle": "phase angle",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex viscosity": "complex viscosity",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "G*/sin(delta)",
            "G*.sin(delta)": "G*.sin(delta)",
            "Stiffness": "stiffness",
            "Oscillation strain rate": "oscillation strain rate",
        },
        "Frequency sweep": {
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement": "oscillation displacement",
            "Velocity": "velocity",
            "Torque": "torque",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress": "oscillation stress",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Complex viscosity": "complex viscosity",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Phase angle": "phase angle",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Strain ratio": "strain ratio",
            "Oscillation strain rate": "oscillation strain rate",
            "Gap": "gap",
        },
        "Flow ramp": {
            "Torque": "torque",
            "Torque (step)": "torque (step)",
            "Velocity": "velocity",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Displacement": "displacement",
            "Displacement (step)": "displacement (step)",
            "Termination reason": "termination reason",
            "Run time": "run time",
            "Time": "time",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Normal stress": "normal stress",
            "Normal stress coefficient": "normal stress coefficient",
            "Shear rate": "shear rate",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Compliance": "compliance",
            "Modulus": "modulus",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
        },
        "Creep": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Torque": "torque",
            "Velocity": "velocity",
            "Displacement": "displacement",
            "Normal force": "normal force",
            "Axial force": "axial force",
            "Gap": "gap",
            "Delta length": "delta length",
            "TransXducerDisplacement": "transducer displacement",
            "TransientMotorTorque": "transient motor torque",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Normal stress": "normal stress",
            "Status bits": "status bits",
            "Torque transducer": "torque transducer",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Displacement (step)": "displacement (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Arbitrary Wave": {
            "Temperature": "temperature",
            "Torque": "torque",
            "Velocity": "velocity",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Normal stress": "normal stress",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Multiwave": {
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Velocity": "velocity",
            "Torque": "torque",
            "Strain": "strain",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress": "oscillation stress",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Complex viscosity": "complex viscosity",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Complex compliance": "complex compliance",
            "Loss compliance": "loss compliance",
            "Storage modulus": "storage modulus",
            "Loss modulus": "Loss modulus",
            "Strain ratio": "strain ratio",
            "Stiffness": "stiffness",
            "Oscillation strain rate": "Oscillation strain rate",
        },
        "Stress Relaxation": {
            "Temperature": "temperature",
            "Velocity": "velocity",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Strain constant (linear)": "strain constant (linear)",
            "Stress": "stress",
            "Step time": "step time",
            "Time": "time",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Temperature ramp": {
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement": "oscillation displacement",
            "Velocity": "velocity",
            "Torque": "torque",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress": "oscillation stress",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Complex viscosity": "complex viscosity",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Phase angle": "phase angle",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Strain ratio": "strain ratio",
            "Oscillation strain rate": "oscillation strain rate",
            "Gap": "gap",
        },
        "Temperature ramp ISO Force": {
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Step time": "step time",
            "Run time": "run time",
            "Time": "time",
            "Normal stress": "normal stress",
            "Strain": "strain",
            "Stress": "stress",
            "Strain constant": "strain constant",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Temperature sweep": {
            "Angular frequency": "angular frequency",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Raw phase": "raw phase",
            "Oscillation force": "oscillation force",
            "Oscillation displacement": "oscillation displacement",
            "Axial displacement": "axial displacement",
            "Run time": "run time",
            "Time": "time",
            "Oscillation stress": "oscillation stress",
            "Oscillation strain": "oscillation strain",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Phase angle": "phase angle",
            "Axial strain": "axial strain",
            "Axial strain %": "axial strain %",
            "Strain constant (linear)": "strain constant (linear)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
        },
        "Temperature Sweep (Multifrequency)": {
            "Variables": "variables",
            "Run time": "run time",
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation force (cmd)": "oscillation force (cmd)",
            "Axial force": "axial force",
            "Oscillation force": "oscillation force",
            "Oscillation displacement": "oscillation displacement",
            "Raw phase (displacement)": "raw phase (displacement)",
            "Delta length": "delta length",
            "Oscillation Force (Drive)": "oscillation force (drive)",
            "Drive Position": "drive position",
            "Temperature set point": "temperature set point",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress": "oscillation stress",
            "Step time": "step time",
            "Time": "time",
            "Status bits": "status bits",
            "Status bits 2": "status bits 2",
            "Normal transducer": "normal transducer",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Phase angle": "phase angle",
            "Pretension ratio": "pretension ratio",
            "Stiffness": "stiffness",
            "Oscillation displacement (drive": "oscillation displacement (drive",  # note: no closing ')'
            "Position": "position",
            "Length": "length",
            "Strain constant (linear)": "strain constant (linear)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
        },
        "Frequency DMA": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement (cmd)": "oscillation displacement (cmd)",
            "Oscillation normal force average": "oscillation normal force average",
            "Axial force": "axial force",
            "Oscillation force": "oscillation force",
            "Oscillation displacement": "oscillation displacement",
            "Raw phase (displacement)": "raw phase (displacement)",
            "Gap": "gap",
            "Delta length": "delta length",
            "X-ducer Displacement": "x-ducer displacement",
            "Oscillation Force (Drive)": "oscillation force (drive)",
            "Drive Position": "drive position",
            "Oscillation strain": "oscillation strain",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Oscillation stress": "oscillation stress",
            "Step time": "step time",
            "Time": "time",
            "Status bits": "status bits",
            "Status bits 2": "status bits 2",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Phase angle": "phase angle",
            "Strain ratio": "strain ratio",
            "Pretension ratio": "pretension ratio",
            "X-ducer stiffness": "x-ducer stiffness",
            "Stiffness": "stiffness",
            "Oscillation displacement (drive": "oscillation displacement (drive",
            "Strain constant (linear)": "strain constant (linear)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
        },
        "Master Curve - shift factors": {
            "Temperature": "temperature",
            "sample density": "density",
            "1/temperature": "1/temperature",
            "aT (x variable)": "aT",
            "bT base": "bt base",
            "bT delta": "bT delta",
            "bT (y variables)": "bT",
        },
        # Frequency sweep steps exported from ARES-G3 (TRIOS 6+) that carry
        # oscillation columns and Time alongside the full complex modulus set.
        # Must be checked BEFORE 'Master Curve - master curve' because the MC
        # schema is a strict subset of these columns and would match first.
        "Frequency sweep (ARES-G3)": {
            "Angular frequency": "angular frequency",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress": "oscillation stress",
            "Time": "time",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "1/temperature": "1/temperature",
            "Phase angle": "phase angle",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex viscosity": "complex viscosity",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "G*/sin(delta)",
            "G*.sin(delta)": "G*.sin(delta)",
            "Oscillation strain rate": "oscillation strain rate",
            "Gap": "gap",
        },
        "Master Curve - master curve": {
            "Angular frequency": "angular frequency",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "1/temperature": "1/temperature",
            "Phase angle": "phase angle",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex viscosity": "complex viscosity",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "G*/sin(delta)",
            "G*.sin(delta)": "G*.sin(delta)",
        },
        "Recovery": {
            "Variables": "variables",
            "Run time": "run time",
            "Temperature": "temperature",
            "Force": "force",
            "Velocity": "velocity",
            "Displacement": "displacement",
            "Drive Position": "drive position",
            "Delta length": "drive length",
            "Temperature set point": "temperature set point",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Step time": "step time",
            "Time": "time",
            "Status bits": "status bits",
            "Normal transducer": "normal transducer",
            "Stress": "stress",
            "Recoverable strain": "recoverable strain",
            "Recoverable compliance": "recoverable compliance",
            "Position": "position",
            "Total Displacement": "total displacement",
            "Displacement(step)": "displacement (step)",
            "Total Force": "total force",
            "Total Stress": "total stress",
            "Thickness": "thickness",
            "Relative Stress": "relative stress",
            "Recovered Strain": "recovered strain",
            "Strain constant (linear)": "strain constant (linear)",
            "Date and time": "data and time",
            "1/temperature": "1/temperature",
            "Modulus": "modulus",
        },
        "Axial": {
            "Variables": "variables",
            "Torque": "torque",
            "Velocity": "velocity",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Displacement": "displacement",
            "Termination reason": "termination reason",
            "Run time": "run time",
            "Time": "time",
            "Stress": "stress",
            "Normal stress": "normal stress",
            "Shear rate": "shear rate",
            "Strain": "strain",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Time": {
            "Variables": "variables",
            "Run time": "run time",
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement (cmd)": "oscillation displacement (cmd)",
            "Oscillation normal force average": "oscillation normal force average",
            "Axial force": "axial force",
            "Oscillation torque": "oscillation torque",
            "Raw phase (torque)": "raw phase (torque)",
            "Oscillation displacement": "oscillation displacement",
            "Gap": "gap",
            "Delta length": "delta length",
            "X-ducer Displacement": "x-ducer displacement",
            "Raw Phase (Xducer displacement)": "raw phase (xducer displacement)",
            "Oscillation normal force": "oscillation normal force",
            "Raw phase (normal force)": "raw phase (normal force)",
            "Velocity": "velocity",
            "Torque": "torque",
            "ScaleTorqWaveData": "scaletorqwavedata",
            "ScaleStrnWaveData": "scalestrnwavedata",
            "ScaleNormWaveData": "scalenormwavedata",
            "Oscillation strain": "oscillation strain",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Oscillation stress": "oscillation stress",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Complex viscosity": "complex viscosity",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Oscillation normal stress": "oscillation normal stress",
            "Phase angle": "phase angle",
            "Status bits": "status bits",
            "Status bits 2": "status bits 2",
            "Torque transducer": "torque transducer",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "X-ducer stiffness": "x-ducer stiffness",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "g*/sin(delta)",
            "G*.sin(delta)": "g*.sin(delta)",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Strain ratio": "strain ratio",
            "Stiffness": "stiffness",
            "Oscillation strain rate": "oscillation strain rate",
        },
        "Frequency Chirp": {
            "Torque": "torque",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Displacement": "displacement",
            "Run time": "run time",
            "Time": "time",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Normal stress": "normal stress",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Strain constant": "strain constant",
            "Torque (step)": "torque (step)",
            "Displacement (step)": "displacement (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Correlated": {
            "Angular frequency": "angular frequency",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Raw phase": "raw phase",
            "Torque": "torque",
            "Velocity": "velocity",
            "Oscillation torque": "oscillation torque",
            "Oscillation displacement": "oscillation displacement",
            "Run time": "run time",
            "Time": "time",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Oscillation stress": "oscillation stress",
            "Oscillation torque (sample)": "oscillation torque (sample)",
            "Oscillation strain": "oscillation strain",
            "Oscillation stress (cmd)": "oscillation stress (cmd)",
            "Loss modulus": "loss modulus",
            "Storage modulus": "storage modulus",
            "Normal stress": "normal stress",
            "Shear rate": "shear rate",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
            "Phase angle": "phase angle",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex viscosity": "complex viscosity",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "G*/sin(delta)",
            "G*.sin(delta)": "G*.sin(delta)",
            "Stiffness": "stiffness",
            "Oscillation strain rate": "oscillation strain rate",
        },
        "Stress Control": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement (cmd)": "oscillation displacement (cmd)",
            "Oscillation normal force average": "oscillation normal force average",
            "Axial force": "axial force",
            "Oscillation torque": "oscillation torque",
            "Raw phase (torque)": "raw phase (torque)",
            "Oscillation displacement": "oscillation displacement",
            "Gap": "gap",
            "Delta length": "delta length",
            "X-ducer Displacement": "x-ducer displacement",
            "Raw Phase (Xducer displacement)": "raw phase (xducer displacement)",
            "Oscillation normal force": "oscillation normal force",
            "Raw phase (normal force)": "raw phase (normal force)",
            "Velocity": "velocity",
            "Torque": "torque",
            "ScaleTorqWaveData": "scaletorqwavedata",
            "ScaleStrnWaveData": "scalestrnwavedata",
            "ScaleNormWaveData": "scalenormwavedata",
            "Oscillation strain": "oscillation strain",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Oscillation stress": "oscillation stress",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Complex viscosity": "complex viscosity",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Oscillation normal stress": "oscillation normal stress",
            "Phase angle": "phase angle",
            "Status bits": "status bits",
            "Status bits 2": "status bits 2",
            "Torque transducer": "torque transducer",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "X-ducer stiffness": "x-ducer stiffness",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "g*/sin(delta)",
            "G*.sin(delta)": "g*.sin(delta)",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Strain ratio": "strain ratio",
            "Stiffness": "stiffness",
            "Oscillation strain rate": "oscillation strain rate",
        },
        "Creep (Recoverable)": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Torque": "torque",
            "Velocity": "velocity",
            "Displacement": "displacement",
            "Normal force": "normal force",
            "Axial force": "axial force",
            "Gap": "gap",
            "Delta length": "delta length",
            "TransXducerDisplacement": "transducer displacement",
            "TransientMotorTorque": "transient motor torque",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Recoverable compliance": "recoverable compliance",
            "Recoverable strain": "recoverable strain",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Normal stress": "normal stress",
            "Status bits": "status bits",
            "Torque transducer": "torque transducer",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Displacement (step)": "displacement (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
            "Modulus": "modulus",
        },
        "Amplitude": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement (cmd)": "oscillation displacement (cmd)",
            "Oscillation normal force average": "oscillation normal force average",
            "Axial force": "axial force",
            "Oscillation torque": "oscillation torque",
            "Raw phase (torque)": "raw phase (torque)",
            "Oscillation displacement": "oscillation displacement",
            "Gap": "gap",
            "Delta length": "delta length",
            "X-ducer Displacement": "x-ducer displacement",
            "Raw Phase (Xducer displacement)": "raw phase (xducer displacement)",
            "Oscillation normal force": "oscillation normal force",
            "Raw phase (normal force)": "raw phase (normal force)",
            "Velocity": "velocity",
            "Torque": "torque",
            "ScaleTorqWaveData": "scaletorqwavedata",
            "ScaleStrnWaveData": "scalestrnwavedata",
            "ScaleNormWaveData": "scalenormwavedata",
            "Oscillation strain": "oscillation strain",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Oscillation stress": "oscillation stress",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Complex viscosity": "complex viscosity",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Oscillation normal stress": "oscillation normal stress",
            "Phase angle": "phase angle",
            "Status bits": "status bits",
            "Status bits 2": "status bits 2",
            "Torque transducer": "torque transducer",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "X-ducer stiffness": "x-ducer stiffness",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Dynamic viscosity": "dynamic viscosity",
            "Out of phase component of η*": "out of phase component of η*",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
            "G*/sin(delta)": "G*/sin(delta)",
            "G*.sin(delta)": "G*.sin(delta)",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Strain ratio": "strain ratio",
            "Stiffness": "stiffness",
            "Oscillation strain rate": "oscillation strain rate",
        },
        # in Y_VALUES_COLUMN_LABELS
        "Flow Sweep": {
            "Torque": "torque",
            "Velocity": "velocity",
            "Step time": "step time",
            "Temperature": "temperature",
            "Axial force": "axial force",
            "Gap": "gap",
            "Displacement": "displacement",
            "Termination reason": "termination reason",
            "Run time": "run time",
            "Time": "time",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Normal stress": "normal stress",
            "Shear rate": "shear rate",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Displacement (step)": "displacement (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
            "Compliance": "compliance",
            "Modulus": "modulus",
        },
        "Temperature Sweep DMA": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Angular frequency": "angular frequency",
            "Oscillation displacement (cmd)": "oscillation displacement (cmd)",
            "Oscillation normal force average": "oscillation normal force average",
            "Axial force": "axial force",
            "Oscillation force": "oscillation force",
            "Oscillation displacement": "oscillation displacement",
            "Raw phase (Axial displacement)": "raw phase (axial displacement)",
            "Gap": "gap",
            "Delta length": "delta length",
            "Axial Xducer compliance": "axial xducer compliance",
            "Oscillation Force (Drive)": "oscillation force (drive)",
            "Drive Position": "drive position",
            "Oscillation strain": "oscillation strain",
            "Oscillation strain (cmd)": "oscillation strain (cmd)",
            "Oscillation stress": "oscillation stress",
            "Step time": "step time",
            "Time": "time",
            "Status bits": "status bits",
            "Status bits 2": "status bits 2",
            "Normal transducer": "normal transducer",
            "Transducer range": "transducer range",
            "Storage modulus": "storage modulus",
            "Loss modulus": "loss modulus",
            "Phase angle": "phase angle",
            "Strain ratio": "strain ratio",
            "Pretension ratio": "pretension ratio",
            "X-ducer stiffness": "x-ducer stiffness",
            "Stiffness": "stiffness",
            "Oscillation displacement (drive": "oscillation displacement (drive",
            "Strain constant (linear)": "strain constant (linear)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Frequency": "frequency",
            "Tan(delta)": "tan(delta)",
            "Complex modulus": "complex modulus",
            "Complex compliance": "complex compliance",
            "Storage compliance": "storage compliance",
            "Loss compliance": "loss compliance",
        },
        "Sine Strain": {
            "Run time": "run time",
            "Temperature": "temperature",
            "Torque": "torque",
            "Velocity": "velocity",
            "Displacement": "displacement",
            "Normal force": "normal force",
            "Axial force": "axial force",
            "Gap": "gap",
            "Delta length": "delta length",
            "TransXducerDisplacement": "transducer displacement",
            "TransientMotorTorque": "transient motor torque",
            "Strain": "strain",
            "Strain (step)": "strain (step)",
            "Stress": "stress",
            "Stress (step)": "stress (step)",
            "Step time": "step time",
            "Time": "time",
            "Shear rate": "shear rate",
            "Normal stress": "normal stress",
            "Strain constant": "strain constant",
            "Viscosity": "viscosity",
            "Torque (step)": "torque (step)",
            "Displacement (step)": "displacement (step)",
            "Date and time": "date and time",
            "1/temperature": "1/temperature",
            "Normal stress coefficient": "normal stress coefficient",
        },
    }

    """The comprehensive sets of traces present in each of the types of supported file."""

    def __init__(self):
        super().__init__()

    @property
    def default_mode(self) -> str:
        """The default mode in which to read the file."""
        return "r"

    @property
    def encoding(self) -> str:
        """The default encoding to use to read the file."""
        return "utf-8"

    def data_from_filepath(self, filepath: str, **read_kwargs) -> measurement_set.MeasurementSet:
        """Convert the file at the given path into appropriate measurement(s).

        Parameters
        ----------
        filepath : str
            The path to the file to convert to a measurement set.

        Returns
        -------
        piblin.data.data_collections.measurement_set.MeasurementSet
            The set of measurements in the file.
        """
        return super().data_from_filepath(filepath, **read_kwargs)

    @staticmethod
    def __extract_wave_data_G2(file_content: list[str]) -> dict:
        """
        This method processes the file content to extract the arbitrary wave information input by the user in the TRIOS
        software. It identifies and parses information about each wave, such as repeat count, rate, coefficients,
        and duration. The extracted data is structured into a dictionary, with each key representing a distinct wave step.
        Parameters
        ----------
        file_content : List[str]
            The content of the file as a list of strings, where each string represents a line from the file.
        Returns
        -------
        Dict
            A dictionary with keys representing each wave step (e.g., 'Arbitrary Wave - 1') and values containing
            detailed wave information such as rate, coefficients, and duration."""

        def extract_wave_coefficients(wave_data: str) -> dict:
            """
            Extract wave coefficients, durations, and wave forms from the wave data string.
            """
            wave_info = {}
            for wave_match in re.finditer(
                r"Wave (\d+)\s+(.+?)\s{2,}([\d.]+\s[a-zA-Z]+)", wave_data
            ):
                wave_num = "wave " + wave_match.group(1)
                coef = wave_match.group(2)
                duration = float(wave_match.group(3).split()[0])

                # Preprocess coef to remove exponent parts before extracting coefficients
                preprocessed_coef = re.sub(r"\^[0-9]+", "", coef)

                wave_info[wave_num] = {
                    "coef": [
                        float(x)
                        for x in re.findall(r"[+-]?(\d+\.\d+|\d+\.|\.\d+|\d+)", preprocessed_coef)
                    ],
                    "duration (s)": duration,
                    "wave form": coef.strip() if wave_num != "wave 1" else "0",
                }
            return wave_info

        data = {}
        wave_counter = 0
        repeat_count = 1  # Default value if no "Repeat Count" is found
        previous_end_temp = 0  # Initialize with a default value for ramps

        for line in file_content:
            # Extracting Repeat Count if available in the line
            repeat_match = re.search(r"Repeat Count: (\d+)", line)
            if repeat_match:
                repeat_count = (
                    int(repeat_match.group(1)) + 1
                )  # Adding 1 to account for the current wave

            match = re.search(
                r"Step (\d+)\s+Other-Arbitrary Wave :.*?rate (\d+)pts/s(.*?)Save images",
                line,
                flags=re.DOTALL,
            )
            if match:
                rate = int(match.group(2))
                wave_data = match.group(3)

                # Replicating wave data based on repeat_count
                for _ in range(repeat_count):
                    wave_counter += 1  # Increment the counter for each replication
                    step = "Arbitrary Wave - " + str(wave_counter)
                    data[step] = {"rate (pts/s)": rate}
                    data[step].update(
                        extract_wave_coefficients(wave_data)
                    )  # Extract wave coefficients

                # Resetting the repeat_count to default
                repeat_count = 1

            elif "Other-Arbitrary Wave Repeat : Ramp is selected" in line:
                # Extract ramp-specific parameters
                temp_match = re.search(r"Temperature (\d+)? °C", line)
                end_temp_match = re.search(r"End temperature ([\d.]+) °C", line)
                ramp_rate_match = re.search(r"Ramp rate ([\d.]+) °C/min", line)
                sampling_rate_match = re.search(r"Sampling rate (\d+)pts/s", line)
                playback_match = re.search(r"Playback Interval ([\d.]+) s", line)
                wave_data = re.search(
                    r"Wave Equation (.*?) Playback Interval", line, flags=re.DOTALL
                )

                if (
                    end_temp_match
                    and ramp_rate_match
                    and sampling_rate_match
                    and playback_match
                    and wave_data
                ):
                    # Use end_temp of the previous wave if temp_match is None
                    initial_temp = int(temp_match.group(1)) if temp_match else previous_end_temp
                    end_temp = float(end_temp_match.group(1))
                    temp_ramp_rate = float(ramp_rate_match.group(1))  # Temperature ramp rate
                    sampling_rate = int(sampling_rate_match.group(1))  # Sampling rate
                    playback_interval = float(playback_match.group(1))
                    wave_details = wave_data.group(1)

                    # Update the previous_end_temp for the next ramp
                    previous_end_temp = end_temp

                    # Calculate ramp duration and total waves
                    delta_temp = end_temp - initial_temp
                    ramp_time = (delta_temp / temp_ramp_rate) * 60  # Convert minutes to seconds

                    # Calculate total wave duration
                    total_wave_duration = sum(
                        float(match.group(1))
                        for match in re.finditer(r"Wave \d+\s+.*?\s+([\d.]+) s", wave_details)
                    )

                    # Total time includes ramp time and wave duration
                    total_time = ramp_time + total_wave_duration
                    repetitions = int(total_time / playback_interval) + (
                        1 if total_time % playback_interval else 0
                    )

                    # Extract individual wave information
                    wave_info = extract_wave_coefficients(wave_details)

                    # Replicate wave data based on repetitions
                    for _ in range(repetitions):
                        wave_counter += 1  # Increment the counter for each wave
                        step = "Arbitrary Wave - " + str(wave_counter)
                        data[step] = {
                            "rate (pts/s)": sampling_rate  # Populate with sampling rate
                        }
                        data[step].update(wave_info)  # Populate wave details

        return data

    @staticmethod
    def __extract_wave_data_DHR(file_content: list[str]) -> dict:
        """
        Extract wave data specific to TA DHR instruments from the file content.

        This method processes the file content to extract the arbitrary wave information input by the user in the TRIOS
        software. It identifies and parses information about each wave, such as repeat count, rate, coefficients,
        and duration. The extracted data is structured into a dictionary, with each key representing a distinct wave step.

        Parameters
        ----------
        file_content : List[str]
            The content of the file as a list of strings, where each string represents a line from the file.

        Returns
        -------
        Dict
            A dictionary with keys representing each wave step (e.g., 'Arbitrary Wave - 1') and values containing
            detailed wave information such as rate, coefficients, and duration.

        Notes
        -----
        - The method searches for specific patterns in the file content to identify and parse wave data.
        - It handles repeat counts for waves, ensuring that repeated wave data is correctly accounted for.
        - The method assumes a specific file format and structure, as expected from Ares G2 instrument outputs.
        """
        data = {}
        original_waves = []
        total_repeat_count = 1  # Default value if no "Repeat Count" is found

        # Extract original waves and repeat count
        for line in file_content:
            if "Other-Arbitrary Wave" in line:
                wave_data = {}
                rate_match = re.search(r"Rate divider \d+ \((\d+) points/s approximately\)", line)
                rate = int(rate_match.group(1)) if rate_match else None

                wave_matches = re.finditer(r"Wave \d+\s+(.*?)\s+(\d+\.\d+) s", line)
                for wave_match in wave_matches:
                    wave_equation = wave_match.group(1).strip()
                    # Preprocess wave equation to remove exponent parts before extracting coefficients
                    preprocessed_wave_equation = re.sub(r"\^[0-9]+", "", wave_equation)
                    duration = float(wave_match.group(2))
                    coef = [
                        float(x)
                        for x in re.findall(
                            r"[+-]?(\d+\.\d+|\d+\.|\.\d+|\d+)", preprocessed_wave_equation
                        )
                    ]
                    wave_data[len(wave_data) + 1] = {
                        "coef": coef,
                        "duration (s)": duration,
                        "wave form": wave_equation,  # Keeping the original wave form for reference
                    }

                original_waves.append({"rate (pts/s)": rate, "waves": wave_data})

            repeat_match = re.search(r"Repeat Count: (\d+)", line)
            if repeat_match:
                total_repeat_count = int(repeat_match.group(1))

        # Replicate waves
        wave_counter = 0
        for _ in range(total_repeat_count + 1):
            for wave in original_waves:
                wave_counter += 1
                wave_id = f"Arbitrary Wave - {wave_counter}"
                data[wave_id] = {"rate (pts/s)": wave["rate (pts/s)"]}
                for i, w in enumerate(wave["waves"].values(), 1):
                    wave_num = f"wave {i}"
                    # Copy the wave data to ensure uniqueness for each wave instance
                    data[wave_id][wave_num] = {key: value for key, value in w.items()}

        return data

    @staticmethod
    def __extract_notes_data(file_content: list[str]) -> str:
        """
        Extract notes data from the file content.
        This method processes the file content to extract the value input by the user in TRIOS under the "note" field.

        Parameters
        ----------
        file_content : List[str]
            The content of the file as a list of strings, each representing a line from the file.

        Returns
        -------
        str
            A single string containing all the notes extracted from the file. The notes are concatenated and
            separated by spaces.

        Notes
        -----
        - The method assumes that the notes section starts after the line containing 'Project' and ends before the line
        containing 'File name'.
        - It strips leading and trailing white spaces from each line before concatenation.
        """
        notes_content = []
        project_found = False
        for line in file_content:
            if "Project" in line:
                project_found = True
                continue
            if "File name" in line:  # stop reading lines for notes once 'File name' is encountered
                break
            if project_found:
                notes_content.append(line.strip())  # remove leading/trailing white spaces
        notes_string = " ".join(notes_content)
        return notes_string

    @classmethod
    def __parse_geometry_list(cls, file_contents: list[str]) -> list[str]:
        """Parse the list of signals from the file header.

        Parameters
        ----------
        file_contents : List of str
            The contents of the Trios DSC .txt file

        Returns
        -------
        List of str
            The names of signals present in the file.
        """
        signal_list_start_index = 1 + cls.__find_regex_position(
            file_contents, cls.__GEOMETRY_HEADER_REGEX
        )
        signal_y_names = []
        for line in file_contents[signal_list_start_index:]:
            matcher = re.match(cls.__GEOMETRY_HEADER_REGEX, line)
            if matcher:
                signal_y_names.append(matcher.groups()[0])
            else:
                break

        return signal_y_names

    @classmethod
    def __parse_analysis_datasets(
        cls, file_contents: list[str], dataset_regexes: dict[str, str]
    ) -> dict[str, zero_dimensional_dataset.ZeroDimensionalDataset]:
        """Parse any datasets from any existing analyses in the Trios DSC .txt file.

        Parameters
        ----------
        file_contents : List of str
            The contents of the Trios DSC .txt file.
        dataset_regexes : Dict
            Dictionary linking each dataset name to a regex for its extraction.
        """

        analysis_datasets = {}
        for dataset_name, dataset_regex in dataset_regexes.items():
            for line in file_contents:
                matched = re.match(dataset_regex, line)
                if matched is not None:
                    value = cls.__parse_str(matched.groups()[0].strip())
                    unit = cls.__parse_str(matched.groups()[1].strip())

                    analysis_datasets[dataset_name] = (
                        zero_dimensional_dataset.ZeroDimensionalDataset.create(
                            value=value, unit=unit, label=dataset_name
                        )
                    )
        return analysis_datasets

    @staticmethod
    def __extract_frequency_chirp_method(file_content: list[str]) -> dict[str, dict[str, float]]:
        """
        Extract Frequency Chirp parameters from the [Procedure: Steps] block.
        Expected patterns (examples):
            - "Frequency 0.1 Hz to 10.0 Hz"
            - "Total length of chirp 10.0 s"
            - "Baseline time 2.5 s"
            - "... (976 points/s approximately)"
            - "Tapering parameter % 5.0"

        Returns
        -------
        Dict[str, Dict[str, float]]
            {
              "Frequency Chirp - 1": {
                  "waiting_time": 2.5,
                  "chirp_length": 10.0,
                  "f_lo_hz": 0.1,
                  "f_hi_hz": 10.0,
                  "tapering_percent": 5.0,
                  "fs": 976.0
              },
              ...
            }
        """
        blob_lines = []
        in_procedure = False
        for line in file_content:
            # Capture the whole [Procedure: Steps] paragraph as one blob
            if "[Procedure: Steps]" in line:
                in_procedure = True
                continue
            if in_procedure:
                if line.startswith("["):  # next section begins
                    break
                blob_lines.append(line.strip())

        # If we didn't find a dedicated block, fall back to scanning all lines
        if not blob_lines:
            blob_lines = [line.strip() for line in file_content]

        # Build a single blob string (regex-friendly)
        blob = " ".join(blob_lines)

        # Split by each occurrence of "Frequency Chirp" to count steps robustly
        # Keep the delimiter using a capturing group.
        parts = re.split(r"(Frequency\s+Chirp)", blob, flags=re.IGNORECASE)

        results: dict[str, dict[str, float]] = {}
        step_idx = 0

        # Iterate over the chunks that follow each "Frequency Chirp"
        for i in range(1, len(parts), 2):
            # parts[i] is "Frequency Chirp" (matched text), parts[i+1] is the following text chunk
            chunk = parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
            # Only treat it as a real step if it’s indeed a Frequency Chirp section
            if not re.search(r"Frequency\s+Chirp", chunk, flags=re.IGNORECASE):
                continue

            step_idx += 1
            step_key = f"Frequency Chirp - {step_idx}"

            # Parse baseline time (waiting time)
            m_baseline = re.search(r"Baseline\s+time\s+([\d.,]+)\s*s", chunk, flags=re.IGNORECASE)
            # Parse chirp total length
            m_len = re.search(
                r"Total\s+length\s+of\s+chirp\s+([\d.,]+)\s*s", chunk, flags=re.IGNORECASE
            )
            # Parse frequency range "Frequency 0.1 Hz to 10.0 Hz"
            m_freq = re.search(
                r"Frequency\s+([\d.,]+)\s*Hz\s*to\s*([\d.,]+)\s*Hz", chunk, flags=re.IGNORECASE
            )
            # Parse sampling rate " (976 points/s approximately)"
            m_fs = re.search(
                r"\(([\d.,]+)\s*points/s\s*approximately\)", chunk, flags=re.IGNORECASE
            )
            # Parse tapering parameter "% 5.0"
            m_taper = re.search(
                r"Tapering\s+parameter\s*%?\s*([\d.,]+)", chunk, flags=re.IGNORECASE
            )

            def _to_float(s: str) -> float:
                s = (s or "").strip()
                if s == "":
                    return float("nan")
                s = s.replace("\u00a0", "").replace(" ", "")
                if "," in s and "." in s:
                    s = s.replace(".", "").replace(",", ".")
                elif "," in s:
                    s = s.replace(",", ".")
                return float(s)

            waiting_time = _to_float(m_baseline.group(1)) if m_baseline else float("nan")
            chirp_length = _to_float(m_len.group(1)) if m_len else float("nan")
            f_lo_hz = _to_float(m_freq.group(1)) if m_freq else float("nan")
            f_hi_hz = _to_float(m_freq.group(2)) if m_freq else float("nan")
            fs = _to_float(m_fs.group(1)) if m_fs else float("nan")
            tapering_percent = _to_float(m_taper.group(1)) if m_taper else float("nan")

            results[step_key] = {
                "waiting_time": waiting_time,
                "chirp_length": chirp_length,
                "f_lo_hz": f_lo_hz,
                "f_hi_hz": f_hi_hz,
                "tapering_percent": tapering_percent,
                "fs": fs,
            }

        return results

    @staticmethod
    def _parse_num(s: str):
        s = (s or "").strip()
        if s == "":
            return np.nan
        # Remove spaces / non-breaking spaces (handles "1 234,56")
        s = s.replace("\u00a0", "").replace(" ", "")
        # If both ',' and '.' appear, assume '.' are thousands sep and ',' is decimal
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        # If only ',' appears, treat it as decimal separator
        elif "," in s:
            s = s.replace(",", ".")
        return float(s)

    @classmethod
    def _data_from_file_contents(cls, file_contents, file_location=None, **read_kwargs):
        """
        This method reads the contents of a ``.txt`` produced by either
        the TRIOS *Export to LIMS* command (TRIOS ≤5) or the TRIOS 6.0+
        *Export to .txt (Full)* command.  The plain *Export to .txt*
        option omits step metadata and will fail to parse.
        The method supports various types of rheological data and is tailored to handle specific file formats outputted by
        TA instruments (Ares G2, DHR, etc..)

        Parameters
        ----------
        file_contents : str
            The raw string content of the .txt file.
        file_location : str, optional
            The location of the file, by default None.
        read_kwargs : dict, optional
            Additional keyword arguments for reading the file, by default an empty dict.

        Returns
        -------
        measurement_set.MeasurementSet
            A structured set of measurements extracted from the file, organized as a MeasurementSet object.

        Raises
        ------
        FileParsingException
            If the file format is not recognized or does not match expected formats for supported instruments.

        Notes
        -----
        - The method splits the file content into lines and processes each line to extract relevant data.
        - It identifies the instrument type from the file metadata and uses the appropriate data extraction method.
        - The method handles various measurement types, including amplitude sweep, frequency sweep, flow ramp, and others,
        depending on the instrument's capabilities.
        """

        file_contents = file_contents.split("\n")
        metadata = {}

        # Top-level/header/sample
        for regex_dict in [
            cls.__TOP_LEVEL_PARAMETER_REGEXES,
            cls.__FILE_PARAMETER_REGEXES,
            cls.__SAMPLE_REGEXES,  # if you add it
            cls.__GEOMETRY_REGEXES,
            cls.__GEOMETRY_EXT_REGEXES,
            cls.__CALIBRATION_PARAM_REGEXES,
        ]:
            metadata.update(cls.__read_keyed_metadata(file_contents, regex_dict))
        instrument_serial_number = metadata.get("instrument_serial_number")

        # If TRIOS wrote ``"Offline"`` in the serial-number field, try to
        # recover the real serial from the instrument-name line (TRIOS
        # writes ``"TASerNo 4010-0321"`` there when the instrument was
        # disconnected at export time).
        if instrument_serial_number == "Offline":
            instrument_name = metadata.get("instrument_name")
            if instrument_name:
                parts = instrument_name.split()
                if len(parts) >= 2:
                    instrument_serial_number = parts[1].split("-")[0]
                    metadata["instrument_serial_number"] = instrument_serial_number
                else:
                    instrument_serial_number = None
                    metadata["instrument_serial_number"] = None
            else:
                instrument_serial_number = None
                metadata["instrument_serial_number"] = None

        # Serial-number dispatch selects the OWChirp wave-metadata parser
        # (Arbitrary Wave layout differs between DHR and ARES G2/G3 /
        # Discovery HR).  Non-OWChirp measurements do not consume
        # ``wave_details`` at all, so if the serial is missing or
        # unfamiliar we still read the file — we simply leave
        # ``wave_details`` empty and let ``OWChirpAnalysis`` raise a
        # targeted error if a caller later tries to analyse an Arbitrary
        # Wave step from an unknown instrument.
        _DHR_SERIALS = {"5343", "5332", "5333"}
        _G2_SERIALS = {"4020", "4010"}
        _G3_SERIALS = {"4060"}
        _DISCOVERY_SERIALS = {"0850"}

        serial_prefix = (
            instrument_serial_number[:4]
            if instrument_serial_number and len(instrument_serial_number) >= 4
            else None
        )

        wave_details: dict[str, Any] = {}
        if serial_prefix in _DHR_SERIALS:
            wave_details = cls.__extract_wave_data_DHR(file_contents)
        elif serial_prefix in _G2_SERIALS | _G3_SERIALS | _DISCOVERY_SERIALS:
            wave_details = cls.__extract_wave_data_G2(file_contents)
        elif serial_prefix is None:
            _logger.debug(
                "TRIOS file has no parseable instrument serial number; "
                "OWChirp wave metadata will not be extracted.",
            )
        else:
            _logger.debug(
                "TRIOS file has unrecognised instrument serial prefix %r; "
                "OWChirp wave metadata will not be extracted.",
                serial_prefix,
            )

        # Now assemble the full details dict
        details: dict[str, Any] = {}

        # 1) wave-specific Arbitrary Wave / Frequency Chirp parameters
        details.update(wave_details)

        # 2) general metadata from the header
        details.update(metadata)

        # 3) Notes block
        notes_data = cls.__extract_notes_data(file_contents)
        details["Notes"] = notes_data

        # 4) Details from Frequency Chirp method (structured per-step blocks)
        freq_chirp_details = cls.__extract_frequency_chirp_method(file_contents)
        if freq_chirp_details:
            details.update(freq_chirp_details)

        block = cls.__find_all_regex_position(file_contents[0:], cls.DATA_HEADER_REGEX)
        block.append(len(file_contents))

        measurements = []

        for index in range(len(block) - 1):
            check = re.match(r"Number of points\t\d+", file_contents[block[index] + 1])
            if check:
                offset = 2  # change between 2 or 1 according to data file structure
            else:
                offset = 1
            method = file_contents[block[index]]
            column_labels = file_contents[block[index] + offset].split("\t")
            column_labels = [column_label.strip() for column_label in column_labels]
            unit_labels = file_contents[block[index] + offset + 1].split("\t")

            x_axis_index = column_labels.index(cls.X_VALUES_COLUMN_LABEL)

            y_axis_index = []

            if all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Amplitude sweep"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Amplitude sweep"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Amplitude"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Amplitude"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Frequency sweep"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Frequency sweep"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Flow ramp"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Flow ramp"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Creep"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Creep"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Creep (Recoverable)"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Creep (Recoverable)"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Arbitrary Wave"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Arbitrary Wave"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Multiwave"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Multiwave"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Stress Relaxation"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Stress Relaxation"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature ramp"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature ramp"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature ramp ISO Force"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature ramp ISO Force"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature sweep"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature sweep"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Frequency DMA"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Frequency DMA"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Master Curve - shift factors"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Master Curve - shift factors"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Frequency sweep (ARES-G3)"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Frequency sweep (ARES-G3)"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Master Curve - master curve"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Master Curve - master curve"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Recovery"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Recovery"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature Sweep (Multifrequency)"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature Sweep (Multifrequency)"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Axial"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Axial"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Correlated"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Correlated"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Frequency Chirp"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Frequency Chirp"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Time"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Time"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Stress Control"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Stress Control"]

            elif all(
                col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Flow Sweep"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Flow Sweep"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature Sweep DMA"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature Sweep DMA"]

            elif all(
                col_label in column_labels
                for col_label in cls.Y_VALUES_COLUMN_LABELS["Sine Strain"]
            ):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Sine Strain"]

            else:
                # Pull a human-friendly method label from the step line
                method_line = file_contents[block[index]].strip()
                if method_line.lower().startswith("step name"):
                    try:
                        method_label = method_line.split("\t", 1)[1].strip()
                    except Exception:
                        method_label = method_line
                else:
                    method_label = method_line

                raise FileParsingException(
                    f"Unsupported/unknown method in this step: {method_label}"
                )

            for col_label in select_col_labels:
                y_axis_index.append(column_labels.index(col_label))

            variable_names = [select_col_labels[col_label] for col_label in select_col_labels]
            variable_names.append(cls.X_VALUES_COLUMN_LABEL)
            variable_units = [
                unit_labels[column_labels.index(col_label)] for col_label in select_col_labels
            ]
            variable_units.append(unit_labels[column_labels.index(cls.X_VALUES_COLUMN_LABEL)])

            dataset = []
            x_values = []
            y_values = []

            # read "Number of points" from the header line just after "Step name"
            n_points = None
            npts_line = file_contents[block[index] + 1].strip()
            m_npts = re.match(r"Number of points\s*\t\s*(\d+)", npts_line, re.IGNORECASE)
            if m_npts:
                n_points = int(m_npts.group(1))

            data_start = block[index] + offset + 2  # first data row after Units line

            if n_points is not None and n_points > 0:
                data_end = data_start + n_points  # read exactly N rows
            else:
                # fallback to old behavior if we somehow can't parse Number of points
                data_end = block[index + 1] - 2

            for line in file_contents[data_start:data_end]:
                if not line.strip():
                    # skip blank lines just in case
                    continue

                data = line.split("\t")

                # x
                x_raw = data[x_axis_index]
                x_values.append(cls._parse_num(x_raw))

                if len(data) == 0:
                    break

                new_y_values = data  # list of strings

                for i in range(len(new_y_values)):
                    if not new_y_values[i]:
                        new_y_values[i] = np.nan
                    else:
                        try:
                            new_y_values[i] = cls._parse_num(new_y_values[i])
                        except Exception:
                            # Columns with HEX/status, etc.
                            new_y_values[i] = 0.0

                y_values.append(new_y_values)

            x_values = np.asarray(x_values, dtype=float)
            y_values = np.asarray(y_values, dtype=float)

            if read_kwargs.get("create_composite_datasets", True):
                datasets = []
                data_arrays = []
                for column in y_axis_index:
                    data_arrays.append(y_values[:, column])
                data_arrays.append(x_values)

                # Choose default (independent, dependent) coordinates for this
                # measurement.  The "natural" mode (default) uses the
                # scientifically meaningful pair for each method (e.g.
                # strain=f(time) for Creep, G'=f(omega) for a frequency
                # sweep).  The "legacy-temperature" mode reproduces the
                # historical Temperature=f(Temperature) fallback, which
                # yields ugly labels but keeps every measurement consistent
                # so that a mixed-method file can be wrapped directly in an
                # ``ExperimentSet``.
                default_coords_mode = read_kwargs.get("default_coords", "natural")
                if default_coords_mode == "legacy-temperature":
                    default_x = cls.X_VALUES_COLUMN_LABEL
                    default_y = cls.Y_VALUES_COLUMN_LABEL
                else:
                    default_x, default_y = _lookup_method_defaults(
                        method,
                        variable_names,
                        cls.X_VALUES_COLUMN_LABEL,
                    )

                composite_dataset = one_d_composite_dataset.OneDimensionalCompositeDataset(
                    data_arrays=data_arrays,
                    data_array_names=variable_names,
                    data_array_units=variable_units,
                    default_independent_name=default_x,
                    default_dependent_name=default_y,
                    source="Converted .txt file from TRIOS .tri file",
                )

                datasets.append(composite_dataset)

            else:
                datasets = []
                for column in y_axis_index:
                    y_name = select_col_labels[column_labels[column]]
                    dataset = one_dimensional_dataset.OneDimensionalDataset.create(
                        y_values=y_values[:, column],
                        y_name=y_name,
                        y_unit=unit_labels[column],
                        x_values=x_values,
                        x_name=column_labels[x_axis_index],
                        x_unit=unit_labels[x_axis_index],
                        source="Converted .txt file from TRIOS .tri file",
                    )
                    datasets.append(dataset)

            measurements.append(
                measurement.Measurement(datasets, conditions={"method": method}, details=details)
            )

        return measurement_set.MeasurementSet(measurements, merge_redundant=False)

    @staticmethod
    def __parse_str(value: str) -> int | float | str | bool:
        """Convert a string to a numerical value if possible.

        Parameters
        ----------
        value : str
            The value to convert.

        Returns
        -------
        int or float or str
            The value (converted or not).
        """
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                if value == "True":
                    return True
                elif value == "False":
                    return False
                else:
                    return value

    @classmethod
    def __find_metadata(cls, file_contents: list[str], regex: str) -> int | float | str:
        """Find and return the value of scalar metadata matching a regex.

        Assumes that, if the regex has capture groups, the *last* one is the value.
        If there are no groups, tries to take text after the first tab.
        """
        for line in file_contents:
            matched = re.match(regex, line)
            if matched is not None:
                # If the regex has capturing groups, use the last group as the value
                if matched.lastindex and matched.lastindex >= 1:
                    raw = matched.group(matched.lastindex)
                else:
                    # Fallback: try to get everything after the first tab
                    parts = line.split("\t", 1)
                    raw = parts[1] if len(parts) == 2 else matched.group(0)
                return cls.__parse_str(raw.strip())

    @classmethod
    def __read_keyed_metadata(
        cls, file_contents: list[str], regex_dict=None
    ) -> dict[str, int | float | str]:
        """Read all scalar metadata from a file.

        Parameters
        ----------
        file_contents : list of str
            The contents of the Trios DSC .txt file.

        Returns
        -------
        dict
            The details common to all measurements.
        """
        if regex_dict is None:
            regex_dict = cls.__TOP_LEVEL_PARAMETER_REGEXES

        details = {}
        for key, regex in regex_dict.items():
            value = cls.__find_metadata(file_contents, regex)
            if value is not None:
                details[key] = value

        return details

    @staticmethod
    def __find_regex_position(file_contents: list[str], regex: str) -> int:
        """Find the index of the first appearance of the regex in the file.

        Parameters
        ----------
        file_contents : list of str
            The contents of the Trios DSC .txt file.
        regex : str
            The regular expression to search for.

        Returns
        -------
        int
            The index of the string list where the regex is first found.
        """
        for i, line in enumerate(file_contents):
            matched = re.match(regex, line)
            if matched:
                return i + 1

        raise FileParsingException("Could not locate regex in file.")

    @staticmethod
    def __find_all_regex_position(file_contents: list[str], regex: str):
        """Find the index of the first appearance of the regex in the file.

        Parameters
        ----------
        file_contents : list of str
            The contents of the file.
        regex : str
            The regular expression to search for.

        Returns
        -------
        List of int
            The index of the string list where the regex is first found.
        """
        match = []
        check = False
        for i, line in enumerate(file_contents):
            matched = re.match(regex, line, re.IGNORECASE)
            if matched:
                check = True
                match.append(i + 1)
        if not check:
            raise FileParsingException(f"Could not locate regex in file.{regex}")
        else:
            return match
