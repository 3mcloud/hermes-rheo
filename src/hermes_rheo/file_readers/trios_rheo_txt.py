from typing import List, Dict, Union
import re
import numpy as np
from piblin.dataio.fileio.read import FileParsingException
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.dataio.fileio.read.file_reader as file_reader


class TriosRheoReader(file_reader.FileReader):
    """
    This class provides a reader for TA Rheometers files that have been exported as .txt format using
    the TRIOS export to LIMS functionality.

    By default, both the independent and dependent variables are set to "Temperature." This setting is chosen as
    temperature is the only rheological variable consistently present across the various measurements provided by
    TRIOS software. For users seeking to further analyze or manipulate the rheological measurements, the TriosRheoReader
    class is compatible with a range of transformations available through the hermes package (https://github.com/3mcloud/hermes-rheo).
    """
    supported_extensions = {'txt'}

    __TOP_LEVEL_PARAMETER_REGEXES: Dict[str, str] = {
        "filename": r"Filename\s+(.*)",
        "instrument_serial_number": r"Instrument serial number\s+(.*)",
        "instrument_name": r"Instrument name\s+(.*)",
        "operator": r"operator\s+(.*)",
        "run_date": r"rundate\s+(.*)",
        "sample_name": r"Sample name\s+(.*)",
        "geometry": r"Geometry name\s+(.*)",\
    }
    """Regular expressions for extracting version parameters from the file."""

    __PROCEDURE_REGEXES: Dict[str, str] = {
        "procedure_name": r"Procedure name\s+(.*)",
        "procedure_segment": r"proceduresegments\s+(.*)",
    }

    __FILE_PARAMETER_REGEXES: Dict[str, str] = {
        "trios_version": r"Trios version\s+(.*)",
        "run_date": r"Run date\s+(.*)",
    }

    __GEOMETRY_REGEXES: Dict[str, str] = {
        "geometry_type": r"Geometry type\s+(.*)",
        "geometry_name": r"name\s+(.*)",
        "geometry_diameter": r"Diameter\s+(.*)",
        "geometry_gap": r"Gap\s+(.*)",
        "geometry_load_gap": r"Loading Gap\s+(.*)",
        "geometry_trim_gap_off": r"Trim gap offset",
        "geometry_material": r"Stainless steel",
    }

    __GEOMETRY_EXT_REGEXES: Dict[str, str] = {
        "geometry_exp_coefficient": r"Expansion coefficient\s+(.*)",
        "geometry_up_compliance": r"Upper Compliance\s+(.*)",
        "geometry_low_compliance": r"Lower Compliance\s+(.*)",
        "geometry_inertia": r"Geometry Inertia\s+(.*)",
        "geometry_up_mass": r"Upper Geometry Mass\s+(.*)",
        "geometry_friction": r"Friction Enabled\s+(.*)",
        "geometry_stress": r"Stress constant\s+(.*)",
        "geometry_strain": r"Strain constant\s+(.*)",
        "geometry_stress_linear": r"stress constant (linear)\s+(.*)",
        "geometry_strain_linear": r"strain constant (linear)\s+(.*)",
        "geometry_stress_constant": r"Normal Stress Constant\s+(.*)"

    }

    __CALIBRATION_PARAM_REGEXES: Dict[str, str] = {
        "SPFrtScaleLow": r"SPFrtScaleLow\s+(.*)",
        "SPFrtScaleMid": r"SPFrtScaleMid\s+(.*)",
        "transducer_full_scale": r"Transducer full scale\s+(.*)",
        "SPNrmScale": r"SPNrmScale\s+(.*)",
        "SPXducerMass": r"SPXducerMass\s+(.*)",
        "SPXducerNormDamping": r"SPXducerNormDamping\s+(.*)",
        "SPXducerNormSpringConst": r"SPXducerNormSpringConst\s+(.*)",

    }

    __REGEX_DICTS = [__PROCEDURE_REGEXES,
                     __FILE_PARAMETER_REGEXES,
                     __GEOMETRY_REGEXES,
                     __GEOMETRY_EXT_REGEXES,
                     __CALIBRATION_PARAM_REGEXES]

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

    X_VALUES_COLUMN_LABEL = 'Temperature'
    Y_VALUES_COLUMN_LABEL = 'Temperature'

    """The label for the x-values column."""

    Y_VALUES_COLUMN_LABELS = {  # EDIT as needed
        'Amplitude sweep': {
            'Angular frequency': 'angular frequency',
            'Step time': 'step time',
            'Temperature': 'temperature',
            'Axial force': 'axial force',
            'Gap': 'gap',
            'Raw phase': 'raw phase',
            'Torque': 'torque',
            'Velocity': 'velocity',
            'Oscillation torque': 'oscillation torque',
            'Oscillation displacement': 'oscillation displacement',
            'Run time': 'run time',
            'Time': 'time',
            'Stress': 'stress',
            'Stress (step)': 'stress (step)',
            'Oscillation stress': 'oscillation stress',
            'Oscillation torque (sample)': 'oscillation torque (sample)',
            'Oscillation strain': 'oscillation strain',
            'Oscillation stress (cmd)': 'oscillation stress (cmd)',
            'Loss modulus': 'loss modulus',
            'Storage modulus': 'storage modulus',
            'Normal stress': 'normal stress',
            'Shear rate': 'shear rate',
            'Oscillation strain (cmd)': 'oscillation strain (cmd)',
            'Strain constant': 'strain constant',
            'Viscosity': 'viscosity',
            'Torque (step)': 'torque (step)',
            'Date and time': 'date and time',
            '1/temperature': '1/temperature',
            'Normal stress coefficient': 'normal stress coefficient',
            'Phase angle': 'phase angle',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Dynamic viscosity': 'dynamic viscosity',
            'Out of phase component of η*': 'out of phase component of η*',
            'Complex viscosity': 'complex viscosity',
            'Complex compliance': 'complex compliance',
            'Storage compliance': 'storage compliance',
            'Loss compliance': 'loss compliance',
            'G*/sin(delta)': 'G*/sin(delta)',
            'G*.sin(delta)': 'G*.sin(delta)',
            'Stiffness': 'stiffness',
            'Oscillation strain rate': 'oscillation strain rate'
        },
        'Frequency sweep': {
            'Temperature': 'temperature',
            'Angular frequency': 'angular frequency',
            'Oscillation displacement': 'oscillation displacement',
            'Velocity': 'velocity',
            'Torque': 'torque',
            'Oscillation strain': 'oscillation strain',
            'Oscillation stress': 'oscillation stress',
            'Stress': 'stress',
            'Stress (step)': 'stress (step)',
            'Complex viscosity': 'complex viscosity',
            'Step time': 'step time',
            'Time': 'time',
            'Shear rate': 'shear rate',
            'Phase angle': 'phase angle',
            'Strain constant': 'strain constant',
            'Viscosity': 'viscosity',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Dynamic viscosity': 'dynamic viscosity',
            'Complex compliance': 'complex compliance',
            'Storage compliance': 'storage compliance',
            'Loss compliance': 'loss compliance',
            'Storage modulus': 'storage modulus',
            'Loss modulus': 'loss modulus',
            'Strain ratio': 'strain ratio',
            'Oscillation strain rate': 'oscillation strain rate'
        },
        'Flow ramp': {
            'Torque': 'torque',
            'Torque (step)': 'torque (step)',
            'Velocity': 'velocity',
            'Step time': 'step time',
            'Temperature': 'temperature',
            'Axial force': 'axial force',
            'Gap': 'gap',
            'Displacement': 'displacement',
            'Displacement (step)': 'displacement (step)',
            'Termination reason': 'termination reason',
            'Run time': 'run time',
            'Time': 'time',
            'Stress': 'stress',
            'Stress (step)': 'stress (step)',
            'Normal stress': 'normal stress',
            'Normal stress coefficient': 'normal stress coefficient',
            'Shear rate': 'shear rate',
            'Strain': 'strain',
            'Strain (step)': 'strain (step)',
            'Strain constant': 'strain constant',
            'Viscosity': 'viscosity',
            'Compliance': 'compliance',
            'Modulus': 'modulus',
            'Date and time': 'date and time',
            '1/temperature': '1/temperature'
        },
        'Creep': {
            'Temperature': 'temperature',
            'Velocity': 'velocity',
            'Strain': 'strain',
            'Stress': 'stress',
            'Time': 'time',
            'Compliance': 'compliance',
            'Modulus': 'modulus',
            'Displacement': 'displacement',
            'Shear rate': 'shear rate',
            'Step time': 'step time',
            'Axial force': 'axial force',
            'Gap': 'gap',
        },
        'Arbitrary Wave': {
            'Temperature': 'temperature',
            'Torque': 'torque',
            'Velocity': 'velocity',
            'Strain': 'strain',
            'Strain (step)': 'strain (step)',
            'Stress': 'stress',
            'Stress (step)': 'stress (step)',
            'Step time': 'step time',
            'Time': 'time',
            'Shear rate': 'shear rate',
            'Normal stress': 'normal stress',
            'Strain constant': 'strain constant',
            'Viscosity': 'viscosity',
            'Compliance': 'compliance',
            'Modulus': 'modulus',
        },
        'Multiwave': {
            'Temperature': 'temperature',
            'Angular frequency': 'angular frequency',
            'Velocity': 'velocity',
            'Torque': 'torque',
            'Strain': 'strain',
            'Oscillation strain': 'oscillation strain',
            'Oscillation stress': 'oscillation stress',
            'Stress': 'stress',
            'Stress (step)': 'stress (step)',
            'Complex viscosity': 'complex viscosity',
            'Step time': 'step time',
            'Time': 'time',
            'Shear rate': 'shear rate',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Dynamic viscosity': 'dynamic viscosity',
            'Complex compliance': 'complex compliance',
            'Loss compliance': 'loss compliance',
            'Storage modulus': 'storage modulus',
            'Loss modulus': 'Loss modulus',
            'Strain ratio': 'strain ratio',
            'Stiffness': 'stiffness',
            'Oscillation strain rate': 'Oscillation strain rate',
        },
        'Stress Relaxation': {
            'Temperature': 'temperature',
            'Velocity': 'velocity',
            'Strain': 'strain',
            'Strain (step)': 'strain (step)',
            'Strain constant (linear)': 'strain constant (linear)',
            'Stress': 'stress',
            'Step time': 'step time',
            'Time': 'time',
            'Compliance': 'compliance',
            'Modulus': 'modulus',
        },
        'Temperature ramp': {
            'Temperature': 'temperature',
            'Angular frequency': 'angular frequency',
            'Oscillation displacement': 'oscillation displacement',
            'Velocity': 'velocity',
            'Torque': 'torque',
            'Oscillation strain': 'oscillation strain',
            'Oscillation stress': 'oscillation stress',
            'Stress': 'stress',
            'Stress (step)': 'stress (step)',
            'Complex viscosity': 'complex viscosity',
            'Step time': 'step time',
            'Time': 'time',
            'Shear rate': 'shear rate',
            'Phase angle': 'phase angle',
            'Strain constant': 'strain constant',
            'Viscosity': 'viscosity',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Dynamic viscosity': 'dynamic viscosity',
            'Complex compliance': 'complex compliance',
            'Storage compliance': 'storage compliance',
            'Loss compliance': 'loss compliance',
            'Storage modulus': 'storage modulus',
            'Loss modulus': 'loss modulus',
            'Strain ratio': 'strain ratio',
            'Oscillation strain rate': 'oscillation strain rate'
        },
        'Temperature ramp ISO Force': {
            'Temperature': 'temperature',
            'Axial force': 'axial force',
            'Gap': 'gap',
            'Step time': 'step time',
            'Run time': 'run time',
            'Time': 'time',
            'Normal stress': 'normal stress',
            'Strain': 'strain',
            'Stress': 'stress',
            'Strain constant': 'strain constant',
            'Compliance': 'compliance',
            'Modulus': 'modulus',
        },
        'Temperature sweep': {
            'Angular frequency': 'angular frequency',
            'Step time': 'step time',
            'Temperature': 'temperature',
            'Axial force': 'axial force',
            'Gap': 'gap',
            'Raw phase': 'raw phase',
            'Oscillation force': 'oscillation force',
            'Oscillation displacement': 'oscillation displacement',
            'Axial displacement': 'axial displacement',
            'Run time': 'run time',
            'Time': 'time',
            'Oscillation stress': 'oscillation stress',
            'Oscillation strain': 'oscillation strain',
            'Strain': 'strain',
            'Strain (step)': 'strain (step)',
            'Storage modulus': 'storage modulus',
            'Loss modulus': 'loss modulus',
            'Phase angle': 'phase angle',
            'Axial strain': 'axial strain',
            'Axial strain %': 'axial strain %',
            'Strain constant (linear)': 'strain constant (linear)',
            'Date and time': 'date and time',
            '1/temperature': '1/temperature',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Complex compliance': 'complex compliance',
            'Storage compliance': 'storage compliance',
            'Loss compliance': 'loss compliance'
        },
        'Frequency DMA': {
            'Run time': 'run time',
            'Temperature': 'temperature',
            'Angular frequency': 'angular frequency',
            'Oscillation displacement (cmd)': 'oscillation displacement (cmd)',
            'Oscillation normal force average': 'oscillation normal force average',
            'Axial force': 'axial force',
            'Oscillation force': 'oscillation force',
            'Oscillation displacement': 'oscillation displacement',
            'Raw phase (displacement)': 'raw phase (displacement)',
            'Gap': 'gap',
            'Delta length': 'delta length',
            'X-ducer Displacement': 'x-ducer displacement',
            'Oscillation Force (Drive)': 'oscillation force (drive)',
            'Drive Position': 'drive position',
            'Oscillation strain': 'oscillation strain',
            'Oscillation strain (cmd)': 'oscillation strain (cmd)',
            'Oscillation stress': 'oscillation stress',
            'Step time': 'step time',
            'Time': 'time',
            'Status bits': 'status bits',
            'Status bits 2': 'status bits 2',
            'Normal transducer': 'normal transducer',
            'Transducer range': 'transducer range',
            'Storage modulus': 'storage modulus',
            'Loss modulus': 'loss modulus',
            'Phase angle': 'phase angle',
            'Strain ratio': 'strain ratio',
            'Pretension ratio': 'pretension ratio',
            'X-ducer stiffness': 'x-ducer stiffness',
            'Stiffness': 'stiffness',
            'Oscillation displacement (drive': 'oscillation displacement (drive',
            'Strain constant (linear)': 'strain constant (linear)',
            'Date and time': 'date and time',
            '1/temperature': '1/temperature',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Complex compliance': 'complex compliance',
            'Storage compliance': 'storage compliance',
            'Loss compliance': 'loss compliance'
        },
        'Master Curve - shift factors': {
            'Temperature': 'temperature',
            'sample density': 'density',
            '1/temperature': '1/temperature',
            'aT (x variable)': 'aT',
            'bT base': 'bt base',
            'bT delta': 'bT delta',
            'bT (y variables)': 'bT',
        },
        'Master Curve - master curve': {
            'Angular frequency': 'angular frequency',
            'Storage modulus': 'storage modulus',
            'Loss modulus': 'loss modulus',
            '1/temperature': '1/temperature',
            'Phase angle': 'phase angle',
            'Frequency': 'frequency',
            'Tan(delta)': 'tan(delta)',
            'Complex modulus': 'complex modulus',
            'Dynamic viscosity': 'dynamic viscosity',
            'Out of phase component of η*': 'out of phase component of η*',
            'Complex viscosity': 'complex viscosity',
            'Complex compliance': 'complex compliance',
            'Storage compliance': 'storage compliance',
            'Loss compliance': 'loss compliance',
            'G*/sin(delta)': 'G*/sin(delta)',
            'G*.sin(delta)': 'G*.sin(delta)',
        },
        "Recovery": {
            'Variables': 'variables',
            'Run time': 'run time',
            'Temperature': 'temperature',
            'Force': 'force',
            'Velocity': 'velocity',
            'Displacement': 'displacement',
            'Drive Position': 'drive position',
            'Delta length': 'drive length',
            'Temperature set point': 'temperature set point',
            'Strain': 'strain',
            'Strain (step)': 'strain (step)',
            'Step time': 'step time',
            'Time': 'time',
            'Status bits': 'status bits',
            'Normal transducer': 'normal transducer',
            'Stress': 'stress',
            'Recoverable strain': 'recoverable strain',
            'Recoverable compliance': 'recoverable compliance',
            'Position': 'position',
            'Total Displacement': 'total displacement',
            'Displacement(step)': 'displacement (step)',
            'Total Force': 'total force',
            'Total Stress': 'total stress',
            'Thickness': 'thickness',
            'Relative Stress': 'relative stress',
            'Recovered Strain': 'recovered strain',
            'Strain constant (linear)': 'strain constant (linear)',
            'Date and time': 'data and time',
            '1/temperature': '1/temperature',
            'Modulus': 'modulus'
        }
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
        return super().data_from_filepath(filepath,
                                          **read_kwargs)

    @staticmethod
    def __extract_wave_data_G2(file_content: List[str]) -> Dict:
        """
        Extract wave data specific to TA Ares G2 instruments from the file content.

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
        wave_counter = 0
        repeat_count = 1  # Default value if no "Repeat Count" is found

        for line in file_content:
            # Extracting Repeat Count if available in the line
            repeat_match = re.search(pattern=r"Repeat Count: (\d+)", string=line)
            if repeat_match:
                repeat_count = int(repeat_match.group(1)) + 1  # Adding 1 to account for the current wave

            match = re.search(pattern=r'Step (\d+)\s+Other-Arbitrary Wave :.*?rate (\d+)pts/s(.*?)Save images',
                              string=line, flags=re.DOTALL)
            if match:
                rate = int(match.group(2))
                wave_data = match.group(3)

                # Replicating wave data based on repeat_count
                for _ in range(repeat_count):
                    wave_counter += 1  # Increment the counter for each replication
                    step = 'Arbitrary Wave - ' + str(wave_counter)
                    data[step] = {'rate (pts/s)': rate}

                    for wave_match in re.finditer(
                            r"Wave ([\d]+)\s+((?:[\d.]*\*?[a-z()^*+\-./\d\s]+)+)\s+([\d.]*\s[a-zA-Z]+)",
                            wave_data):
                        wave_num = 'wave ' + wave_match.group(1)
                        coef = wave_match.group(2)
                        duration = float(wave_match.group(3).split()[0])

                        # Preprocess coef to remove exponent parts before extracting coefficients
                        preprocessed_coef = re.sub(r"\^[0-9]+", "", coef)

                        if wave_num not in data[step]:
                            data[step][wave_num] = {}
                        data[step][wave_num]['coef'] = [float(x) for x in re.findall(r"[+-]?(\d+\.\d+|\d+\.|\.\d+|\d+)", preprocessed_coef)]
                        data[step][wave_num]['duration (s)'] = duration
                        data[step][wave_num]['wave form'] = coef.strip() if wave_num != 'wave 1' else '0'

                # Resetting the repeat_count to default
                repeat_count = 1

        return data

    @staticmethod
    def __extract_wave_data_DHR(file_content: List[str]) -> Dict:
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

                wave_matches = re.finditer(
                    r"Wave \d+\s+(.*?)\s+(\d+\.\d+) s", line)
                for wave_match in wave_matches:
                    wave_equation = wave_match.group(1).strip()
                    # Preprocess wave equation to remove exponent parts before extracting coefficients
                    preprocessed_wave_equation = re.sub(r"\^[0-9]+", "", wave_equation)
                    duration = float(wave_match.group(2))
                    coef = [float(x) for x in
                            re.findall(r"[+-]?(\d+\.\d+|\d+\.|\.\d+|\d+)", preprocessed_wave_equation)]
                    wave_data[len(wave_data) + 1] = {
                        'coef': coef,
                        'duration (s)': duration,
                        'wave form': wave_equation  # Keeping the original wave form for reference
                    }

                original_waves.append({'rate (pts/s)': rate, 'waves': wave_data})

            repeat_match = re.search(r"Repeat Count: (\d+)", line)
            if repeat_match:
                total_repeat_count = int(repeat_match.group(1))

        # Replicate waves
        wave_counter = 0
        for _ in range(total_repeat_count + 1):
            for wave in original_waves:
                wave_counter += 1
                wave_id = f'Arbitrary Wave - {wave_counter}'
                data[wave_id] = {'rate (pts/s)': wave['rate (pts/s)']}
                for i, w in enumerate(wave['waves'].values(), 1):
                    wave_num = f'wave {i}'
                    # Copy the wave data to ensure uniqueness for each wave instance
                    data[wave_id][wave_num] = {key: value for key, value in w.items()}

        return data

    @staticmethod
    def __extract_notes_data(file_content: List[str]) -> str:
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
            if 'Project' in line:
                project_found = True
                continue
            if 'File name' in line:  # stop reading lines for notes once 'File name' is encountered
                break
            if project_found:
                notes_content.append(line.strip())  # remove leading/trailing white spaces
        notes_string = ' '.join(notes_content)
        return notes_string

    @classmethod
    def __parse_geometry_list(cls, file_contents: List[str]) -> List[str]:
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
        signal_list_start_index = 1 + cls.__find_regex_position(file_contents, cls.__GEOMETRY_HEADER_REGEX)
        signal_y_names = []
        for line in file_contents[signal_list_start_index:]:
            matcher = re.match(cls.__GEOMETRY_HEADER_REGEX, line)
            if matcher:
                signal_y_names.append(matcher.groups()[0])
            else:
                break

        return signal_y_names

    @classmethod
    def __parse_analysis_datasets(cls,
                                  file_contents: List[str],
                                  dataset_regexes: Dict[str, str]) -> \
            Dict[str,
                 zero_dimensional_dataset.ZeroDimensionalDataset]:
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

                    analysis_datasets[dataset_name] = \
                        zero_dimensional_dataset.ZeroDimensionalDataset.create(value=value,
                                                                               unit=unit,
                                                                               label=dataset_name)
        return analysis_datasets

    @classmethod
    def _data_from_file_contents(cls, file_contents, file_location=None, **read_kwargs):
        """
        This method read the contents of .txt exported using TRIOS export to LIMS functionality.
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
        metadata = cls.__read_keyed_metadata(file_contents)
        instrument_serial_number = metadata.get("instrument_serial_number")

        if instrument_serial_number and len(instrument_serial_number) >= 4:
            serial_prefix = instrument_serial_number[:4]
        else:
            raise ValueError(f"Invalid instrument serial number format: {instrument_serial_number}")

        # Define sets for instrument serial numbers
        dhr_serial_numbers = {"5343", "5332"}
        g2_serial_numbers = {"4020", "4010"}

        # Determine the correct extraction method based on the first four digits of the instrument serial number
        if serial_prefix in dhr_serial_numbers:
            details = cls.__extract_wave_data_DHR(file_contents)
        elif serial_prefix in g2_serial_numbers:
            details = cls.__extract_wave_data_G2(file_contents)
        else:
            raise ValueError(f"Unsupported instrument serial number: {serial_prefix}")

        # Update details with additional metadata
        details.update(metadata)

        # Extract additional data (example: notes)
        notes_data = cls.__extract_notes_data(file_contents)
        details['Notes'] = notes_data

        block = cls.__find_all_regex_position(file_contents[0:], cls.DATA_HEADER_REGEX)
        block.append(len(file_contents))

        measurements = []

        for index in range(len(block) - 1):
            check = re.match(r'Number of points\t\d+', file_contents[block[index] + 1])
            if check:
                offset = 2         # change between 2 or 1 according to data file structure
            else:
                offset = 1
            method = file_contents[block[index]]
            column_labels = file_contents[block[index] + offset].split("\t")
            column_labels = [column_label.strip() for column_label in column_labels]
            unit_labels = file_contents[block[index] + offset + 1].split("\t")

            x_axis_index = column_labels.index(cls.X_VALUES_COLUMN_LABEL)

            y_axis_index = []

            if all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Amplitude sweep"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Amplitude sweep"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Frequency sweep"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Frequency sweep"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Flow ramp"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Flow ramp"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Creep"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Creep"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Arbitrary Wave"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Arbitrary Wave"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Multiwave"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Multiwave"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Stress Relaxation"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Stress Relaxation"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature ramp"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature ramp"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature ramp ISO Force"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature ramp ISO Force"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Temperature sweep"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Temperature sweep"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Frequency DMA"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Frequency DMA"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Master Curve - shift factors"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Master Curve - shift factors"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Master Curve - master curve"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Master Curve - master curve"]

            elif all(col_label in column_labels for col_label in cls.Y_VALUES_COLUMN_LABELS["Recovery"]):
                select_col_labels = cls.Y_VALUES_COLUMN_LABELS["Recovery"]

            else:
                raise FileParsingException("The method of choice is currently not supported.")

            for col_label in select_col_labels:
                y_axis_index.append(column_labels.index(col_label))

            variable_names = [select_col_labels[col_label] for col_label in select_col_labels]
            variable_names.append(cls.X_VALUES_COLUMN_LABEL)
            variable_units = [unit_labels[column_labels.index(col_label)] for col_label in select_col_labels]
            variable_units.append(unit_labels[column_labels.index(cls.X_VALUES_COLUMN_LABEL)])

            dataset = []
            x_values = []
            y_values = []

            for line in file_contents[block[index] + offset + 2: block[index + 1] - 2]:
                data = line.split('\t')
                num_columns = len(data)

                new_x_values = data[x_axis_index]
                x_values.append(new_x_values)
                if len(data) == 0:
                    break
                new_y_values = data

                for i in range(len(new_y_values)):
                    if not new_y_values[i]:
                        new_y_values[i] = np.NAN
                    else:
                        try:
                            new_y_values[i] = float(new_y_values[i])
                        except:
                            new_y_values[i] = float(0)  # This will put 0s in the columns with HEX values in them (e.g. Status bits and Status bits 2)

                y_values.append(new_y_values)

            x_values = np.asarray(x_values, dtype=float)
            y_values = np.asarray(y_values, dtype=float)

            if read_kwargs.get("create_composite_datasets", True):
                datasets = []

                data_arrays = []
                for i, column in enumerate(y_axis_index):
                    data_arrays.append(y_values[:, column])

                data_arrays.append(x_values)

                composite_dataset = one_d_composite_dataset.OneDimensionalCompositeDataset(
                    data_arrays=data_arrays,
                    data_array_names=variable_names,
                    data_array_units=variable_units,
                    default_independent_name=cls.X_VALUES_COLUMN_LABEL,
                    default_dependent_name=cls.Y_VALUES_COLUMN_LABEL,
                    source='Converted .txt file from TRIOS .tri file')

                datasets.append(composite_dataset)

            else:
                datasets = []
                for i, column in enumerate(y_axis_index):
                    y_name = select_col_labels[column_labels[column]]
                    dataset = one_dimensional_dataset.OneDimensionalDataset.create(
                        y_values=y_values[:, column],
                        y_name=y_name,
                        y_unit=unit_labels[column],
                        x_values=x_values,
                        x_name=column_labels[x_axis_index],
                        x_unit=unit_labels[x_axis_index],
                        source='Converted .txt file from TRIOS .tri file'
                    )
                    datasets.append(dataset)

            measurements.append(
                measurement.Measurement(
                    datasets, conditions={"method": method}, details=details
                )
            )

        return measurement_set.MeasurementSet(measurements,
                                              merge_redundant=False)

    @staticmethod
    def __parse_str(value: str) -> Union[int, float, str, bool]:
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
    def __find_metadata(cls, file_contents: List[str], regex: str) -> Union[int, float, str]:
        """Find and return the value of scalar metadata matching a regex.

        Parameters
        ----------
        file_contents : list of str
            The contents of the Trios DSC .txt file.
        regex : str
            The regex for the scalar metadata.

        Returns
        -------
        int or float or str
            The value for the metadata.
        """
        for line in file_contents:
            matched = re.match(regex, line)
            if matched is not None:
                return cls.__parse_str(matched.groups()[0].strip())

    @classmethod
    def __read_keyed_metadata(cls, file_contents: List[str], regex_dict=None) -> Dict[str, Union[int, float, str]]:
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
    def __find_regex_position(file_contents: List[str], regex: str) -> int:
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
    def __find_all_regex_position(file_contents: List[str], regex: str):
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
