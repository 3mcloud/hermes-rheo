from typing import Tuple, Dict, List, Optional, Any
import math
import copy
from scipy.fftpack import fft, fftshift
import numpy as np
from piblin.transform.abc.measurement_transform import MeasurementTransform
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset


class RheoAnalysis(MeasurementTransform):
    """
    The RheoAnalysis class, a subclass of MeasurementTransform, is designed for analyzing rheological data, particularly
    focusing on the transformation and analysis of data collected from rheological experiments involving
    Arbitrary Waves (e.g. OWChirp).

    Methods:
        prepare_chirp_data: Prepares chirp data by adjusting units and switching coordinates.
        average_until_tw: Applies a filter to the signal by averaging until a specified waiting time.
        average_over_chirp: Filters the signal by averaging over the length of the chirp only.
        average_over_all: Filters the signal by averaging over the entire duration, including waiting time.
        filter_signal: Applies a selected filtering method to the signal.
        select_best_filter_method: Determines the best filtering method based on the symmetry of the corrected signal.
        calculate_wave_parameters: Calculates various wave parameters from the wave data used to generate the chirp.
        moduli_ofr_chirp: Computes viscoelastic properties by applying fast fourier transform from strain and stress
        data collected during chirp experiments.
        _apply: Applies transformations to the target dataset based on the specified method.

    """

    def __init__(self, owchirp_waiting_time='before_signal', cutoff_points=0, filter_criterion="endpoints", *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.owchirp_waiting_time = owchirp_waiting_time
        self.cutoff_points = cutoff_points
        self.filter_criterion = filter_criterion  # Store filter_criterion as an instance variable

    @staticmethod
    def prepare_owchirp_data(dataset: one_d_composite_dataset, time: str, strain: str, stress: str, temperature: str,
                             cutoff_points: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares and processes Arbitrary Wave (e.g. chirp) data by switching coordinates and adjusting the units of
        stress and strain for Fast Fourier Transform (FFT) analysis. It also discards data before a specified cutoff time.

        Args:
            dataset (one_d_composite_dataset): The dataset to process containing time, strain, stress, and temperature data.
            time (str): The name of the time coordinate in the dataset.
            strain (str): The name of the strain coordinate in the dataset.
            stress (str): The name of the stress coordinate in the dataset.
            temperature (str): The name of the temperature coordinate in the dataset.
            cutoff_points (float): Time (in seconds) to discard data before the cutoff.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the processed arrays:
                - time values (np.ndarray): The time values starting from the cutoff point onward.
                - strain values (np.ndarray): The corresponding strain values, converted to unitless if originally in %.
                - stress values (np.ndarray): The corresponding stress values, converted to Pa if originally in MPa.
                - temperature values (np.ndarray): The temperature values starting from the cutoff point.

        Raises:
            ValueError:
                - If `cutoff_points` is greater than the maximum value of the dataset time values.
                - If `strain_values` are not set after switching coordinates to strain.
                - If `stress_values` are not set after switching coordinates to stress.
                - If an unsupported unit is found in the stress values.
        """
        # Switch coordinates to time and strain for processing
        dataset.switch_coordinates(independent_name=time, dependent_name=strain)
        time_values = dataset.x_values

        # Validate the cutoff point is less than or equal to the maximum time value
        if cutoff_points > time_values.max():
            raise ValueError(
                f"Cutoff point {cutoff_points} is greater than the maximum time value {time_values.max()}.")

        # Find the index corresponding to the cutoff time
        cutoff_index = (np.abs(time_values - cutoff_points)).argmin()

        # Adjust strain values: convert to unitless if in percentage
        if dataset.y_unit == "%":
            strain_values = copy.deepcopy(dataset.y_values[cutoff_index:]) * 0.01
            dataset.y_unit = "unitless"
        else:
            strain_values = copy.deepcopy(dataset.y_values[cutoff_index:])

        # Check if strain values are set after switching coordinates
        if strain_values is None:
            raise ValueError("Strain values not set after switching coordinates to strain.")

        # Switch coordinates to time and stress
        dataset.switch_coordinates(independent_name=time, dependent_name=stress)

        # Adjust stress values: convert MPa to Pa if necessary
        if dataset.y_unit == "MPa":
            stress_values = copy.deepcopy(dataset.y_values[cutoff_index:]) * 1e6
            dataset.y_unit = "Pa"
        elif dataset.y_unit == "Pa":
            stress_values = copy.deepcopy(dataset.y_values[cutoff_index:])
        else:
            raise ValueError(f"Unsupported stress unit '{dataset.y_unit}'. Expected 'Pa' or 'MPa'.")

        # Check if stress values are set after switching coordinates
        if stress_values is None:
            raise ValueError("Stress values not set after switching coordinates to stress.")

        # Switch coordinates to time and temperature
        dataset.switch_coordinates(independent_name=time, dependent_name=temperature)
        temperature_values = copy.deepcopy(dataset.y_values[cutoff_index:])

        # Ensure temperature_values is an ndarray
        temperature_values = np.asarray(temperature_values)

        # Ensure time_values is an ndarray
        time_values = np.asarray(time_values[cutoff_index:])

        # Return processed time, strain, stress, and temperature values
        return time_values[cutoff_index:], strain_values, stress_values, temperature_values

    @staticmethod
    def average_until_tw(t_w: float, time: List[float], signal: List[float]) -> np.ndarray:
        """
        Average until waiting time (t_w): This function subtracts the initial value of the strain/stress during
        the waiting time. It is usually the best option to correct the strain signal.

        Args:
            t_w (float): The waiting time before the chirp starts.
            time (List[float]): A list of time values corresponding to the signal.
            signal (List[float]): A list of signal values that need to be corrected.

        Returns:
            np.ndarray: An array representing the corrected signal after subtracting the initial average value.

        Raises:
            IndexError: If the time list does not contain values less than or equal to t_w.
        """
        # Ensure time and signal are converted to np.ndarray for consistent operations
        time = np.asarray(time)
        signal = np.asarray(signal)

        # Find the last index where time is less than or equal to t_w
        try:
            index_tw = np.where(time <= t_w)[0][-1]
        except IndexError:
            raise IndexError(f"No values in the time array are less than or equal to t_w = {t_w}")

        # Calculate the average of the signal up to that index
        initial_value = np.mean(signal[:index_tw + 1])

        # Subtract the average from the entire signal
        corrected_signal = signal - initial_value

        return corrected_signal

    @staticmethod
    def average_over_chirp(t_w: float, time: List[float], signal: List[float]) -> np.ndarray:
        """
        Average over chirp only: this subtracts the average of the signal over the length of the signal alone, excluding
        the initial waiting time. It can be the best option for the stress signal since the stress can be settling to
        zero during the initial waiting time.

        Args:
            t_w (float): The waiting time before the chirp starts.
            time (List[float]): A list of time values corresponding to the signal.
            signal (List[float]): A list of signal values that need to be corrected.

        Returns:
            np.ndarray: A NumPy array representing the corrected signal after subtracting the average value post t_w.

        Raises:
            IndexError: If the time list does not contain values less than or equal to t_w.
        """
        # Ensure time and signal are converted to np.ndarray for consistent operations
        time = np.asarray(time)
        signal = np.asarray(signal)

        # Find the last index where time is less than or equal to t_w
        try:
            index_tw = np.where(time <= t_w)[0][-1]
        except IndexError:
            raise IndexError(f"No values in the time array are less than or equal to t_w = {t_w}")

        # Calculate the average of the signal after t_w
        avg_signal = np.mean(signal[index_tw + 1:])

        # Subtract the average from the entire signal
        corrected_signal = signal - avg_signal

        return corrected_signal

    @staticmethod
    def average_over_all(t_w: float, time: List[float], signal: List[float]) -> np.ndarray:
        """
        Average over all: This subtracts the time-averaged value of the signal
        before and after the waiting time (t_w).

        Args:
            t_w (float): The waiting time before the chirp starts.
            time (List[float]): A list of time values corresponding to the signal.
            signal (List[float]): A list of signal values that need to be corrected.

        Returns:
            np.ndarray: A NumPy array representing the corrected signal after subtracting the average values
                        before and after t_w from the entire signal.

        Raises:
            IndexError: If the time list does not contain values less than or equal to t_w.
        """
        # Ensure time and signal are converted to np.ndarray for consistent operations
        time = np.asarray(time)
        signal = np.asarray(signal)

        # Find the last index where time is less than or equal to t_w
        try:
            index_tw = np.where(time <= t_w)[0][-1]
        except IndexError:
            raise IndexError(f"No values in the time array are less than or equal to t_w = {t_w}")

        # Calculate and subtract the averages before and after the index_tw from the entire signal
        corrected_signal = signal - np.mean(signal[:index_tw + 1]) - np.mean(signal[index_tw + 1:])

        return corrected_signal

    def filter_signal(self, t_w: float, time: List[float], signal: List[float], method: str = 'tw') -> Optional[
        np.ndarray]:
        """
        Filters the signal based on the specified method and a time threshold.

        Args:
            t_w (float): The time threshold used for filtering.
            time (List[float]): A list of time values corresponding to the signal.
            signal (List[float]): A list of signal values that need to be filtered.
            method (str, optional): The method of filtering. Options are 'tw', 'chirp', 'all', or 'none'.
                                    Defaults to 'tw'.

        Returns:
            Optional[np.ndarray]: A NumPy array representing the filtered signal.
                                   Returns None if an invalid method is selected.

        Raises:
            IndexError: If the time list does not contain values less than or equal to t_w.
        """
        # Ensure time and signal are converted to np.ndarray for consistent operations
        time = np.asarray(time)
        signal = np.asarray(signal)

        # Choose the filtering method based on the method argument
        if method == 'tw':
            corrected_signal = self.average_until_tw(t_w, time, signal)
        elif method == 'chirp':
            corrected_signal = self.average_over_chirp(t_w, time, signal)
        elif method == 'all':
            corrected_signal = self.average_over_all(t_w, time, signal)
        elif method == "none" or method == "None":
            corrected_signal = signal
        else:
            print('Invalid method selected. Choose "tw", "chirp", "all", or "none"')
            return None

        return corrected_signal

    def select_best_filter_method(self, t_w: float, time: List[float], signal: List[float],
                                  criterion: str = "symmetry") -> Tuple[np.ndarray, str]:
        """
        Iterates over different filtering methods, applies each to the signal, and evaluates the corrected signal.
        The best filtering method is selected based on the specified criterion.

        Args:
            t_w (float): The time threshold used for filtering.
            time (List[float]): A list of time values corresponding to the signal.
            signal (List[float]): A list of signal values to be filtered.
            criterion (str): The criterion for selecting the best method. Options:
                - "symmetry": Selects the method with the most symmetrical signal (lowest sum of absolute values).
                - "DMA": Selects the method with minimal bias for DMA instruments (combined start/end avg + std dev).

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the best corrected signal as a NumPy array
                                    and the name of the best method as a string.

        Raises:
            ValueError: If an invalid criterion is provided.
        """
        # Ensure time and signal are converted to np.ndarray for consistent operations
        time = np.asarray(time)
        signal = np.asarray(signal)

        methods = ['tw', 'chirp', 'all', 'none']
        best_method = None
        min_score = float('inf')

        # Validate criterion
        valid_criteria = ["symmetry", "endpoints"]
        if criterion not in valid_criteria:
            raise ValueError(f"Invalid criterion '{criterion}'. Must be one of {valid_criteria}.")

        for method in methods:
            corrected_signal = self.filter_signal(t_w, time, signal, method)
            if corrected_signal is None:
                continue

            # Calculate score based on the specified criterion
            if criterion == "symmetry":
                score = np.sum(np.abs(corrected_signal))
            elif criterion == "endpoints":
                start_end_avg = (abs(corrected_signal[0]) + abs(corrected_signal[-1])) / 2
                std_dev = np.std(corrected_signal)
                score = start_end_avg + std_dev

            if score < min_score:
                min_score = score
                best_method = method

        best_corrected_signal = self.filter_signal(t_w, time, signal, best_method)
        return best_corrected_signal, best_method


    @staticmethod
    def calculate_wave_parameters(wave_data: Dict[str, dict]) -> Tuple[float, float, float, float, float]:
        """
        Calculates wave parameters from the provided wave data. Depending on the number of waves present,
        it calculates the waiting time (tw), duration of the signal (T), tapering parameter (r), initial frequency (w0),
        and final frequency (w1) of the wave.

        Args:
            wave_data (dict): A dictionary containing wave information. The dictionary is expected to contain keys such as:
                - 'duration (s)': The duration of each wave in seconds.
                - 'coef': Coefficients corresponding to the wave parameters.

        Returns:
            Tuple[float, float, float, float, float]: A tuple containing:
                - tw (float): The waiting time before the wave starts.
                - T (float): The total duration of the signal.
                - r (float): The tapering parameter
                - w0 (float): The initial frequency (low frequency) of the wave.
                - w1 (float): The final frequency (high frequency) of the wave.

        """
        # Extract the wave information
        duration_values = [wave_data[f'wave {i}']['duration (s)'] for i in range(1, len(wave_data))]
        coef_values = [wave_data[f'wave {i}']['coef'] for i in range(1, len(wave_data))]

        if len(duration_values) == 1:
            tw = 0
            T = duration_values[0]
            r = 0
            alpha = coef_values[1][2]
            w0_alpha = coef_values[1][3]

        elif len(duration_values) == 2:  # Split if tw is in front or back
            if coef_values[0] == [0.0]:  # tw is in front
                tw = duration_values[0]
                T = duration_values[1]
                r = 0
                alpha = coef_values[1][2]
                w0_alpha = coef_values[1][3]
            else:  # tw is in the back
                tw = duration_values[1]
                T = duration_values[0]
                r = 0
                alpha = coef_values[1][2]
                w0_alpha = coef_values[1][3]

        elif len(duration_values) == 3:
            tw = 0
            T = sum(duration_values)
            pirT = coef_values[0][1]
            alpha = coef_values[0][4]
            w0_alpha = coef_values[0][5]
            r = round(math.pi / (pirT * T), 4)

        else:
            if coef_values[0] == [0.0]:  # tw is in front
                tw = duration_values[0]
                T = sum(duration_values[1:])
                pirT = coef_values[1][1]
                alpha = coef_values[1][4]
                w0_alpha = coef_values[1][5]
                r = round(math.pi / (pirT * T), 4)
            else:
                tw = duration_values[3]
                T = sum(duration_values) - tw
                pirT = coef_values[0][1]
                alpha = coef_values[0][4]
                w0_alpha = coef_values[0][5]
                r = round(math.pi / (pirT * T), 4)

        # Calculate initial frequency w0
        w0 = round(alpha * w0_alpha, 3)

        # Calculate high frequency w1
        w1 = round(w0 * math.exp(alpha * T), 3)

        return tw, T, r, w0, w1

    @staticmethod
    def moduli_owchirp_strain_controlled(strain: np.ndarray, stress: np.ndarray, fs: float, om1: float, om2: float) -> \
            Tuple[np.ndarray, ...]:
        """
        Computes the elastic modulus (G'), loss modulus (G''), and complex modulus from strain and stress in the time
        domain. The Fourier Transform is applied to the strain and stress, and the results are filtered between
        frequencies om1 and om2.

        Note:
            This code is adapted from the MITOWCh MATLAB code by Alessandro Perego (aperego@mmm.com),
            with specific Python replacements for MATLAB functions like padarray, pow2, nextpow2, and isrow.

        Args:
            strain (np.ndarray): The strain data as a NumPy array.
            stress (np.ndarray): The stress data as a NumPy array.
            fs (float): The sampling frequency.
            om1 (float): The lower angular frequency bound.
            om2 (float): The upper angular frequency bound.

        Returns:
            Tuple[np.ndarray, ...]: A tuple containing:
                - frequency (np.ndarray): The frequency array.
                - strain_final (np.ndarray): The corrected strain array.
                - stress_final (np.ndarray): The corrected stress array.
                - elastic_modulus_final (np.ndarray): The elastic modulus array (G').
                - loss_modulus_final (np.ndarray): The loss modulus array (G'').
                - complex_modulus (np.ndarray): The complex modulus array.
        """

        # Calculate lower and upper frequency bounds
        f1 = om1 / (2 * np.pi)
        f2 = om2 / (2 * np.pi)
        # Find the length of the input signals
        m = len(strain)
        # Find the next power of 2 greater than the length of the input signals
        n = 2 ** (int(np.log2(m)) + 1)

        # Pad the input data with zeros
        strain_padded = np.pad(strain, (0, n - m), 'constant')
        stress_padded = np.pad(stress, (0, n - m), 'constant')

        # FFT
        strain_fft = fft(strain_padded, n)
        stress_ftt = fft(stress_padded, n)

        # Shifting Signal
        f = (-n / 2 + np.arange(n)) * (fs / n)  # 0-centered frequency range
        shifted_strain = fftshift(strain_fft)  # Rearrange y values
        shifted_stress = fftshift(stress_ftt)  # Rearrange y values

        # Computing Moduli
        moduli = shifted_stress[int(n / 2):] / shifted_strain[int(n / 2):]
        frequency_range = f[int(n / 2):]
        upper_frequency = frequency_range[frequency_range <= f2]
        lower_frequency = upper_frequency[upper_frequency >= f1]
        frequency = 2 * np.pi * lower_frequency
        # Transpose if W is 1D array
        if np.size(frequency) == 1:
            frequency = frequency[np.newaxis]
        l1 = len(upper_frequency)
        l2 = len(lower_frequency)
        elastic_modulus = np.real(moduli)
        loss_modulus = np.imag(moduli)
        complex_modulus = moduli[l1 - l2:l1]
        elastic_modulus_final = elastic_modulus[l1 - l2:l1]
        loss_modulus_final = loss_modulus[l1 - l2:l1]
        # Transpose if Ge or Gv is 1D array
        if np.size(complex_modulus) == 1:
            complex_modulus = complex_modulus[np.newaxis]
        if np.size(elastic_modulus_final) == 1:
            elastic_modulus_final = elastic_modulus_final[np.newaxis]
        if np.size(loss_modulus_final) == 1:
            loss_modulus_final = loss_modulus_final[np.newaxis]

        # Correct for right absolute value
        stress_corrected = shifted_stress[int(n / 2):] / fs
        strain_corrected = shifted_strain[int(n / 2):] / fs
        stress_final = stress_corrected[l1 - l2:l1]
        strain_final = strain_corrected[l1 - l2:l1]
        # Transpose if Stress2 or Strain2 is 1D array
        if np.size(stress_final) == 1:
            stress_final = stress_final[np.newaxis]
        if np.size(strain_final) == 1:
            strain_final = strain_final[np.newaxis]

        return frequency, strain_final, stress_final, elastic_modulus_final, loss_modulus_final, complex_modulus,

    @staticmethod
    def moduli_owchirp_stress_controlled(strain_rate: np.ndarray, stress: np.ndarray, fs: float, om1: float,
                                         om2: float) -> \
            Tuple[np.ndarray, ...]:
        """
        Computes the elastic modulus (G'), loss modulus (G''), and complex modulus from strain and stress in the time
        domain. The Fourier Transform is applied to the strain rate and stress, and the results are filtered between
        frequencies om1 and om2.

        Note:
            This code is adapted from the MITOWCh MATLAB code by Alessandro Perego (aperego@mmm.com),
            with specific Python replacements for MATLAB functions like padarray, pow2, nextpow2, and isrow.

        Important:
            For stress-controlled instruments, it's preferable to use strain rate instead of strain for calculating
            the moduli due to the inertia of the instrument. However, the current implementation does not perform
            inertia correction, and users should be aware of potential bias in the results.

        Args:
            strain_rate (np.ndarray): The strain rate data as a NumPy array.
            stress (np.ndarray): The stress data as a NumPy array.
            fs (float): The sampling frequency.
            om1 (float): The lower angular frequency bound.
            om2 (float): The upper angular frequency bound.

        Returns:
            Tuple[np.ndarray, ...]: A tuple containing:
                - frequency (np.ndarray): The frequency array.
                - strain_final (np.ndarray): The corrected strain array.
                - stress_final (np.ndarray): The corrected stress array.
                - elastic_modulus_final (np.ndarray): The elastic modulus array (G').
                - loss_modulus_final (np.ndarray): The loss modulus array (G'').
                - complex_modulus (np.ndarray): The complex modulus array.
        """
        # Calculate lower and upper frequency bounds
        f1 = om1 / (2 * np.pi)
        f2 = om2 / (2 * np.pi)
        # Find the length of the input signals
        m = len(strain_rate)
        # Find the next power of 2 greater than the length of the input signals
        n = 2 ** (int(np.log2(m)) + 1)

        # Pad the input data with zeros
        strain_rate_padded = np.pad(strain_rate, (0, n - m), 'constant')
        stress_padded = np.pad(stress, (0, n - m), 'constant')

        # FFT
        strain_rate_fft = fft(strain_rate_padded, n)
        stress_ftt = fft(stress_padded, n)

        # Shifting Signal
        f = (-n / 2 + np.arange(n)) * (fs / n)  # 0-centered frequency range
        shifted_strain_rate = fftshift(strain_rate_fft)  # Rearrange y values
        shifted_stress = fftshift(stress_ftt)  # Rearrange y values

        # Computing Moduli
        complex_viscosity = shifted_stress[int(n / 2):] / shifted_strain_rate[int(n / 2):]
        frequency_range = f[int(n / 2):]
        upper_frequency = frequency_range[frequency_range <= f2]
        lower_frequency = upper_frequency[upper_frequency >= f1]
        frequency = 2 * np.pi * lower_frequency
        # Transpose if W is 1D array
        if np.size(frequency) == 1:
            frequency = frequency[np.newaxis]
        l1 = len(upper_frequency)
        l2 = len(lower_frequency)
        moduli = complex_viscosity * (1j*frequency_range*2*np.pi)
        elastic_modulus = np.real(moduli)
        loss_modulus = np.imag(moduli)
        complex_modulus = moduli[l1 - l2:l1]
        elastic_modulus_final = elastic_modulus[l1 - l2:l1]
        loss_modulus_final = loss_modulus[l1 - l2:l1]
        # Transpose if Ge or Gv is 1D array
        if np.size(complex_modulus) == 1:
            complex_modulus = complex_modulus[np.newaxis]
        if np.size(elastic_modulus_final) == 1:
            elastic_modulus_final = elastic_modulus_final[np.newaxis]
        if np.size(loss_modulus_final) == 1:
            loss_modulus_final = loss_modulus_final[np.newaxis]

        # Correct for right absolute value
        stress_corrected = shifted_stress[int(n / 2):] / fs
        strain_corrected = shifted_strain_rate[int(n / 2):] / fs
        stress_final = stress_corrected[l1 - l2:l1]
        strain_final = strain_corrected[l1 - l2:l1]
        # Transpose if Stress2 or Strain2 is 1D array
        if np.size(stress_final) == 1:
            stress_final = stress_final[np.newaxis]
        if np.size(strain_final) == 1:
            strain_final = strain_final[np.newaxis]

        return frequency, strain_final, stress_final, elastic_modulus_final, loss_modulus_final, complex_modulus,

    def _apply(self, target, **kwargs):
        """
        Applies the appropriate data processing pipeline to the given target dataset based on the measurement method.

        Args:
            target: The dataset target, which contains details of the measurement method, instrument, and data.
            kwargs: Additional keyword arguments for processing.

        Raises:
            ValueError: If no valid method is found or the method is not supported in the current RheoAnalysis pipeline.
            ValueError: If an unsupported TA instrument serial number is encountered.

        Returns:
            target: The processed dataset with the appropriate corrections and transformations applied.
        """
        method = target.conditions['method']
        method_key = method.split('\t')[-1]

        if 'Arbitrary Wave' in method:
            method_key = method.split('\t')[-1]  # Extract method key, e.g., 'Arbitrary Wave - 2'

            # Handle 'Repeat' in the method key
            if 'Repeat' in method_key:
                method_key = method_key.replace(' Repeat',
                                                '')  # Drop 'Repeat', e.g., 'Arbitrary Wave Repeat - 1' -> 'Arbitrary Wave - 1'

            # Check if the method key indicates a wave step > 1
            if 'Arbitrary Wave' in method_key:
                wave_number = int(method_key.split('-')[-1].strip())  # Extract the wave number

                # Reset to 'Arbitrary Wave - 1' if wave number > 1
                if wave_number > 1:
                    method_key = 'Arbitrary Wave - 1'

            wave_data = target.details[method_key]

            strain_applied = wave_data['wave 2']['coef'][0]
            waiting_time, oscillation_period, taping_parameter, initial_frequency, final_frequency = \
                self.calculate_wave_parameters(wave_data)

            sampling_frequency = wave_data['rate (pts/s)']
            original_dataset = target.datasets[0]

            time, strain, stress, temperature = self.prepare_owchirp_data(
                original_dataset, 'step time', 'strain', 'stress', 'temperature', self.cutoff_points)
            strain_filtered, filter_used_strain = self.select_best_filter_method(waiting_time, time, strain,
                                                                                 criterion=self.filter_criterion)
            stress_filtered, filter_used_stress = self.select_best_filter_method(waiting_time, time, stress,
                                                                                 criterion=self.filter_criterion)
            # Handle the owchirp_waiting_time configurations
            if self.owchirp_waiting_time == 'before_signal':
                time_tw = time[time > waiting_time]
                strain_filtered_tw = strain_filtered[time > waiting_time]
                stress_filtered_tw = stress_filtered[time > waiting_time]

            elif self.owchirp_waiting_time == 'after_signal':
                signal_end_time = oscillation_period
                signal_start_time = time[0] + 0.1
                mask = (time > signal_start_time) & (time < signal_end_time)
                time_tw = time[mask]
                strain_filtered_tw = strain_filtered[mask]
                stress_filtered_tw = stress_filtered[mask]

            elif self.owchirp_waiting_time == 'all_signal':
                time_tw = time
                strain_filtered_tw = strain_filtered
                stress_filtered_tw = stress_filtered

            strain_rate = np.diff(strain_filtered_tw) / np.diff(time_tw)

            # Stress-controlled instruments
            if target.details["instrument_serial_number"] == "5343" or target.details[
                "instrument_serial_number"] == "5332":
                frequency_radians, fourier_transform_strain, fourier_transform_stress, storage_modulus, loss_modulus, complex_modulus = \
                    self.moduli_owchirp_stress_controlled(
                        strain_rate, stress_filtered_tw, sampling_frequency,
                        initial_frequency, final_frequency)
            else:
                frequency_radians, fourier_transform_strain, fourier_transform_stress, storage_modulus, loss_modulus, complex_modulus = \
                    self.moduli_owchirp_strain_controlled(
                        strain_filtered_tw, stress_filtered_tw, sampling_frequency, initial_frequency, final_frequency)

            # Process the frequency-space data and related moduli
            frequency_hertz = frequency_radians / (2 * np.pi)
            tan_delta = loss_modulus / storage_modulus
            complex_viscosity = np.sqrt(storage_modulus ** 2 + loss_modulus ** 2) / frequency_radians
            complex_modulus_abs = np.abs(complex_modulus)

            datasets = []
            data_arrays = [
                frequency_radians,
                fourier_transform_strain, fourier_transform_stress, storage_modulus, loss_modulus,
                complex_modulus_abs, frequency_hertz, tan_delta, complex_viscosity]

            variable_names = ['angular frequency', 'fourier_transform_strain', 'fourier_transform_stress',
                              'storage modulus', 'loss modulus', 'complex modulus', 'frequency_hz', 'tan(delta)',
                              'complex viscosity']

            variable_units = ['rad/s', 'a.u.', 'a.u ', 'Pa', 'Pa', 'Pa', 'Hz', 'rad', 'Pa s']  # a.u. = arbitrary units

            dataset_fourier_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
                data_arrays=data_arrays,
                data_array_names=variable_names,
                data_array_units=variable_units,
                default_independent_name='angular frequency',
                default_dependent_name='storage modulus',
                source='Dataset in frequency space')

            # Time-space dataset
            time_filtered = time
            dataset_time_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
                data_arrays=[time_filtered, strain_filtered, stress_filtered, strain_rate],
                data_array_names=['adjusted time', 'adjusted strain', 'adjusted stress', 'adjusted strain rate'],
                data_array_units=['s', 'a.u', 'Pa', '1/s'])

            original_dataset = one_d_composite_dataset.OneDimensionalCompositeDataset.from_datasets([original_dataset,
                                                                                                     dataset_time_space,])

            # # Masked time-space dataset
            # dataset_masked_time_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
            #     data_arrays=[time_tw, strain_filtered_tw, stress_filtered_tw],
            #     data_array_names=['time masked', 'strain filtered masked', 'stress filtered masked'],
            #     data_array_units=['s', 'a.u', 'Pa'],
            #     default_independent_name='time masked',
            #     default_dependent_name='strain filtered masked',
            #     source='Dataset in time space')
            #


            datasets.append(dataset_fourier_space)
            datasets.append(original_dataset)
            # datasets.append(dataset_masked_time_space)

            target.datasets = datasets

            # Add details about the processed data
            target.add_detail('strain_applied', strain_applied)
            target.add_detail('waiting_time', waiting_time)
            target.add_detail('taping_parameter', taping_parameter)
            target.add_detail('filter_used_strain', filter_used_strain)
            target.add_detail('filter_used_stress', filter_used_stress)
            target.add_detail('initial_frequency', initial_frequency)
            target.add_detail('final_frequency', final_frequency)
            target.add_detail('oscillation_period', oscillation_period)
            target.add_detail('sampling_frequency', sampling_frequency)

        elif 'Creep' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='time', dependent_name='strain')

        elif 'Amplitude sweep' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='oscillation strain (cmd)',
                                                dependent_name='storage modulus')

        elif 'Flow ramp' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='shear rate', dependent_name='viscosity')


        elif 'Frequency' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='angular frequency', dependent_name='storage modulus')

        elif 'Multiwave' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='angular frequency', dependent_name='storage modulus')

        elif 'Stress Relaxation' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='step time', dependent_name='modulus')

        elif 'Temperature ramp' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='temperature', dependent_name='storage modulus')

        elif 'Temperature ramp ISO force' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='temperature', dependent_name='modulus')

        elif 'Step name\tTemperature Sweep' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='temperature', dependent_name='storage modulus')

        elif 'Frequency DMA' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='angular frequency', dependent_name='storage modulus')

        elif 'Master Curve - shift factors' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='temperature', dependent_name='aT')

        elif 'Master Curve - master curve' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='angular frequency', dependent_name='storage modulus')

        elif 'Recovery' in method:
            original_dataset = target.datasets[0]
            original_dataset.switch_coordinates(independent_name='step time', dependent_name='modulus')

        else:
            # Raise an error if the method is not supported in the RheoAnalysis pipeline
            raise ValueError(f"The method '{method}' is currently not supported in the RheoAnalysis pipeline.")

        return target

