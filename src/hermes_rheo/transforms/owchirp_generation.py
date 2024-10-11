import matplotlib.pyplot as plt
import math
import numpy as np
import os

class OWChirpGeneration:
    """
    Class to generate and visualize an Optimal Window Chirp (OWChirp) signal.

    Attributes:
        output_file_name (str): The name of the file to save the signal data.
        waiting_time (float): The waiting time in seconds.
        signal_duration (float): The chirp signal length in seconds.
        strain_amplitude (float): The strain amplitude as a percentage.
        initial_frequency (float): The initial frequency in rad/s.
        final_frequency (float): The final frequency in rad/s.
        tapering_parameter (float): The tapering parameter as a percentage.
        initial_phase_shift (float): The initial phase angle in degrees.
    """

    def __init__(self, output_file_name: str, waiting_time: float, signal_duration: float,
                 strain_amplitude: float, initial_frequency: float, final_frequency: float,
                 tapering_parameter: float, initial_phase_shift: float):
        """
        Initialize the OWChirpGeneration class with user-specified parameters.

        Parameters:
            output_file_name (str): The name of the file to save the signal data.
            waiting_time (float): The waiting time in seconds.
            signal_duration (float): The chirp signal length in seconds.
            strain_amplitude (float): The strain amplitude as a percentage.
            initial_frequency (float): The initial frequency in rad/s.
            final_frequency (float): The final frequency in rad/s.
            tapering_parameter (float): The tapering parameter as a percentage.
            initial_phase_shift (float): The initial phase angle in degrees.
        """
        self.output_file_name = output_file_name
        self.waiting_time = waiting_time
        self.signal_duration = signal_duration
        self.strain_amplitude = strain_amplitude
        self.initial_frequency = initial_frequency
        self.final_frequency = final_frequency
        self.tapering_parameter = tapering_parameter
        self.initial_phase_shift = initial_phase_shift
        self.time_bandwidth = round(signal_duration * (final_frequency - initial_frequency) / (2 * math.pi), 0)
        self.initial_phase_shift_radians = round(initial_phase_shift * math.pi / 180, 4)
        self.shorter_chirp_signal = round(2 * 2 * math.pi / (3 * initial_frequency), 4)



        # Precompute constants and equations
        self._precompute_constants()

    def _precompute_constants(self):
        """Precompute constants used in the chirp equations."""
        self.exponential_growth_rate = round(
            math.log(self.final_frequency / self.initial_frequency) / self.signal_duration, 4)
        self.w0_divided_by_alpha = round(self.initial_frequency / self.exponential_growth_rate, 4)
        self.w0_exp_waiting_time = round(
            self.w0_divided_by_alpha / math.exp(self.exponential_growth_rate * self.waiting_time), 4)

        # Check for zero tapering parameter to avoid division by zero
        if self.tapering_parameter != 0:
            self.pi_over_taper = round(math.pi / (self.tapering_parameter * self.signal_duration), 4)
            self.phase_shift_c1 = round(-self.pi_over_taper * self.waiting_time - math.pi / 2, 4)
            self.phase_shift_c2 = round(
                -self.pi_over_taper * self.waiting_time - math.pi / self.tapering_parameter + math.pi / 2, 4)
        else:
            self.pi_over_taper = 0
            self.phase_shift_c1 = 0
            self.phase_shift_c2 = 0

        # Calculate time segments based on the tapering parameter
        if self.tapering_parameter == 0:
            self.segment_duration_1 = self.signal_duration
            self.segment_duration_2 = 0
            self.segment_duration_3 = 0
        elif self.tapering_parameter == 1:
            self.segment_duration_1 = self.signal_duration
            self.segment_duration_2 = 0
            self.segment_duration_3 = 0
        elif self.tapering_parameter > 1:
            self.segment_duration_1 = self.signal_duration / 2
            self.segment_duration_2 = self.signal_duration / 2
            self.segment_duration_3 = 0
        else:
            self.segment_duration_1 = self.signal_duration * self.tapering_parameter / 2
            self.segment_duration_2 = self.signal_duration * (1 - self.tapering_parameter)
            self.segment_duration_3 = self.signal_duration - self.segment_duration_1 - self.segment_duration_2

        # Define zone equations
        self._define_zone_equations()

    def _define_zone_equations(self):
        """Define equations for different zones based on the tapering parameter."""
        if self.initial_phase_shift_radians < 0:
            if self.tapering_parameter < 1:
                self.zone2_equation = f'{self.strain_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c1})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})'
                self.zone3_equation = f'{self.strain_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})'
                self.zone4_equation = f'{self.strain_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c2})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})'
            else:
                self.zone2_equation = f'{self.strain_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})'
                self.zone3_equation = self.zone2_equation
                self.zone4_equation = self.zone2_equation
        else:
            if self.tapering_parameter < 1:
                self.zone2_equation = f'{self.strain_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c1})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})'
                self.zone3_equation = f'{self.strain_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})'
                self.zone4_equation = f'{self.strain_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c2})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})'
            else:
                self.zone2_equation = f'{self.strain_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})'
                self.zone3_equation = self.zone2_equation
                self.zone4_equation = self.zone2_equation

    def _chirp_function(self, time: float) -> float:
        """Compute the chirp signal value at a given time."""
        if self.tapering_parameter == 0:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time <= self.signal_duration + self.waiting_time:
                return self.strain_amplitude * math.sin(self.w0_exp_waiting_time * math.exp(
                    self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)
        elif self.tapering_parameter == 1:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time <= self.segment_duration_1 + self.waiting_time:
                return self.strain_amplitude * (
                    math.cos(self.pi_over_taper * time + self.phase_shift_c1)) ** 2 * math.sin(
                    self.w0_exp_waiting_time * math.exp(
                        self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)
        elif 0 < self.tapering_parameter < 1:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time <= self.segment_duration_1 + self.waiting_time:
                return self.strain_amplitude * (
                    math.cos(self.pi_over_taper * time + self.phase_shift_c1)) ** 2 * math.sin(
                    self.w0_exp_waiting_time * math.exp(
                        self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)
            elif self.segment_duration_1 + self.waiting_time < time <= self.segment_duration_1 + self.segment_duration_2 + self.waiting_time:
                return self.strain_amplitude * math.sin(self.w0_exp_waiting_time * math.exp(
                    self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)
            elif time > self.segment_duration_1 + self.segment_duration_2 + self.waiting_time:
                return self.strain_amplitude * (
                    math.cos(self.pi_over_taper * time + self.phase_shift_c2)) ** 2 * math.sin(
                    self.w0_exp_waiting_time * math.exp(
                        self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)
        elif self.tapering_parameter > 1:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time < self.segment_duration_1 + self.waiting_time:
                return self.strain_amplitude * (
                    math.cos(self.pi_over_taper * time + self.phase_shift_c1)) ** 2 * math.sin(
                    self.w0_exp_waiting_time * math.exp(
                        self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)
            elif self.segment_duration_1 + self.waiting_time <= time <= self.segment_duration_1 + self.segment_duration_2 + self.waiting_time:
                return self.strain_amplitude * (
                    math.cos(self.pi_over_taper * time + self.phase_shift_c2)) ** 2 * math.sin(
                    self.w0_exp_waiting_time * math.exp(
                        self.exponential_growth_rate * time) - self.w0_divided_by_alpha + self.initial_phase_shift_radians)

    def visualize(self, display_r0=False, filepath=None):
        """Plot the chirp waveform and save data and plot to user-specified folder."""

        time_values = np.arange(0, self.signal_duration + self.waiting_time, 0.0001)
        strain_values = list(map(self._chirp_function, time_values))

        # Filter out NoneType values
        valid_data = [(t, s) for t, s in zip(time_values, strain_values) if s is not None]

        # Unpack the valid time and strain values
        if valid_data:
            time_values_filtered, strain_values_filtered = zip(*valid_data)

            # Save filtered time and strain values to the specified text file
            txt_file_path = os.path.join(filepath, 'owchirp_waveform_xy.txt')
            np.savetxt(txt_file_path, np.column_stack((time_values_filtered, strain_values_filtered)),
                       header="Time(s)\tStrain", fmt='%.5e', delimiter='\t')

            # Plot the chirp signal
            plt.plot(time_values_filtered, strain_values_filtered, 'b-', label='Chirp Signal')

        if display_r0:
            # Temporarily set tapering parameter to 0 to compute the r=0 chirp
            original_tapering_parameter = self.tapering_parameter
            self.tapering_parameter = 0
            self._precompute_constants()  # Recompute constants for r=0
            r0_strain_values = list(map(self._chirp_function, time_values))
            plt.plot(time_values, r0_strain_values, color='gray', linestyle='--', label='Chirp Signal (r=0)', zorder=-1)
            # Restore original tapering parameter
            self.tapering_parameter = original_tapering_parameter
            self._precompute_constants()  # Recompute constants for the original tapering parameter

        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.axhline(0, linestyle='--', color='black')
        plt.axvline(self.waiting_time, linestyle='--', color='black')
        plt.legend(loc='best')

        png_file_path = os.path.join(filepath, 'owchirp_waveform_xy.png')
        plt.savefig(png_file_path, dpi=100)

        plt.show()

    def print_parameters(self):
        """Print the parameters and equations for the chirp signal."""
        print(f'Shorter chirp signal: {self.shorter_chirp_signal} (s), TB = {self.time_bandwidth}')
        print(f'Total length of chirp signal: {self.signal_duration} (s)')
        print(f'Initial Frequency: {self.initial_frequency} (rad/s)')
        print(f'Final Frequency: {self.final_frequency} (rad/s)')
        print(f'Strain amplitude: {self.strain_amplitude} ')
        print(f'Waiting time: {self.waiting_time} (s)')
        print(f'Tapering parameter: {self.tapering_parameter} ')
        print(f'Time Bandwidth: {self.time_bandwidth}')
        print(f'Initial Phase Shift: {self.initial_phase_shift} (°)')
        print('-------------------------------------------------------------------------')
        print(
            f'For t = waiting_time = {self.waiting_time}s write: {"Deactivate" if self.tapering_parameter == 0 else 0}')
        print(f'For the first segment duration = {round(self.segment_duration_1, 3)}s write: {self.zone2_equation}')
        print(f'For the second segment duration = {round(self.segment_duration_2, 3)}s write: {self.zone3_equation}')
        print(f'For the third segment duration = {round(self.segment_duration_3, 3)}s write: {self.zone4_equation}')
        print('--------------------------------------------------------------------------')

    def to_txt(self, filepath):
        """Save the parameters and equations to a text file in the specified folder."""
        output_file = os.path.join(filepath, 'owchirp_parameters.txt')

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f'Shorter chirp signal: {self.shorter_chirp_signal} (s), TB = {self.time_bandwidth}\n')
            file.write(f'Total length of chirp signal: {self.signal_duration} (s)\n')
            file.write(f'Initial Frequency: {self.initial_frequency} (rad/s)\n')
            file.write(f'Final Frequency: {self.final_frequency} (rad/s)\n')
            file.write(f'Strain amplitude: {self.strain_amplitude * 100} (%)\n')
            file.write(f'Waiting time: {self.waiting_time} (s)\n')
            file.write(f'Tapering parameter: {self.tapering_parameter} (%)\n')
            file.write(f'Time Bandwidth: {self.time_bandwidth}\n')
            file.write(f'Initial Phase Shift: {self.initial_phase_shift} (°)\n')
            file.write('-------------------------------------------------------------------------\n')
            file.write(
                f'For t = waiting_time = {self.waiting_time}s write: {"Deactivate" if self.tapering_parameter == 0 else 0}\n')
            file.write(
                f'For the first segment duration = {round(self.segment_duration_1, 3)}s write: {self.zone2_equation}\n')
            file.write(
                f'For the second segment duration = {round(self.segment_duration_2, 3)}s write: {self.zone3_equation}\n')
            file.write(
                f'For the third segment duration = {round(self.segment_duration_3, 3)}s write: {self.zone4_equation}\n')
            file.write('--------------------------------------------------------------------------\n')

