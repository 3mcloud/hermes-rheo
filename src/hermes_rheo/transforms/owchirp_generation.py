import math
import os

import matplotlib.pyplot as plt
import numpy as np


class OWChirpGeneration:
    """Generate and visualize an optimally windowed chirp (OWCh) waveform.

    The waveform is an exponential chirp with a cosine-squared taper, as
    described by Geri, Keshavarz and McKinley (see References). It is
    meant to be entered piecewise into a rheometer's arbitrary-waveform
    editor.

    Parameters
    ----------
    output_file_name : str
        Name of the output file.
    waiting_time : float
        Delay before the chirp starts, in s.
    signal_duration : float
        Chirp length, in s.
    signal_amplitude : float
        Amplitude of the imposed strain or stress, in rheometer units.
    initial_frequency : float
        Initial frequency, in rad/s.
    final_frequency : float
        Final frequency, in rad/s.
    tapering_parameter : float
        Taper fraction r of the signal duration (0 disables tapering).
    initial_phase_shift : float
        Initial phase, in degrees.

    References
    ----------
    Geri, M., Keshavarz, B., Divoux, T., Clasen, C., Curtis, D. J. and McKinley, G. H., *Time-resolved mechanical spectroscopy of soft materials via optimally windowed chirps*, Phys. Rev. X **8**, 041042 (2018). DOI: 10.1103/PhysRevX.8.041042.
    Perego, A., Vadillo, D. C., Mills, M. J. L., Das, M. and McKinley, G. H., *Evaluation of optimally windowed chirp signals in industrial rheological measurements: method development and data processing*, Rheologica Acta **64**, 391–406 (2025). DOI: 10.1007/s00397-025-01511-0.
    """

    def __init__(
        self,
        output_file_name: str,
        waiting_time: float,
        signal_duration: float,
        signal_amplitude: float,
        initial_frequency: float,
        final_frequency: float,
        tapering_parameter: float,
        initial_phase_shift: float,
    ):
        self.output_file_name = output_file_name
        self.waiting_time = waiting_time
        self.signal_duration = signal_duration
        self.signal_amplitude = signal_amplitude
        self.initial_frequency = initial_frequency
        self.final_frequency = final_frequency
        self.tapering_parameter = tapering_parameter
        self.initial_phase_shift = initial_phase_shift
        self.time_bandwidth = round(
            signal_duration * (final_frequency - initial_frequency) / (2 * math.pi), 0
        )
        self.initial_phase_shift_radians = round(initial_phase_shift * math.pi / 180, 4)
        self.shorter_chirp_signal = round(2 * 2 * math.pi / (3 * initial_frequency), 4)

        self._precompute_constants()

    def _precompute_constants(self):
        """Precompute constants used in the chirp equations."""
        self.exponential_growth_rate = round(
            math.log(self.final_frequency / self.initial_frequency) / self.signal_duration, 4
        )
        self.w0_divided_by_alpha = round(self.initial_frequency / self.exponential_growth_rate, 4)
        self.w0_exp_waiting_time = round(
            self.w0_divided_by_alpha / math.exp(self.exponential_growth_rate * self.waiting_time), 4
        )

        if self.tapering_parameter != 0:
            self.pi_over_taper = round(
                math.pi / (self.tapering_parameter * self.signal_duration), 4
            )
            self.phase_shift_c1 = round(-self.pi_over_taper * self.waiting_time - math.pi / 2, 4)
            self.phase_shift_c2 = round(
                -self.pi_over_taper * self.waiting_time
                - math.pi / self.tapering_parameter
                + math.pi / 2,
                4,
            )
        else:
            self.pi_over_taper = 0
            self.phase_shift_c1 = 0
            self.phase_shift_c2 = 0

        if self.tapering_parameter in (0, 1):
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
            self.segment_duration_3 = (
                self.signal_duration - self.segment_duration_1 - self.segment_duration_2
            )

        self._define_zone_equations()

    def _define_zone_equations(self):
        """Build the waveform expressions for the rheometer waveform editor."""
        if self.initial_phase_shift_radians < 0:
            if self.tapering_parameter < 1:
                self.zone2_equation = f"{self.signal_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c1})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})"
                self.zone3_equation = f"{self.signal_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})"
                self.zone4_equation = f"{self.signal_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c2})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})"
            else:
                self.zone2_equation = f"{self.signal_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}-{abs(self.initial_phase_shift_radians)})"
                self.zone3_equation = self.zone2_equation
                self.zone4_equation = self.zone2_equation
        else:
            if self.tapering_parameter < 1:
                self.zone2_equation = f"{self.signal_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c1})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})"
                self.zone3_equation = f"{self.signal_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})"
                self.zone4_equation = f"{self.signal_amplitude}*cos({self.pi_over_taper}*t {self.phase_shift_c2})^2*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})"
            else:
                self.zone2_equation = f"{self.signal_amplitude}*sin({self.w0_exp_waiting_time}*exp({self.exponential_growth_rate}*t)-{self.w0_divided_by_alpha}+{self.initial_phase_shift_radians})"
                self.zone3_equation = self.zone2_equation
                self.zone4_equation = self.zone2_equation

    def _chirp_function(self, time: float) -> float:
        """Return the waveform at ``time``, or None outside the defined segments."""
        if self.tapering_parameter == 0:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time <= self.signal_duration + self.waiting_time:
                return self.signal_amplitude * math.sin(
                    self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                    - self.w0_divided_by_alpha
                    + self.initial_phase_shift_radians
                )
        elif self.tapering_parameter == 1:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time <= self.segment_duration_1 + self.waiting_time:
                return (
                    self.signal_amplitude
                    * (math.cos(self.pi_over_taper * time + self.phase_shift_c1)) ** 2
                    * math.sin(
                        self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                        - self.w0_divided_by_alpha
                        + self.initial_phase_shift_radians
                    )
                )
        elif 0 < self.tapering_parameter < 1:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time <= self.segment_duration_1 + self.waiting_time:
                return (
                    self.signal_amplitude
                    * (math.cos(self.pi_over_taper * time + self.phase_shift_c1)) ** 2
                    * math.sin(
                        self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                        - self.w0_divided_by_alpha
                        + self.initial_phase_shift_radians
                    )
                )
            elif (
                self.segment_duration_1 + self.waiting_time
                < time
                <= self.segment_duration_1 + self.segment_duration_2 + self.waiting_time
            ):
                return self.signal_amplitude * math.sin(
                    self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                    - self.w0_divided_by_alpha
                    + self.initial_phase_shift_radians
                )
            elif time > self.segment_duration_1 + self.segment_duration_2 + self.waiting_time:
                return (
                    self.signal_amplitude
                    * (math.cos(self.pi_over_taper * time + self.phase_shift_c2)) ** 2
                    * math.sin(
                        self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                        - self.w0_divided_by_alpha
                        + self.initial_phase_shift_radians
                    )
                )
        elif self.tapering_parameter > 1:
            if time < self.waiting_time:
                return 0
            elif self.waiting_time < time < self.segment_duration_1 + self.waiting_time:
                return (
                    self.signal_amplitude
                    * (math.cos(self.pi_over_taper * time + self.phase_shift_c1)) ** 2
                    * math.sin(
                        self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                        - self.w0_divided_by_alpha
                        + self.initial_phase_shift_radians
                    )
                )
            elif (
                self.segment_duration_1 + self.waiting_time
                <= time
                <= self.segment_duration_1 + self.segment_duration_2 + self.waiting_time
            ):
                return (
                    self.signal_amplitude
                    * (math.cos(self.pi_over_taper * time + self.phase_shift_c2)) ** 2
                    * math.sin(
                        self.w0_exp_waiting_time * math.exp(self.exponential_growth_rate * time)
                        - self.w0_divided_by_alpha
                        + self.initial_phase_shift_radians
                    )
                )

    def visualize(self, display_r0=False, filepath=None, y_axis="strain"):
        """Plot the waveform and save its data and figure.

        Parameters
        ----------
        display_r0 : bool
            Overlay the untapered (r=0) chirp.
        filepath : str
            Folder where the waveform data and plot are saved.
        y_axis : {"strain", "stress"}
            Imposed quantity; sets only the axis label and data-file header.
        """
        if y_axis not in ("strain", "stress"):
            raise ValueError(f"y_axis must be 'strain' or 'stress', got {y_axis!r}")

        time_values = np.arange(0, self.signal_duration + self.waiting_time, 0.0001)
        strain_values = list(map(self._chirp_function, time_values))

        valid_data = [(t, s) for t, s in zip(time_values, strain_values) if s is not None]

        if valid_data:
            time_values_filtered, strain_values_filtered = zip(*valid_data)

            txt_file_path = os.path.join(filepath, "owchirp_waveform_xy.txt")
            np.savetxt(
                txt_file_path,
                np.column_stack((time_values_filtered, strain_values_filtered)),
                header=f"Time(s)\t{y_axis.capitalize()}",
                fmt="%.5e",
                delimiter="\t",
            )

            plt.plot(time_values_filtered, strain_values_filtered, "b-", label="OWChirp Signal")

        if display_r0:
            original_tapering_parameter = self.tapering_parameter
            self.tapering_parameter = 0
            self._precompute_constants()
            r0_strain_values = list(map(self._chirp_function, time_values))
            plt.plot(
                time_values,
                r0_strain_values,
                color="gray",
                linestyle="--",
                label="Chirp Signal (r=0)",
                zorder=-1,
            )
            self.tapering_parameter = original_tapering_parameter
            self._precompute_constants()

        plt.xlabel("Time (s)")
        plt.ylabel(y_axis.capitalize())
        plt.axhline(0, linestyle="--", color="black")
        plt.axvline(self.waiting_time, linestyle="--", color="black")
        plt.legend(loc="best")

        png_file_path = os.path.join(filepath, "owchirp_waveform_xy.png")
        plt.savefig(png_file_path, dpi=100)

        plt.show()

    def print_parameters(self):
        """Print the chirp parameters and waveform expressions."""
        print(f"Shorter chirp signal: {self.shorter_chirp_signal} (s), TB = {self.time_bandwidth}")
        print(f"Total length of chirp signal: {self.signal_duration} (s)")
        print(f"Initial Frequency: {self.initial_frequency} (rad/s)")
        print(f"Final Frequency: {self.final_frequency} (rad/s)")
        print(f"Signal amplitude: {self.signal_amplitude} ")
        print(f"Waiting time: {self.waiting_time} (s)")
        print(f"Tapering parameter: {self.tapering_parameter} ")
        print(f"Time Bandwidth: {self.time_bandwidth}")
        print(f"Initial Phase Shift: {self.initial_phase_shift} (°)")
        print("-------------------------------------------------------------------------")
        print(
            f"For t = waiting_time = {self.waiting_time}s write: {'Deactivate' if self.tapering_parameter == 0 else 0}"
        )
        print(
            f"For the first segment duration = {round(self.segment_duration_1, 3)}s write: {self.zone2_equation}"
        )
        print(
            f"For the second segment duration = {round(self.segment_duration_2, 3)}s write: {self.zone3_equation}"
        )
        print(
            f"For the third segment duration = {round(self.segment_duration_3, 3)}s write: {self.zone4_equation}"
        )
        print("--------------------------------------------------------------------------")

    def to_txt(self, filepath):
        """Write the chirp parameters and waveform expressions to ``owchirp_parameters.txt`` in ``filepath``."""
        output_file = os.path.join(filepath, "owchirp_parameters.txt")

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(
                f"Shorter chirp signal: {self.shorter_chirp_signal} (s), TB = {self.time_bandwidth}\n"
            )
            file.write(f"Total length of chirp signal: {self.signal_duration} (s)\n")
            file.write(f"Initial Frequency: {self.initial_frequency} (rad/s)\n")
            file.write(f"Final Frequency: {self.final_frequency} (rad/s)\n")
            file.write(f"Signal amplitude: {self.signal_amplitude}\n")
            file.write(f"Waiting time: {self.waiting_time} (s)\n")
            file.write(f"Tapering parameter: {self.tapering_parameter} (%)\n")
            file.write(f"Time Bandwidth: {self.time_bandwidth}\n")
            file.write(f"Initial Phase Shift: {self.initial_phase_shift} (°)\n")
            file.write(
                "-------------------------------------------------------------------------\n"
            )
            file.write(
                f"For t = waiting_time = {self.waiting_time}s write: {'Deactivate' if self.tapering_parameter == 0 else 0}\n"
            )
            file.write(
                f"For the first segment duration = {round(self.segment_duration_1, 3)}s write: {self.zone2_equation}\n"
            )
            file.write(
                f"For the second segment duration = {round(self.segment_duration_2, 3)}s write: {self.zone3_equation}\n"
            )
            file.write(
                f"For the third segment duration = {round(self.segment_duration_3, 3)}s write: {self.zone4_equation}\n"
            )
            file.write(
                "--------------------------------------------------------------------------\n"
            )
