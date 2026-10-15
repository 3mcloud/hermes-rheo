"""OWChirp analysis for transient rheological data.

This module implements the *Optimally Windowed Chirp* (OWChirp) time-resolved
mechanical spectroscopy technique of Geri, Keshavarz, Divoux, Clasen, Curtis
and McKinley [Geri2018]_.  A chirp is a broad-band strain (or stress) signal
whose instantaneous frequency sweeps exponentially from ``omega_lo`` to
``omega_hi`` over a short window, allowing the full linear viscoelastic
spectrum to be measured in a fraction of a second.

The class :class:`OWChirpAnalysis` is a
:class:`~piblin.transform.abc.measurement_transform.MeasurementTransform`
that consumes a measurement carrying either:

* An **Arbitrary Wave** step (a TRIOS-programmed chirp using the arbitrary-wave
  generator, i.e. TA Instruments ARES G2), or
* A **Frequency Chirp** step (TRIOS's built-in OWChirp  on newer instruments and software
  releases, e.g. TA Instruments ARES G3).

References
----------
.. [Geri2018] Geri, M., Keshavarz, B., Divoux, T., Clasen, C., Curtis, D. J.
   and McKinley, G. H., *Time-Resolved Mechanical Spectroscopy of Soft
   Materials via Optimally Windowed Chirps*, Phys. Rev. X 8, 041042 (2018).
   DOI: `10.1103/PhysRevX.8.041042
   <https://doi.org/10.1103/PhysRevX.8.041042>`_ (open access) —
   preprint: `arXiv:1804.03061 <https://arxiv.org/abs/1804.03061>`_.
.. [Perego2025] Perego, A. *et al.*, *Evaluation of Optimally Windowed
   Chirp Signals in Industrial Rheological Measurements: Method
   Development and Data Processing*, Rheologica Acta (2025).
   DOI: `10.1007/s00397-025-01511-0
   <https://doi.org/10.1007/s00397-025-01511-0>`_.  Application of
   ``hermes-rheo`` to time-resolved OWChirp analysis.

"""

from __future__ import annotations

import copy
import logging
import math
import re
from typing import Any

import numpy as np
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.transform.abc.measurement_transform import MeasurementTransform
from scipy.fftpack import fft, fftshift

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit-prefix table used by :meth:`OWChirpAnalysis.parse_value_with_prefix`
# ---------------------------------------------------------------------------
_UNIT_PREFIXES: dict[str, float] = {
    "m": 1e-3,
    "µ": 1e-6,
    "u": 1e-6,  # ASCII substitute for µ
    "n": 1e-9,
    "k": 1e3,
    "M": 1e6,
}


# ---------------------------------------------------------------------------
# Method-string canonicalisation (shared with the reader's per-method defaults)
# ---------------------------------------------------------------------------


def _canonicalise_method(method_line: str) -> str:
    """Return the canonical, lower-cased form of a TRIOS step-name line.

    Parameters
    ----------
    method_line : str
        Raw value from ``measurement.conditions["method"]``.

    Returns
    -------
    str
        Lower-cased method name with ``"Step name\\t"`` prefix, ``" Repeat"``
        tag, and trailing ``- N`` counter removed.

    Examples
    --------
    >>> _canonicalise_method("Step name\\tArbitrary Wave - 3")
    'arbitrary wave'
    >>> _canonicalise_method("Frequency Chirp Repeat - 1")
    'frequency chirp'
    """
    s = str(method_line).strip()
    if s.lower().startswith("step name"):
        parts = s.split("\t", 1)
        if len(parts) == 2:
            s = parts[1]
    s = s.replace(" Repeat", "")
    s = re.sub(r"\s*-\s*\d+\s*$", "", s)
    return s.strip().lower()


# ---------------------------------------------------------------------------
# OWChirpAnalysis
# ---------------------------------------------------------------------------


class OWChirpAnalysis(MeasurementTransform):
    """Extract time-resolved moduli from an OWChirp (Arbitrary Wave / Frequency Chirp) step.

    Consumes a
    :class:`~piblin.data.data_collections.measurement.Measurement`
    whose ``conditions["method"]`` contains ``"Arbitrary Wave"`` or
    ``"Frequency Chirp"`` (Geri *et al.* [1]_; Perego *et al.* [2]_).  The raw time-domain strain/stress signal is:

    1. Optionally trimmed to the region of interest (``owchirp_waiting_time``).
    2. Baseline-corrected using one of several candidate filters
       (``filter_criterion`` picks the best automatically).
    3. Zero-padded and Fourier-transformed to yield the complex modulus
       :math:`G^{*}(\\omega) = \\tilde{\\sigma}(\\omega) / \\tilde{\\gamma}(\\omega)`.
    4. Split into storage :math:`G'(\\omega)` and loss :math:`G''(\\omega)`
       moduli, along with :math:`|G^{*}|`, ``tan(delta)`` and the complex
       viscosity :math:`|\\eta^{*}| = |G^{*}| / \\omega`.

    The results are added as a **frequency-domain**
    :class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
    stored at index 0 of the target's ``datasets``.  The original raw dataset
    concatenated with the baseline-corrected time series is stored at index 1.

    Parameters
    ----------
    owchirp_waiting_time : {"before_signal", "after_signal", "all_signal", \
                           "inside_signal", "before_and_after"}, default \
                          "before_signal"
        Region of the raw signal to feed to the FFT.  ``"before_signal"``
        drops the pre-chirp waiting window; ``"inside_signal"`` keeps only
        the chirp block; ``"all_signal"`` uses everything.
    cutoff_points : float, default 0
        Drop all raw samples with ``step_time`` less than this value before
        further processing.  Useful when the very early samples are
        contaminated by instrument transients.
    filter_criterion : {"symmetry", "endpoints"}, default "endpoints"
        Score used by :meth:`select_best_filter_method` to pick between
        the ``tw``, ``chirp``, ``all`` and ``none`` baseline-subtraction
        strategies.

    Notes
    -----
    Stress-controlled instruments (TA DHR, serial numbers: ``5343`` / ``5332``)
    require the inertia-compensated variant
    :meth:`moduli_owchirp_stress_controlled_IC`.  This is used
    automatically for **Frequency Chirp** steps, which carry the
    geometry/instrument constants in the file header.  For **Arbitrary
    Wave** steps the constants are not exported so the strain-controlled
    variant is always used; users on a DHR are encouraged to switch to the
    Frequency Chirp macro for accurate stress-controlled results.

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
    >>> mset = TriosRheoReader().data_from_filepath(
    ...     "owchirp_experiment.txt", create_composite_datasets=True
    ... )  # doctest: +SKIP
    >>> processed = OWChirpAnalysis()(mset)  # doctest: +SKIP
    >>> fspace = processed.measurements[0].datasets[0]  # doctest: +SKIP
    >>> fspace.independent_variable_name, fspace.dependent_variable_name  # doctest: +SKIP
    ('angular frequency', 'storage modulus')

    References
    ----------
    .. [1] Geri, M. *et al.*, *Time-Resolved Mechanical Spectroscopy of Soft
       Materials via Optimally Windowed Chirps*, Phys. Rev. X 8, 041042 (2018).
       DOI: `10.1103/PhysRevX.8.041042
       <https://doi.org/10.1103/PhysRevX.8.041042>`_ (open access) —
       preprint: `arXiv:1804.03061 <https://arxiv.org/abs/1804.03061>`_.
    .. [2] Perego, A. *et al.*, *Evaluation of Optimally Windowed Chirp
       Signals in Industrial Rheological Measurements: Method Development
       and Data Processing*, Rheologica Acta (2025).
       DOI: `10.1007/s00397-025-01511-0
       <https://doi.org/10.1007/s00397-025-01511-0>`_.
    """

    def __init__(
        self,
        owchirp_waiting_time: str = "before_signal",
        cutoff_points: float = 0,
        filter_criterion: str = "endpoints",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.owchirp_waiting_time = owchirp_waiting_time
        self.cutoff_points = cutoff_points
        self.filter_criterion = filter_criterion

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_value_with_prefix(text: str | None) -> float:
        """Parse a numeric string with an SI unit prefix into SI base units.

        Handles inputs like ``"2.02 mrad/N.m"``, ``"1.85047 µN.m.s²"``,
        ``"0.276709 µN.m/(rad/s)"`` and ``"4.0 1/rad"``.  Only the numeric
        magnitude and the immediately-following unit prefix are examined; the
        rest of the unit string is ignored.

        Parameters
        ----------
        text : str or None
            The raw string to parse.  ``None`` returns ``nan``.

        Returns
        -------
        float
            The numeric magnitude scaled by the SI multiplier of the detected
            prefix (or unscaled if no prefix is present).

        Examples
        --------
        >>> OWChirpAnalysis.parse_value_with_prefix('2.5 mrad')
        0.0025
        >>> OWChirpAnalysis.parse_value_with_prefix('1e3 N.m')
        1000.0
        """
        if text is None:
            return float("nan")

        s = str(text).strip()

        # Extract the numeric part (support optional exponent)
        num_match = re.match(r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)", s)
        if not num_match:
            return float("nan")
        value = float(num_match.group(1))

        # Look for a prefix immediately before the first alphabetic unit
        # character.
        unit_part = s[num_match.end() :].strip().replace(" ", "")
        m = re.match(r"([mµunkM])([A-Za-z])", unit_part)
        multiplier = 1.0
        if m:
            prefix = m.group(1)
            if prefix in _UNIT_PREFIXES:
                multiplier = _UNIT_PREFIXES[prefix]

        return value * multiplier

    @staticmethod
    def prepare_owchirp_data(
        dataset: one_d_composite_dataset.OneDimensionalCompositeDataset,
        time: str,
        strain: str,
        stress: str,
        temperature: str,
        cutoff_points: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract time, strain, stress and temperature arrays from a raw dataset.

        Switches the composite dataset's coordinates in turn to pull each
        variable out, converting strain to unitless (from ``%``) and stress
        to Pa (from ``MPa``) when needed.  Samples with ``time <
        cutoff_points`` are discarded.

        Parameters
        ----------
        dataset : OneDimensionalCompositeDataset
            The raw time-domain OWChirp dataset.
        time, strain, stress, temperature : str
            Names of the columns to pull.
        cutoff_points : float
            Time (in seconds) below which samples are discarded.

        Returns
        -------
        time_values, strain_values, stress_values, temperature_values : np.ndarray
            The four aligned arrays starting from the cutoff index.

        Raises
        ------
        ValueError
            If *cutoff_points* exceeds the maximum time value or an unknown
            stress unit is encountered.
        """
        dataset.switch_coordinates(independent_name=time, dependent_name=strain)
        time_values = dataset.x_values

        if cutoff_points > time_values.max():
            raise ValueError(
                f"Cutoff point {cutoff_points} is greater than the maximum "
                f"time value {time_values.max()}."
            )

        cutoff_index = (np.abs(time_values - cutoff_points)).argmin()

        if dataset.y_unit == "%":
            strain_values = copy.deepcopy(dataset.y_values[cutoff_index:]) * 0.01
            dataset.y_unit = "unitless"
        else:
            strain_values = copy.deepcopy(dataset.y_values[cutoff_index:])

        if strain_values is None:
            raise ValueError("Strain values not set after switching coordinates to strain.")

        dataset.switch_coordinates(independent_name=time, dependent_name=stress)

        if dataset.y_unit == "MPa":
            stress_values = copy.deepcopy(dataset.y_values[cutoff_index:]) * 1e6
            dataset.y_unit = "Pa"
        elif dataset.y_unit == "Pa":
            stress_values = copy.deepcopy(dataset.y_values[cutoff_index:])
        else:
            raise ValueError(f"Unsupported stress unit '{dataset.y_unit}'. Expected 'Pa' or 'MPa'.")

        if stress_values is None:
            raise ValueError("Stress values not set after switching coordinates to stress.")

        dataset.switch_coordinates(independent_name=time, dependent_name=temperature)
        temperature_values = copy.deepcopy(dataset.y_values[cutoff_index:])

        temperature_values = np.asarray(temperature_values)
        time_values = np.asarray(time_values[cutoff_index:])

        return (
            time_values,
            strain_values,
            stress_values,
            temperature_values,
        )

    # ------------------------------------------------------------------
    # Baseline-subtraction
    # ------------------------------------------------------------------

    @staticmethod
    def _find_tw_index(t_w: float, time: np.ndarray) -> int:
        """Return the largest index i with ``time[i] <= t_w`` (raises IndexError otherwise)."""
        time = np.asarray(time)
        try:
            return int(np.where(time <= t_w)[0][-1])
        except IndexError as exc:
            raise IndexError(
                f"No values in the time array are less than or equal to t_w = {t_w}"
            ) from exc

    @staticmethod
    def average_until_tw(t_w: float, time: list[float], signal: list[float]) -> np.ndarray:
        """Subtract the mean of *signal* over ``time <= t_w`` from *signal*.

        Best default for the *strain* channel: the pre-chirp waiting window
        should be zero, so removing its mean corrects for any residual bias.
        """
        signal = np.asarray(signal)
        index_tw = OWChirpAnalysis._find_tw_index(t_w, time)
        initial_value = np.mean(signal[: index_tw + 1])
        return signal - initial_value

    @staticmethod
    def average_over_chirp(t_w: float, time: list[float], signal: list[float]) -> np.ndarray:
        """Subtract the mean of *signal* AFTER the waiting window.

        Best default for the stress controlled instruments.
        """
        signal = np.asarray(signal)
        index_tw = OWChirpAnalysis._find_tw_index(t_w, time)
        avg_signal = np.mean(signal[index_tw + 1 :])
        return signal - avg_signal

    @staticmethod
    def average_over_all(t_w: float, time: list[float], signal: list[float]) -> np.ndarray:
        """Subtract the sum of pre-``t_w`` and post-``t_w`` means from *signal*."""
        signal = np.asarray(signal)
        index_tw = OWChirpAnalysis._find_tw_index(t_w, time)
        return signal - np.mean(signal[: index_tw + 1]) - np.mean(signal[index_tw + 1 :])

    def filter_signal(
        self,
        t_w: float,
        time: list[float],
        signal: list[float],
        method: str = "tw",
    ) -> np.ndarray | None:
        """Return *signal* baseline-corrected with the chosen *method*.

        Parameters
        ----------
        t_w : float
            Waiting time before the chirp begins (seconds).
        time, signal : list of float
            The raw time and signal arrays.
        method : {"tw", "chirp", "all", "none"}, default "tw"
            ``"tw"`` — subtract mean over ``time <= t_w``.
            ``"chirp"`` — subtract mean over ``time > t_w``.
            ``"all"`` — subtract both means.
            ``"none"`` — return the signal unchanged.

        Returns
        -------
        np.ndarray or None
            The corrected signal, or ``None`` when *method* is invalid.
        """
        if method == "tw":
            return self.average_until_tw(t_w, time, signal)
        if method == "chirp":
            return self.average_over_chirp(t_w, time, signal)
        if method == "all":
            return self.average_over_all(t_w, time, signal)
        if method in ("none", "None"):
            return np.asarray(signal)
        _logger.warning("Invalid filter method %r; choose 'tw', 'chirp', 'all' or 'none'.", method)
        return None

    def select_best_filter_method(
        self,
        t_w: float,
        time: list[float],
        signal: list[float],
        criterion: str = "symmetry",
    ) -> tuple[np.ndarray, str]:
        """Try every baseline filter and return the one that best matches *criterion*.

        Parameters
        ----------
        t_w, time, signal :
            See :meth:`filter_signal`.
        criterion : {"symmetry", "endpoints"}
            ``"symmetry"`` scores by :math:`\\sum |signal|` (smaller is
            better).  ``"endpoints"`` scores by
            :math:`(|s_0| + |s_{-1}|)/2 + \\sigma_s`.

        Returns
        -------
        best_corrected_signal : np.ndarray
        best_method : str
            One of ``"tw"``, ``"chirp"``, ``"all"``, ``"none"``.

        Raises
        ------
        ValueError
            If *criterion* is not one of the accepted values.
        """
        valid_criteria = ("symmetry", "endpoints")
        if criterion not in valid_criteria:
            raise ValueError(f"Invalid criterion {criterion!r}. Must be one of {valid_criteria}.")

        methods = ("tw", "chirp", "all", "none")
        best_method = None
        min_score = float("inf")

        for method in methods:
            corrected_signal = self.filter_signal(t_w, time, signal, method)
            if corrected_signal is None:
                continue

            if criterion == "symmetry":
                score = np.sum(np.abs(corrected_signal))
            else:  # endpoints
                start_end_avg = (abs(corrected_signal[0]) + abs(corrected_signal[-1])) / 2
                std_dev = np.std(corrected_signal)
                score = start_end_avg + std_dev

            if score < min_score:
                min_score = score
                best_method = method

        best_corrected_signal = self.filter_signal(t_w, time, signal, best_method)
        return best_corrected_signal, best_method

    # ------------------------------------------------------------------
    # Wave-parameter parsing (Arbitrary Wave)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_wave_parameters(
        wave_data: dict[str, dict],
    ) -> tuple[float, float, float, float, float]:
        """Extract ``(tw, T, r, w0, w1)`` from a TRIOS Arbitrary Wave block.

        Parameters
        ----------
        wave_data : dict
            The per-step dict stored in ``target.details["Arbitrary Wave - N"]``.
            Contains sub-dicts ``wave 1``, ``wave 2``, ... each with keys
            ``"duration (s)"`` and ``"coef"``.

        Returns
        -------
        tw : float
            Waiting time before the chirp starts.
        T : float
            Duration of the chirp signal (seconds).
        r : float
            Tapering parameter (0 = no tapering).
        w0 : float
            Initial (lowest) angular frequency (rad/s).
        w1 : float
            Final (highest) angular frequency (rad/s).
        """
        duration_values = [wave_data[f"wave {i}"]["duration (s)"] for i in range(1, len(wave_data))]
        coef_values = [wave_data[f"wave {i}"]["coef"] for i in range(1, len(wave_data))]

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
            if coef_values[0] == [0.0]:
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

        w0 = round(alpha * w0_alpha, 3)
        w1 = round(w0 * math.exp(alpha * T), 3)
        return tw, T, r, w0, w1

    # ------------------------------------------------------------------
    # FFT moduli extraction
    # ------------------------------------------------------------------

    @staticmethod
    def moduli_owchirp_strain_controlled(
        strain: np.ndarray,
        stress: np.ndarray,
        fs: float,
        om1: float,
        om2: float,
    ) -> tuple[np.ndarray, ...]:
        """Compute :math:`G'`, :math:`G''` and :math:`G^{*}` from strain-controlled data.

        Both signals are zero-padded to ``n = 2**(floor(log2(N)) + 1)`` points
        (the next power of two above the signal length, for an efficient
        radix-2 FFT) and the complex modulus is taken as the ratio of their
        transforms in the frequency band ``[om1, om2]``:

        .. math:: G^{*}(\\omega) = \\tilde{\\sigma}(\\omega) / \\tilde{\\gamma}(\\omega)

        Parameters
        ----------
        strain, stress : np.ndarray
            The baseline-corrected time-domain signals (units: unitless / Pa).
        fs : float
            Sampling frequency (samples per second).
        om1, om2 : float
            Lower and upper angular-frequency bounds of the analysis band
            (rad/s).

        Returns
        -------
        frequency : np.ndarray
            Angular frequencies in the analysis band (rad/s).
        strain_final, stress_final : np.ndarray
            Fourier coefficients of strain and stress in the analysis band.
        elastic_modulus_final : np.ndarray
            The storage modulus :math:`G'(\\omega)` (Pa).
        loss_modulus_final : np.ndarray
            The loss modulus :math:`G''(\\omega)` (Pa).
        complex_modulus : np.ndarray
            The complex modulus :math:`G^{*}(\\omega)`.

        References
        ----------
        Adapted from the MITOWCh MATLAB code by Geri et al. (2019).
        """
        f1 = om1 / (2 * np.pi)
        f2 = om2 / (2 * np.pi)
        m = len(strain)
        n = 2 ** (int(np.log2(m)) + 1)

        strain_padded = np.pad(strain, (0, n - m), "constant")
        stress_padded = np.pad(stress, (0, n - m), "constant")

        strain_fft = fft(strain_padded, n)
        stress_ftt = fft(stress_padded, n)

        f = (-n / 2 + np.arange(n)) * (fs / n)
        shifted_strain = fftshift(strain_fft)
        shifted_stress = fftshift(stress_ftt)

        moduli = shifted_stress[int(n / 2) :] / shifted_strain[int(n / 2) :]
        frequency_range = f[int(n / 2) :]
        upper_frequency = frequency_range[frequency_range <= f2]
        lower_frequency = upper_frequency[upper_frequency >= f1]
        frequency = 2 * np.pi * lower_frequency
        if np.size(frequency) == 1:
            frequency = frequency[np.newaxis]
        l1 = len(upper_frequency)
        l2 = len(lower_frequency)
        elastic_modulus = np.real(moduli)
        loss_modulus = np.imag(moduli)
        complex_modulus = moduli[l1 - l2 : l1]
        elastic_modulus_final = elastic_modulus[l1 - l2 : l1]
        loss_modulus_final = loss_modulus[l1 - l2 : l1]
        if np.size(complex_modulus) == 1:
            complex_modulus = complex_modulus[np.newaxis]
        if np.size(elastic_modulus_final) == 1:
            elastic_modulus_final = elastic_modulus_final[np.newaxis]
        if np.size(loss_modulus_final) == 1:
            loss_modulus_final = loss_modulus_final[np.newaxis]

        stress_corrected = shifted_stress[int(n / 2) :] / fs
        strain_corrected = shifted_strain[int(n / 2) :] / fs
        stress_final = stress_corrected[l1 - l2 : l1]
        strain_final = strain_corrected[l1 - l2 : l1]
        if np.size(stress_final) == 1:
            stress_final = stress_final[np.newaxis]
        if np.size(strain_final) == 1:
            strain_final = strain_final[np.newaxis]

        return (
            frequency,
            strain_final,
            stress_final,
            elastic_modulus_final,
            loss_modulus_final,
            complex_modulus,
        )

    @staticmethod
    def moduli_owchirp_stress_controlled_IC(
        time: np.ndarray,
        strain: np.ndarray,
        stress: np.ndarray,
        fs: float,
        om1: float,
        om2: float,
        K_stress: float,
        K_strain: float,
        I: float,
        c: float,
        mu: float,
    ) -> tuple[np.ndarray, ...]:
        r"""Inertia- and compliance-corrected stress-controlled OWChirp moduli.

        For stress-controlled instruments (e.g. TA instrument DHR series) the strain
        differentiation and system-level corrections must be applied to
        recover the linear viscoelastic moduli:

        .. math::

           K_{stiff}(\omega) = \frac{\tilde{\sigma}(\omega)}{\tilde{\dot{\gamma}}(\omega)}
                              \cdot j\omega \cdot \frac{K_{strain}}{K_{stress}}

           K'_{el} = \operatorname{Re}(K_{stiff}) + I\omega^{2}, \quad
           K'_{visc} = \operatorname{Im}(K_{stiff}) - \mu \omega

           G'(\omega) = \frac{K_{stress}}{K_{strain}}
              \cdot \frac{K'_{el}(1 - c K'_{el}) - c (K'_{visc})^{2}}
                         {(1 - c K'_{el})^{2} + (c K'_{visc})^{2}}

        Parameters
        ----------
        time : np.ndarray
            Time array (s) used to compute the strain rate.
        strain : np.ndarray
            Strain array (unitless).
        stress : np.ndarray
            Stress array (Pa).
        fs : float
            Sampling frequency (Hz).
        om1, om2 : float
            Angular-frequency (rad/s).
        K_stress : float
            Torque-to-stress conversion factor.
        K_strain : float
            Displacement-to-strain conversion factor.
        I : float
            Total system inertia (instrument + geometry).
        c : float
            Geometry compliance coefficient.
        mu : float
            Bearing-friction coefficient.

        Returns
        -------
        frequency, strain_final, stress_final : np.ndarray
        elastic_modulus_final, loss_modulus_final : np.ndarray
        complex_modulus, complex_viscosity : np.ndarray

        References
        ----------
        Adapted from the MITOWCh MATLAB code by Geri et al. (2019).
        """
        strain_rate = np.diff(strain) / np.diff(time)

        f1 = om1 / (2 * np.pi)
        f2 = om2 / (2 * np.pi)
        m = len(strain_rate)
        n = 2 ** (int(np.log2(m)) + 1)

        strain_rate_padded = np.pad(strain_rate, (0, n - m), "constant")
        stress_padded = np.pad(stress, (0, n - m), "constant")
        strain_padded = np.pad(strain, (0, n - m), "constant")

        stress_padded_fft = fft(stress_padded, n)
        strain_padded_fft = fft(strain_padded, n)
        strain_rate_padded_fft = fft(strain_rate_padded, n)

        f = (-n / 2 + np.arange(n)) * (fs / n)
        shifted_strain_rate = fftshift(strain_rate_padded_fft)
        shifted_stress = fftshift(stress_padded_fft)
        shifted_strain = fftshift(strain_padded_fft)

        frequency_range = f[int(n / 2) :]
        upper_frequency = frequency_range[frequency_range <= f2]
        lower_frequency = upper_frequency[upper_frequency >= f1]
        frequency = 2 * np.pi * lower_frequency

        if np.size(frequency) == 1:
            frequency = frequency[np.newaxis]
        l1 = len(upper_frequency)
        l2 = len(lower_frequency)

        # Stiffness with inertia + friction correction, then modulus with
        # compliance correction.
        stiffness = (
            (shifted_stress[int(n / 2) :] / shifted_strain_rate[int(n / 2) :])
            * (1j * frequency_range * 2 * np.pi)
            * K_strain
            / K_stress
        )

        elastic_stiffness = np.real(stiffness) + I * (frequency_range * 2 * np.pi) ** 2
        loss_stiffness = np.imag(stiffness) - mu * (frequency_range * 2 * np.pi)

        elastic_modulus = (
            (K_stress / K_strain)
            * (elastic_stiffness * (1 - c * elastic_stiffness) - c * loss_stiffness**2)
            / ((1 - c * elastic_stiffness) ** 2 + (c * loss_stiffness) ** 2)
        )
        loss_modulus = (
            (K_stress / K_strain)
            * loss_stiffness
            / ((1 - c * elastic_stiffness) ** 2 + (c * loss_stiffness) ** 2)
        )

        elastic_modulus_final = elastic_modulus[l1 - l2 : l1]
        loss_modulus_final = loss_modulus[l1 - l2 : l1]
        complex_modulus = (elastic_modulus_final**2 + loss_modulus_final**2) ** 0.5
        complex_viscosity = complex_modulus / frequency

        stress_corrected = shifted_stress[int(n / 2) :] / fs
        strain_corrected = shifted_strain[int(n / 2) :] / fs
        stress_final = stress_corrected[l1 - l2 : l1]
        strain_final = strain_corrected[l1 - l2 : l1]
        if np.size(stress_final) == 1:
            stress_final = stress_final[np.newaxis]
        if np.size(strain_final) == 1:
            strain_final = strain_final[np.newaxis]

        return (
            frequency,
            strain_final,
            stress_final,
            elastic_modulus_final,
            loss_modulus_final,
            complex_modulus,
            complex_viscosity,
        )

    # ------------------------------------------------------------------
    # Frequency Chirp parameter extraction (structured + legacy fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frequency_chirp_params(details: dict[str, Any]) -> dict[str, float]:
        """Parse a Frequency Chirp step's parameters from a measurement's details dict.

        The preferred path uses the structured per-step dict written by newer
        TRIOS exports (``details["Frequency Chirp - N"]`` with keys
        ``waiting_time``, ``chirp_length``, ``f_lo_hz``, ``f_hi_hz``,
        ``tapering_percent``, ``fs``, ...).  Older exports encoded the same
        information in free-form step-text lines; the fallback path scrapes
        those with a series of regexes.

        Parameters
        ----------
        details : dict
            The ``target.details`` dict of a Frequency Chirp measurement.

        Returns
        -------
        dict
            Chirp parameters keyed by the names ``waiting_time``,
            ``chirp_length``, ``oscillation_period``, ``initial_frequency``
            (rad/s), ``final_frequency`` (rad/s), ``taping_parameter`` (%),
            ``sampling_frequency`` (pts/s), plus any extra step metadata and
            geometry / inertia parameters that were present.
        """
        import numpy as _np  # local alias to avoid confusion if needed

        def _to_float(val: Any) -> float:
            if val is None:
                return float("nan")
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip().replace("\u00a0", "").replace(" ", "")
            if s == "":
                return float("nan")
            if "," in s and "." in s:
                s = s.replace(".", "").replace(",", ".")
            elif "," in s:
                s = s.replace(",", ".")
            try:
                return float(s)
            except Exception:
                return float("nan")

        def _to_bool_flag(s: str) -> bool | None:
            if not isinstance(s, str):
                return None
            s_norm = s.strip().lower()
            if s_norm in {"on", "yes", "true"}:
                return True
            if s_norm in {"off", "no", "false"}:
                return False
            return None

        def _attach_geometry_metadata(result: dict[str, Any], details: dict[str, Any]) -> None:
            extra_keys = [
                "geometry_compliance",
                "geometry_inertia",
                "geometry_friction",
                "geometry_stress_constant",
                "instrument_inertia",
                "geometry_strain_constant",
            ]
            for key in extra_keys:
                if key in details and key not in result:
                    result[key] = details[key]

        # ---------- 1) Preferred: structured per-step dict(s) ----------
        step_items: list[tuple[str, dict[str, Any]]] = []
        for k, v in details.items():
            if isinstance(k, str) and re.search(
                r"^Frequency\s*Chirp\s*-\s*\d+\s*$", k, flags=re.IGNORECASE
            ):
                if isinstance(v, dict) and any(
                    x in v for x in ("waiting_time", "chirp_length", "f_lo_hz", "f_hi_hz")
                ):
                    step_items.append((k, v))

        if step_items:

            def _step_num(key: str) -> int:
                m = re.search(r"(\d+)\s*$", key)
                return int(m.group(1)) if m else -1

            step_items.sort(key=lambda kv: _step_num(kv[0]))
            _, step = step_items[-1]

            waiting_time = _to_float(step.get("waiting_time"))
            chirp_length = _to_float(step.get("chirp_length"))
            f_lo_hz = _to_float(step.get("f_lo_hz"))
            f_hi_hz = _to_float(step.get("f_hi_hz"))
            taper_pct = _to_float(step.get("tapering_percent"))
            fs = _to_float(step.get("fs"))

            result: dict[str, Any] = {
                "waiting_time": waiting_time,
                "chirp_length": chirp_length,
                "oscillation_period": chirp_length,
                "initial_frequency": 2 * _np.pi * f_lo_hz
                if _np.isfinite(f_lo_hz)
                else float("nan"),
                "final_frequency": 2 * _np.pi * f_hi_hz if _np.isfinite(f_hi_hz) else float("nan"),
                "taping_parameter": taper_pct,
                "sampling_frequency": fs,
            }
            for key, val in step.items():
                if key not in result:
                    result[key] = val
            _attach_geometry_metadata(result, details)
            return result

        # ---------- 2) Fallback: legacy regex scrape ----------
        blob_parts = [v for v in details.values() if isinstance(v, str)]
        blob = " ".join(blob_parts)

        m_baseline = re.search(r"Baseline\s*time\s*([\d.,]+)\s*s", blob, flags=re.IGNORECASE)
        m_len = re.search(r"Total\s*length\s*of\s*chirp\s*([\d.,]+)\s*s", blob, flags=re.IGNORECASE)
        m_freq = re.search(
            r"Frequency\s*([\d.,]+)\s*Hz\s*to\s*([\d.,]+)\s*Hz", blob, flags=re.IGNORECASE
        )
        m_taper = re.search(r"Tapering\s*parameter\s*%?\s*([\d.,]+)", blob, flags=re.IGNORECASE)
        m_fs = re.search(r"\(([\d.,]+)\s*points/s\s*approximately\)", blob, flags=re.IGNORECASE)

        if not (m_baseline and m_len and m_freq):
            raise ValueError(
                "Could not parse Frequency Chirp metadata (Baseline time / chirp length / frequency range)."
            )

        waiting_time = _to_float(m_baseline.group(1))
        chirp_length = _to_float(m_len.group(1))
        f_lo_hz = _to_float(m_freq.group(1))
        f_hi_hz = _to_float(m_freq.group(2))
        tapering_percent = _to_float(m_taper.group(1)) if m_taper else float("nan")
        fs = _to_float(m_fs.group(1)) if m_fs else float("nan")

        m_mode = re.search(
            r"Oscillation-Frequency\s*Chirp\s*:\s*([A-Za-z]+)\s+is\s+selected",
            blob,
            flags=re.IGNORECASE,
        )
        m_temp = re.search(r"Temperature\s*([-\d.,]+)\s*°C", blob, flags=re.IGNORECASE)
        m_end_temp = re.search(r"End\s*temperature\s*([-\d.,]+)\s*°C", blob, flags=re.IGNORECASE)
        m_soak = re.search(r"Soak\s*time\s*([\d.,]+)\s*s", blob, flags=re.IGNORECASE)
        m_soak2 = re.search(
            r"Soak\s*time\s*after\s*ramp\s*([\d.,]+)\s*s", blob, flags=re.IGNORECASE
        )
        m_ramp = re.search(r"Ramp\s*rate\s*([\d.,]+)\s*°C/min", blob, flags=re.IGNORECASE)
        m_stress = re.search(r"Stress\s*([-\d.,]+)\s*Pa", blob, flags=re.IGNORECASE)
        m_ppd = re.search(r"Points\s*per\s*decade\s*([\d.,]+)", blob, flags=re.IGNORECASE)
        m_cf = re.search(r"Control\s*frequency\s*([\d.,]+)\s*Hz", blob, flags=re.IGNORECASE)
        m_maxfrac = re.search(
            r"Maximum\s*frequency\s*fraction\s*([\d.,]+)", blob, flags=re.IGNORECASE
        )
        m_phase = re.search(r"Phase\s*angle\s*shift\s*([-\d.,]+)\s*°", blob, flags=re.IGNORECASE)
        m_eqvel = re.search(
            r"Equilibration\s*angular\s*velocity\s*tolerance\s*([\d.,]+)\s*rad/s",
            blob,
            flags=re.IGNORECASE,
        )
        m_relax = re.search(r"Relaxation\s*time\s*([\d.,]+)\s*s", blob, flags=re.IGNORECASE)
        m_limit = re.search(r"Limit\s*checking\s*Enabled:\s*(\w+)", blob, flags=re.IGNORECASE)
        m_equil = re.search(r"Equilibrium\s*Enabled:\s*(\w+)", blob, flags=re.IGNORECASE)

        result = {
            "waiting_time": waiting_time,
            "chirp_length": chirp_length,
            "oscillation_period": chirp_length,
            "initial_frequency": 2 * _np.pi * f_lo_hz,
            "final_frequency": 2 * _np.pi * f_hi_hz,
            "taping_parameter": tapering_percent,
            "sampling_frequency": fs,
            "mode": m_mode.group(1) if m_mode else None,
            "start_temperature_c": _to_float(m_temp.group(1)) if m_temp else float("nan"),
            "end_temperature_c": _to_float(m_end_temp.group(1)) if m_end_temp else float("nan"),
            "ramp_rate_c_per_min": _to_float(m_ramp.group(1)) if m_ramp else float("nan"),
            "soak_time_s": _to_float(m_soak.group(1)) if m_soak else float("nan"),
            "soak_time_after_ramp_s": _to_float(m_soak2.group(1)) if m_soak2 else float("nan"),
            "stress_pa": _to_float(m_stress.group(1)) if m_stress else float("nan"),
            "points_per_decade": _to_float(m_ppd.group(1)) if m_ppd else float("nan"),
            "control_frequency_hz": _to_float(m_cf.group(1)) if m_cf else float("nan"),
            "max_frequency_fraction": _to_float(m_maxfrac.group(1)) if m_maxfrac else float("nan"),
            "phase_angle_shift_deg": _to_float(m_phase.group(1)) if m_phase else float("nan"),
            "equilibration_velocity_tol_rad_per_s": _to_float(m_eqvel.group(1))
            if m_eqvel
            else float("nan"),
            "relaxation_time_s": _to_float(m_relax.group(1)) if m_relax else float("nan"),
            "limit_checking_enabled": _to_bool_flag(m_limit.group(1)) if m_limit else None,
            "equilibrium_enabled": _to_bool_flag(m_equil.group(1)) if m_equil else None,
        }
        _attach_geometry_metadata(result, details)
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    #: TA rheometer serial numbers whose OWChirp analysis must use the
    #: inertia-compensated stress-controlled variant.
    _STRESS_CONTROLLED_SERIALS = frozenset({"5343", "5332"})

    def _apply(self, target: Any, **kwargs: Any) -> Any:
        """Dispatch on the measurement's method string.

        Measurements whose method is neither Arbitrary Wave nor Frequency
        Chirp are returned unchanged — their natural default coordinates are
        set at read time.
        """
        method_raw = target.conditions.get("method", "")
        m = _canonicalise_method(method_raw)

        if "arbitrary wave" in m:
            return self._apply_arbitrary_wave(target, method_raw=method_raw)
        if "frequency chirp" in m:
            return self._apply_frequency_chirp(target)

        # Non-chirp measurement: no-op.  The reader has already installed
        # scientifically-natural default coordinates so this transform is a
        # transparent pass-through for e.g. Creep, Frequency Sweep,
        # Temperature Ramp, etc.
        return target

    # ------------------------------------------------------------------
    # Arbitrary Wave path
    # ------------------------------------------------------------------

    def _apply_arbitrary_wave(self, target: Any, method_raw: str) -> Any:
        # Locate the wave-definition dict stored under the step name.
        method_key = method_raw.split("\t")[-1]  # e.g. 'Arbitrary Wave - 2'
        if "Repeat" in method_key:
            method_key = method_key.replace(" Repeat", "")
        if "Arbitrary Wave" in method_key:
            try:
                wave_number = int(method_key.split("-")[-1].strip())
                if wave_number > 1:
                    # Later Arbitrary Wave repeats reference the first one's
                    # definition block.
                    method_key = "Arbitrary Wave - 1"
            except Exception:
                pass

        if method_key not in target.details:
            serial = target.details.get("instrument_serial_number")
            raise ValueError(
                f"OWChirpAnalysis: Arbitrary Wave metadata for step "
                f"{method_key!r} was not extracted by the reader.  This "
                f"usually means the instrument serial number is missing or "
                f"unrecognised (got {serial!r}); supported prefixes are "
                f"5343/5332/5333 (DHR), 4020/4010 (ARES-G2), 4060 "
                f"(ARES-G3) and 0850 (Discovery HR).  Non-OWChirp "
                f"measurements are unaffected by this."
            )
        wave_data = target.details[method_key]

        strain_applied = wave_data["wave 2"]["coef"][0]
        (waiting_time, oscillation_period, taping_parameter, initial_frequency, final_frequency) = (
            self.calculate_wave_parameters(wave_data)
        )

        sampling_frequency = wave_data["rate (pts/s)"]
        original_dataset = target.datasets[0]

        time, strain, stress, temperature = self.prepare_owchirp_data(
            original_dataset,
            "step time",
            "strain",
            "stress",
            "temperature",
            self.cutoff_points,
        )
        strain_filtered, filter_used_strain = self.select_best_filter_method(
            waiting_time,
            time,
            strain,
            criterion=self.filter_criterion,
        )
        stress_filtered, filter_used_stress = self.select_best_filter_method(
            waiting_time,
            time,
            stress,
            criterion=self.filter_criterion,
        )

        if self.owchirp_waiting_time == "before_signal":
            time_tw = time[time > waiting_time]
            strain_filtered_tw = strain_filtered[time > waiting_time]
            stress_filtered_tw = stress_filtered[time > waiting_time]
        elif self.owchirp_waiting_time == "after_signal":
            signal_end_time = oscillation_period
            signal_start_time = time[0] + 0.1
            mask = (time > signal_start_time) & (time < signal_end_time)
            time_tw = time[mask]
            strain_filtered_tw = strain_filtered[mask]
            stress_filtered_tw = stress_filtered[mask]
        else:  # 'all_signal' or any other fallback
            time_tw = time
            strain_filtered_tw = strain_filtered
            stress_filtered_tw = stress_filtered

        strain_rate = np.gradient(strain_filtered_tw, time_tw)

        # Historically an ``if instrument in {5343,5332}`` branch dispatched
        # to a ``moduli_owchirp_stress_controlled`` method that was never
        # implemented — the comparison against the raw serial string always
        # returned False, so the strain-controlled variant was always used.
        # Arbitrary Wave files do not carry the geometry / instrument
        # constants required for the inertia-compensated variant either, so
        # we simply use the strain-controlled path.  Users on stress-
        # controlled instruments who need inertia compensation should use
        # the Frequency Chirp macro (which does export the constants).
        (
            frequency_radians,
            fourier_transform_strain,
            fourier_transform_stress,
            storage_modulus,
            loss_modulus,
            complex_modulus,
        ) = self.moduli_owchirp_strain_controlled(
            strain_filtered_tw,
            stress_filtered_tw,
            sampling_frequency,
            initial_frequency,
            final_frequency,
        )
        if target.details.get("instrument_serial_number", "").startswith(
            tuple(self._STRESS_CONTROLLED_SERIALS)
        ):
            _logger.info(
                "Arbitrary Wave step on stress-controlled instrument (serial "
                "%r): using strain-controlled OWChirp moduli. For inertia "
                "compensation switch to the Frequency Chirp macro.",
                target.details.get("instrument_serial_number"),
            )

        frequency_hertz = frequency_radians / (2 * np.pi)
        tan_delta = loss_modulus / storage_modulus
        complex_viscosity = np.sqrt(storage_modulus**2 + loss_modulus**2) / frequency_radians
        complex_modulus_abs = np.abs(complex_modulus)

        dataset_fourier_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[
                frequency_radians,
                fourier_transform_strain,
                fourier_transform_stress,
                storage_modulus,
                loss_modulus,
                complex_modulus_abs,
                frequency_hertz,
                tan_delta,
                complex_viscosity,
            ],
            data_array_names=[
                "angular frequency",
                "fourier_transform_strain",
                "fourier_transform_stress",
                "storage modulus",
                "loss modulus",
                "complex modulus",
                "frequency_hz",
                "tan(delta)",
                "complex viscosity",
            ],
            data_array_units=["rad/s", "a.u.", "a.u.", "Pa", "Pa", "Pa", "Hz", "", "Pa s"],
            default_independent_name="angular frequency",
            default_dependent_name="storage modulus",
            source="Dataset in frequency space",
        )

        dataset_time_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[time, strain_filtered, stress_filtered, strain_rate],
            data_array_names=[
                "adjusted time",
                "adjusted strain",
                "adjusted stress",
                "adjusted strain rate",
            ],
            data_array_units=["s", "a.u", "Pa", "1/s"],
        )

        original_dataset = one_d_composite_dataset.OneDimensionalCompositeDataset.from_datasets(
            [target.datasets[0], dataset_time_space]
        )

        target.datasets = [dataset_fourier_space, original_dataset]

        target.add_detail("strain_applied", strain_applied)
        target.add_detail("waiting_time", waiting_time)
        target.add_detail("taping_parameter", taping_parameter)
        target.add_detail("filter_used_strain", filter_used_strain)
        target.add_detail("filter_used_stress", filter_used_stress)
        target.add_detail("initial_frequency", initial_frequency)
        target.add_detail("final_frequency", final_frequency)
        target.add_detail("oscillation_period", oscillation_period)
        target.add_detail("sampling_frequency", sampling_frequency)

        return target

    # ------------------------------------------------------------------
    # Frequency Chirp path
    # ------------------------------------------------------------------

    def _apply_frequency_chirp(self, target: Any) -> Any:
        params = self._extract_frequency_chirp_params(target.details)

        waiting_time = params["waiting_time"]
        oscillation_period = params["oscillation_period"]
        initial_frequency = params["initial_frequency"]
        final_frequency = params["final_frequency"]
        taping_parameter = params["taping_parameter"] / 100
        sampling_frequency = params["sampling_frequency"]

        c = self.parse_value_with_prefix(params.get("geometry_compliance"))
        I_g = self.parse_value_with_prefix(params.get("geometry_inertia"))
        mu = self.parse_value_with_prefix(params.get("geometry_friction"))
        K_stress = self.parse_value_with_prefix(params.get("geometry_stress_constant"))
        I_m = self.parse_value_with_prefix(params.get("instrument_inertia"))
        K_strain = self.parse_value_with_prefix(params.get("geometry_strain_constant"))
        I = I_g + I_m

        original_dataset = target.datasets[0]
        time, strain, stress, temperature = self.prepare_owchirp_data(
            original_dataset,
            "step time",
            "strain",
            "stress",
            "temperature",
            self.cutoff_points,
        )
        strain_filtered, filter_used_strain = self.select_best_filter_method(
            waiting_time,
            time,
            strain,
            criterion=self.filter_criterion,
        )
        stress_filtered, filter_used_stress = self.select_best_filter_method(
            waiting_time,
            time,
            stress,
            criterion=self.filter_criterion,
        )

        chirp_start = waiting_time
        chirp_end = oscillation_period

        if self.owchirp_waiting_time == "before_signal":
            mask = time > chirp_start
        elif self.owchirp_waiting_time == "after_signal":
            mask = (time > chirp_start + 1e-1) & (time < chirp_end)
        elif self.owchirp_waiting_time == "inside_signal":
            mask = (time > chirp_start) & (time < chirp_end)
        elif self.owchirp_waiting_time == "before_and_after":
            mask = (time <= chirp_start) | (time >= chirp_end)
        else:  # 'all_signal'
            mask = slice(None)

        time_tw = time[mask]
        strain_filtered_tw = strain_filtered[mask]
        stress_filtered_tw = stress_filtered[mask]

        strain_rate = np.gradient(strain_filtered_tw, time_tw)  # noqa: F841 — computed for the time-space dataset

        # Frequency Chirp exports carry the geometry / instrument constants,
        # so the inertia-compensated (IC) variant is always used.  On
        # strain-controlled instruments the ``K_strain / K_stress`` ratio and
        # the compliance / inertia terms are small, so the IC result reduces
        # to the strain-controlled moduli.
        (
            frequency_radians,
            fourier_transform_strain,
            fourier_transform_stress,
            storage_modulus,
            loss_modulus,
            complex_modulus,
            complex_viscosity,
        ) = self.moduli_owchirp_stress_controlled_IC(
            time_tw,
            strain_filtered_tw,
            stress_filtered_tw,
            sampling_frequency,
            initial_frequency,
            final_frequency,
            K_stress,
            K_strain,
            I,
            c,
            mu,
        )

        frequency_hertz = frequency_radians / (2 * np.pi)
        tan_delta = loss_modulus / storage_modulus
        complex_viscosity = np.sqrt(storage_modulus**2 + loss_modulus**2) / frequency_radians
        complex_modulus_abs = np.abs(complex_modulus)

        dataset_fourier_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[
                frequency_radians,
                fourier_transform_strain,
                fourier_transform_stress,
                storage_modulus,
                loss_modulus,
                complex_modulus_abs,
                frequency_hertz,
                tan_delta,
                complex_viscosity,
            ],
            data_array_names=[
                "angular frequency",
                "fourier_transform_strain",
                "fourier_transform_stress",
                "storage modulus",
                "loss modulus",
                "complex modulus",
                "frequency_hz",
                "tan(delta)",
                "complex viscosity",
            ],
            data_array_units=["rad/s", "a.u.", "a.u.", "Pa", "Pa", "Pa", "Hz", "", "Pa s"],
            default_independent_name="angular frequency",
            default_dependent_name="storage modulus",
            source="Dataset in frequency space",
        )

        dataset_time_space = one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[time, strain_filtered, stress_filtered, strain_rate],
            data_array_names=[
                "adjusted time",
                "adjusted strain",
                "adjusted stress",
                "adjusted strain rate",
            ],
            data_array_units=["s", "a.u", "Pa", "1/s"],
            default_independent_name="adjusted time",
            default_dependent_name="adjusted stress",
            source="Dataset in time space",
        )

        original_dataset = one_d_composite_dataset.OneDimensionalCompositeDataset.from_datasets(
            [original_dataset, dataset_time_space]
        )

        target.datasets = [dataset_fourier_space, original_dataset]

        target.add_detail("waiting_time", waiting_time)
        target.add_detail("taping_parameter", taping_parameter)
        target.add_detail("initial_frequency", initial_frequency)
        target.add_detail("final_frequency", final_frequency)
        target.add_detail("oscillation_period", oscillation_period)
        target.add_detail("sampling_frequency", sampling_frequency)
        target.add_detail("filter_used_strain", filter_used_strain)
        target.add_detail("filter_used_stress", filter_used_stress)

        return target
