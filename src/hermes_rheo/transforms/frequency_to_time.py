"""Frequency-to-time reconstruction of relaxation modulus using iRheoTempo V9

This transform reconstructs the time-domain relaxation modulus :math:`G(t)`
from frequency-domain oscillatory data :math:`G'(\omega), G''(\omega)`
through the interval-slope Fourier reconstruction used in iRheoTempo V9
(Tassieri and Ramirez, 2026) [1]_.

The transform is designed to compose with existing hermes-rheo pipelines,
for example on raw frequency sweeps, after
:class:`~hermes_rheo.transforms.split_temperature_sweep.SplitTemperatureSweep`,
or after :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`.

Notes
-----
Given :math:`\eta^*(\omega) = G^*(\omega)/(i\omega)` with
:math:`\eta^*(\omega)=\eta'(\omega)-i\eta''(\omega)`, the reconstruction
uses piecewise-linear interval slopes in frequency space:

.. math::

   a_{1,k}=\frac{\eta'_{k+1}-\eta'_k}{\omega_{k+1}-\omega_k},\quad
   a_{2,k}=\frac{\eta''_{k+1}-\eta''_k}{\omega_{k+1}-\omega_k}

.. math::

   G_{\eta'}(t)=\frac{2}{\pi t^2}\sum_k
   \left[\cos(\omega_{R,k}t)-\cos(\omega_{L,k}t)\right]a_{1,k}

.. math::

   G_{\eta''}(t)=\frac{2}{\pi t^2}\sum_k
   \left[\sin(\omega_{R,k}t)-\sin(\omega_{L,k}t)\right]a_{2,k}
   +\frac{2}{\pi}\frac{\eta''(0)}{t}

References
----------
.. [1] Tassieri, M. and Ramirez, J., *i-Rheo Tempo: Open-Access
   Analytical Tool for Broadband Viscoelastic Characterisation* (2026).
   DOI: `10.5525/gla.researchdata.2230
   <https://doi.org/10.5525/gla.researchdata.2230>`_.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple
import importlib
import warnings

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.optimize import fmin

from piblin.transform.abc.measurement_transform import MeasurementTransform
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset


def _sanitize_omega_moduli(
    omega: np.ndarray,
    g_prime: np.ndarray,
    g_double_prime: np.ndarray,
    units: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted, finite, unique arrays in angular-frequency units

    Parameters
    ----------
    omega : np.ndarray
        Frequency axis in either rad/s or Hz
    g_prime : np.ndarray
        Storage modulus in Pa
    g_double_prime : np.ndarray
        Loss modulus in Pa
    units : str
        Either ``"rad/s"`` or ``"Hz"``

    Returns
    -------
    tuple of np.ndarray
        ``(omega, g_prime, g_double_prime)`` sanitized and sorted
    """
    omega = np.asarray(omega, dtype=float).ravel()
    g_prime = np.asarray(g_prime, dtype=float).ravel()
    g_double_prime = np.asarray(g_double_prime, dtype=float).ravel()

    if units.lower() == "hz":
        omega = 2.0 * np.pi * omega

    mask = np.isfinite(omega) & np.isfinite(g_prime) & np.isfinite(g_double_prime)
    mask &= omega > 0.0
    omega = omega[mask]
    g_prime = g_prime[mask]
    g_double_prime = g_double_prime[mask]

    if omega.size < 5:
        raise ValueError("FrequencyToTime requires at least 5 finite positive frequency points")

    order = np.argsort(omega)
    omega = omega[order]
    g_prime = g_prime[order]
    g_double_prime = g_double_prime[order]

    omega_unique, idx = np.unique(omega, return_index=True)
    return omega_unique, g_prime[idx], g_double_prime[idx]


def _estimate_eta10_polynomial(
    omega: np.ndarray,
    eta1: np.ndarray,
    lf_poly_order: int,
    lf_fit_decades: float,
) -> float:
    """Estimate :math:`\eta'(0)` from a low-frequency log-log polynomial fit"""
    wmin = float(np.min(omega))
    wmax = float(np.max(omega))
    lf_fit_decades = float(lf_fit_decades)
    lf_poly_order = max(1, int(lf_poly_order))

    if not np.isfinite(lf_fit_decades) or lf_fit_decades <= 0.0:
        lf_fit_decades = 1.0

    w_lf_fit_max = min(wmin * (10.0 ** lf_fit_decades), wmax)
    idx_lf = (omega >= wmin) & (omega <= w_lf_fit_max)

    min_fit_pts = max(6, lf_poly_order + 2)
    if np.count_nonzero(idx_lf) < min_fit_pts:
        idx_lf = np.zeros_like(omega, dtype=bool)
        idx_lf[: min(min_fit_pts, omega.size)] = True

    x_lf = np.log10(omega[idx_lf])
    y_lf = np.log10(np.maximum(np.abs(eta1[idx_lf]), np.finfo(float).tiny))
    p_lf = np.polyfit(x_lf, y_lf, lf_poly_order)

    eta10_mag = 10.0 ** np.polyval(p_lf, np.log10(wmin))
    sgn = np.sign(eta1[0])
    if sgn == 0.0:
        sgn = 1.0
    return float(sgn * eta10_mag)


def _build_interpolation_dataset(
    omega: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    eta10_fit: float,
    eta20: float,
    hf_poly_order: int,
    hf_fit_decades: float,
    hf_ext_decades: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LF anchor + experimental + HF extension interpolation arrays"""
    wmin = float(np.min(omega))
    wmax = float(np.max(omega))

    hf_fit_decades = float(hf_fit_decades)
    hf_ext_decades = float(hf_ext_decades)
    hf_poly_order = max(1, int(hf_poly_order))

    if not np.isfinite(hf_fit_decades) or hf_fit_decades <= 0.0:
        hf_fit_decades = 1.0
    if not np.isfinite(hf_ext_decades) or hf_ext_decades < 0.0:
        hf_ext_decades = 1.0

    w_hf_fit_min = max(wmax / (10.0 ** hf_fit_decades), wmin)
    idx_hf = (omega >= w_hf_fit_min) & (omega <= wmax)
    min_fit_pts = max(6, hf_poly_order + 2)
    if np.count_nonzero(idx_hf) < min_fit_pts:
        idx_hf = np.zeros_like(omega, dtype=bool)
        idx_hf[max(0, omega.size - min_fit_pts):] = True

    x_hf = np.log10(omega[idx_hf])
    y1_hf = np.log10(np.maximum(np.abs(eta1[idx_hf]), np.finfo(float).tiny))
    y2_hf = np.log10(np.maximum(np.abs(eta2[idx_hf]), np.finfo(float).tiny))

    p1_hf = np.polyfit(x_hf, y1_hf, hf_poly_order)
    p2_hf = np.polyfit(x_hf, y2_hf, hf_poly_order)

    w_hf = wmax * (10.0 ** hf_ext_decades)
    n_hf = 60
    w_hf_grid = np.logspace(np.log10(wmax), np.log10(w_hf), n_hf)

    e1_hf = 10.0 ** np.polyval(p1_hf, np.log10(w_hf_grid))
    e2_hf = 10.0 ** np.polyval(p2_hf, np.log10(w_hf_grid))

    sgn1h = np.sign(eta1[-1])
    if sgn1h == 0.0:
        sgn1h = 1.0
    e1_hf = sgn1h * e1_hf

    w_data = np.concatenate([[0.0], omega, w_hf_grid[1:]])
    e1_data = np.concatenate([[eta10_fit], eta1, e1_hf[1:]])
    e2_data = np.concatenate([[eta20], eta2, e2_hf[1:]])

    idx = np.argsort(w_data)
    w_data = w_data[idx]
    e1_data = e1_data[idx]
    e2_data = e2_data[idx]

    w_data, iu = np.unique(w_data, return_index=True)
    return w_data, e1_data[iu], e2_data[iu]


def _build_interpolants(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    interpolation_method: str,
    csaps_smoothing: float,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Create interpolant callables for eta1 and eta2"""
    method = interpolation_method.strip().lower()
    if method == "pchip":
        return PchipInterpolator(x, y1), PchipInterpolator(x, y2)

    if method == "spline":
        return CubicSpline(x, y1), CubicSpline(x, y2)

    if method == "csaps":
        try:
            csaps = getattr(importlib.import_module("csaps"), "csaps")

            p = float(np.clip(csaps_smoothing, 0.0, 1.0))

            def pp1(xq: np.ndarray) -> np.ndarray:
                return np.asarray(csaps(x, y1, xq, smooth=p), dtype=float)

            def pp2(xq: np.ndarray) -> np.ndarray:
                return np.asarray(csaps(x, y2, xq, smooth=p), dtype=float)

            return pp1, pp2
        except Exception:
            warnings.warn(
                "csaps interpolation requested but package is unavailable; falling back to cubic spline",
                RuntimeWarning,
                stacklevel=2,
            )
            return CubicSpline(x, y1), CubicSpline(x, y2)

    raise ValueError("interpolation_method must be one of {'pchip', 'spline', 'csaps'}")


def _resample(
    w_data: np.ndarray,
    pp1: Callable[[np.ndarray], np.ndarray],
    pp2: Callable[[np.ndarray], np.ndarray],
    eta10_fit: float,
    eta20: float,
    resample_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate interpolants on the dense V9 processing grid"""
    nres = max(int(resample_points), 50)
    w_lf_min = 1e-15
    w_max = float(np.max(w_data))
    if not (np.isfinite(w_max) and w_max > w_lf_min):
        raise ValueError("Invalid resampling range: max frequency must exceed 1e-15 rad/s")

    w_pos = np.logspace(np.log10(w_lf_min), np.log10(w_max), nres - 1)
    w_proc = np.concatenate([[0.0], w_pos])

    eta1_proc = np.asarray(pp1(w_proc), dtype=float)
    eta2_proc = np.asarray(pp2(w_proc), dtype=float)

    eta1_proc[0] = eta10_fit
    eta2_proc[0] = eta20
    return w_proc, eta1_proc, eta2_proc


def _compute_delta_a(
    w: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return slope-jump diagnostics for eta1 and eta2"""
    dw = np.diff(w)
    a1 = np.diff(eta1) / dw
    a2 = np.diff(eta2) / dw

    da1 = np.concatenate([[a1[0]], np.diff(a1), [-a1[-1]]])
    da2 = np.concatenate([[a2[0]], np.diff(a2), [-a2[-1]]])
    wk = np.concatenate([[w[0]], w[1:-1], [w[-1]]])
    return wk, da1, da2


def _compute_gt_v8(
    w: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    wmin_exp: float,
    wmax_exp: float,
    n_time_points: int,
    eta20: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct G(t) from interval slopes of eta1 and eta2"""
    d_w = np.diff(w)
    a1 = np.diff(eta1) / d_w
    a2 = np.diff(eta2) / d_w

    n_time_points = max(100, int(n_time_points))
    t_min = 1.0 / float(wmax_exp)
    t_max = 1.0 / float(wmin_exp)
    t = np.logspace(np.log10(t_min), np.log10(t_max), n_time_points)

    wl = w[:-1][None, :]
    wr = w[1:][None, :]
    tt = t[:, None]

    g1 = (2.0 / np.pi) * (1.0 / t**2) * ((np.cos(wr * tt) - np.cos(wl * tt)) @ a1)
    g2 = (2.0 / np.pi) * (1.0 / t**2) * ((np.sin(wr * tt) - np.sin(wl * tt)) @ a2)
    g2 = g2 + (2.0 / np.pi) * eta20 * (1.0 / t)
    return t, g1, g2


def _compute_constant_threshold(
    w_exp: np.ndarray,
    eta1_exp: np.ndarray,
    eta2_exp: np.ndarray,
) -> float:
    """Return constant threshold diagnostic from first experimental interval"""
    if w_exp.size < 2:
        return float("nan")

    w1 = float(w_exp[0])
    w2 = float(w_exp[1])
    dw1 = w2 - w1
    if not (np.isfinite(dw1) and dw1 > 0.0):
        return float("nan")

    a1 = (eta1_exp[1] - eta1_exp[0]) / dw1
    a2 = (eta2_exp[1] - eta2_exp[0]) / dw1

    gthr1 = (2.0 / np.pi) * abs(a1) * w1 * dw1
    gthr2 = (2.0 / np.pi) * abs(a2) * w1 * dw1
    return float(max(gthr1, gthr2))


def _compute_mrae(
    t_model: np.ndarray,
    g_model: np.ndarray,
    t_exp: np.ndarray,
    g_exp: np.ndarray,
) -> Optional[float]:
    """Compute mean relative absolute error on the overlapping log-time domain"""
    t_model = np.asarray(t_model, dtype=float).ravel()
    g_model = np.asarray(g_model, dtype=float).ravel()
    t_exp = np.asarray(t_exp, dtype=float).ravel()
    g_exp = np.asarray(g_exp, dtype=float).ravel()

    mask_m = np.isfinite(t_model) & np.isfinite(g_model) & (t_model > 0.0)
    mask_e = np.isfinite(t_exp) & np.isfinite(g_exp) & (t_exp > 0.0)
    if np.count_nonzero(mask_m) < 3 or np.count_nonzero(mask_e) < 3:
        return None

    t_m = t_model[mask_m]
    g_m = g_model[mask_m]
    t_e = t_exp[mask_e]
    g_e = g_exp[mask_e]

    order_m = np.argsort(t_m)
    order_e = np.argsort(t_e)
    t_m, g_m = t_m[order_m], g_m[order_m]
    t_e, g_e = t_e[order_e], g_e[order_e]

    t_lo = max(float(np.min(t_m)), float(np.min(t_e)))
    t_hi = min(float(np.max(t_m)), float(np.max(t_e)))
    if not (t_hi > t_lo > 0.0):
        return None

    overlap = (t_e >= t_lo) & (t_e <= t_hi)
    if np.count_nonzero(overlap) < 3:
        return None

    # Relative error becomes numerically meaningless when the reference
    # signal approaches machine zero in the long-time tail.
    amp_ref = np.max(np.abs(g_e[overlap]))
    valid_amp = np.abs(g_e) >= max(1e-8 * amp_ref, np.finfo(float).tiny)
    overlap = overlap & valid_amp
    if np.count_nonzero(overlap) < 3:
        return None

    f = interp1d(
        np.log10(t_m),
        g_m,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    g_pred = f(np.log10(t_e[overlap]))
    denom = np.maximum(np.abs(g_e[overlap]), np.finfo(float).tiny)
    return float(np.mean(np.abs(g_pred - g_e[overlap]) / denom))


def _find_lowest_frequency_crossover(
    omega: np.ndarray,
    g_prime: np.ndarray,
    g_double_prime: np.ndarray,
) -> Tuple[float, Optional[int]]:
    """Locate the lowest-frequency :math:`G'=G''` crossover

    A single sign change of ``G'-G''`` is bracketed and the crossover
    frequency is linearly interpolated between the two endpoints.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency axis in rad/s, sorted ascending
    g_prime : np.ndarray
        Storage modulus in Pa
    g_double_prime : np.ndarray
        Loss modulus in Pa

    Returns
    -------
    tuple
        ``(wcross, idx_cross)`` where ``wcross`` is the crossover frequency
        (or ``nan`` if none is found) and ``idx_cross`` is the left index of
        the bracketing pair (or ``None``)
    """
    w = np.asarray(omega, dtype=float).ravel()
    gp = np.asarray(g_prime, dtype=float).ravel()
    gpp = np.asarray(g_double_prime, dtype=float).ravel()
    d = gp - gpp
    idx = np.where(d[:-1] * d[1:] <= 0.0)[0]
    if idx.size == 0:
        return float("nan"), None
    idx_cross = int(idx[0])
    w1, w2 = float(w[idx_cross]), float(w[idx_cross + 1])
    d1, d2 = float(d[idx_cross]), float(d[idx_cross + 1])
    if d2 == d1:
        return float(w1), idx_cross
    wcross = w1 + (w2 - w1) * (-d1) / (d2 - d1)
    return float(wcross), idx_cross


def _maxwell_objective(
    g0: float,
    tau: float,
    omega: np.ndarray,
    g_prime_data: np.ndarray,
    g_double_prime_data: np.ndarray,
) -> float:
    """Sum of squared relative residuals for a single-mode Maxwell fit

    Uses the model
    :math:`G'(\\omega)=G_0(\\omega\\tau)^2/(1+(\\omega\\tau)^2)`,
    :math:`G''(\\omega)=G_0(\\omega\\tau)/(1+(\\omega\\tau)^2)`
    with small floor values to avoid division by zero.
    """
    wt = omega * tau
    gp_fit = g0 * (wt ** 2) / (1.0 + wt ** 2)
    gpp_fit = g0 * wt / (1.0 + wt ** 2)

    ref1 = max(np.max(np.abs(g_prime_data)), np.finfo(float).tiny) * 1e-6
    ref2 = max(np.max(np.abs(g_double_prime_data)), np.finfo(float).tiny) * 1e-6
    den1 = np.maximum(np.abs(g_prime_data), ref1)
    den2 = np.maximum(np.abs(g_double_prime_data), ref2)
    e1 = (gp_fit - g_prime_data) / den1
    e2 = (gpp_fit - g_double_prime_data) / den2
    return float(np.sum(e1 ** 2) + np.sum(e2 ** 2))


def _fit_maxwell_tail(
    omega: np.ndarray,
    g_prime: np.ndarray,
    g_double_prime: np.ndarray,
    n_tail_points: int = 200,
) -> Tuple[bool, float, float, float, np.ndarray, np.ndarray]:
    """Fit a single-mode Maxwell to the low-frequency region

    Selects all measured frequencies at or below the first
    :math:`G'=G''` crossover, fits :math:`G_0` and :math:`\\tau` in
    log-space by minimising :func:`_maxwell_objective`, and returns the
    long-time exponential tail
    :math:`G_{\\mathrm{tail}}(t)=G_0\\exp(-t/\\tau)` sampled between
    :math:`1/\\omega_c` and :math:`1/\\omega_{\\min}`.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency in rad/s (sorted ascending)
    g_prime : np.ndarray
        Storage modulus in Pa
    g_double_prime : np.ndarray
        Loss modulus in Pa
    n_tail_points : int, default 200
        Number of log-spaced time samples in the reported tail

    Returns
    -------
    tuple
        ``(available, G0, tau, wcross, t_tail, G_tail)``. ``available`` is
        ``False`` and the numeric outputs are ``nan`` / empty arrays when a
        crossover cannot be bracketed or the fit window is too short.
    """
    empty = np.array([], dtype=float)

    wcross, idx_cross = _find_lowest_frequency_crossover(omega, g_prime, g_double_prime)
    if not np.isfinite(wcross) or idx_cross is None:
        return False, float("nan"), float("nan"), float("nan"), empty, empty

    idx_fit = omega <= wcross
    if int(np.count_nonzero(idx_fit)) < 4:
        return False, float("nan"), float("nan"), float("nan"), empty, empty

    w = omega[idx_fit]
    gp = g_prime[idx_fit]
    gpp = g_double_prime[idx_fit]

    tau0 = 1.0 / wcross
    g00 = max(float(np.interp(wcross, omega, g_double_prime)), float(np.finfo(float).tiny))
    x0 = np.log([g00, tau0])
    xfit = fmin(
        lambda x: _maxwell_objective(np.exp(x[0]), np.exp(x[1]), w, gp, gpp),
        x0,
        disp=False,
    )

    g0 = float(np.exp(xfit[0]))
    tau = float(np.exp(xfit[1]))

    wmin_exp = float(np.min(omega))
    tcross = 1.0 / wcross
    tmax = 1.0 / wmin_exp
    if not (np.isfinite(tcross) and np.isfinite(tmax) and tcross < tmax):
        return False, g0, tau, wcross, empty, empty

    t_tail = np.logspace(np.log10(tcross), np.log10(tmax), int(max(50, n_tail_points)))
    g_tail = g0 * np.exp(-t_tail / tau)
    return True, g0, tau, float(wcross), t_tail, g_tail


@dataclass
class FrequencyToTimeResult:
    """Container for outputs of :class:`FrequencyToTime`"""

    omega_exp: np.ndarray
    eta1_exp: np.ndarray
    eta2_exp: np.ndarray

    eta10_fit: float

    W_interpdata: np.ndarray
    eta1_interpdata: np.ndarray
    eta2_interpdata: np.ndarray

    W_proc: np.ndarray
    eta1_proc: np.ndarray
    eta2_proc: np.ndarray

    t_report: np.ndarray
    G_eta1: np.ndarray
    G_eta2: np.ndarray

    delta_a1_raw: np.ndarray
    delta_a2_raw: np.ndarray
    delta_a1_proc: np.ndarray
    delta_a2_proc: np.ndarray
    W_raw: np.ndarray

    G_threshold_const: float
    mrae_eta1: Optional[float]
    mrae_eta2: Optional[float]

    dataset: one_d_composite_dataset.OneDimensionalCompositeDataset

    maxwell_tail_available: bool = False
    maxwell_G0: float = float("nan")
    maxwell_tau: float = float("nan")
    maxwell_wcross: float = float("nan")
    maxwell_ttail: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    maxwell_Gtail: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


class FrequencyToTime(MeasurementTransform):
    """Reconstruct relaxation modulus from oscillatory frequency-domain data

    The transform consumes a measurement whose active dataset contains
    ``angular frequency``, ``storage modulus`` and ``loss modulus`` arrays.
    It prepends a new time-domain dataset with
    ``time``, ``G_from_eta1`` and ``G_from_eta2``.

    Parameters
    ----------
    units : {"rad/s", "Hz"}, default "rad/s"
        Units of the input frequency axis
    lf_poly_order : int, default 1
        Polynomial order for low-frequency :math:`\eta'(0)` fit
    lf_fit_decades : float, default 0.1
        Width of low-frequency fitting window in decades
    hf_poly_order : int, default 1
        Polynomial order for high-frequency extension
    hf_fit_decades : float, default 1.0
        Width of high-frequency fitting window in decades
    hf_ext_decades : float, default 1.0
        High-frequency extension span beyond the maximum measured frequency
    interpolation_method : {"pchip", "spline", "csaps"}, default "spline"
        Interpolant used for viscosity curves
    csaps_smoothing : float, default 0.999
        Smoothing parameter used when ``interpolation_method='csaps'``
    resample_points : int, default 5000
        Number of dense resampling points in frequency space
    enforce_eta2_zero : bool, default True
        If True, force :math:`\eta''(0)` to ``eta2_zero``
    eta2_zero : float, default 0.0
        Zero-frequency anchor value for :math:`\eta''(0)`
    n_time_points : int, default 600
        Number of time points in the reconstructed output window
    experimental_gt : tuple[np.ndarray, np.ndarray] or None, default None
        Optional ``(t, G(t))`` used only for MRAE diagnostics
    fit_maxwell_tail : bool, default False
        If True, fit a single-mode Maxwell to the low-frequency region
        below the first :math:`G'=G''` crossover.  The fitted
        :math:`G_0,\\tau,\\omega_c` and the long-time exponential tail
        :math:`G_{\\mathrm{tail}}(t)=G_0\\exp(-t/\\tau)` are stored on the
        result and made available on ``target.details``.
    use_maxwell_eta10 : bool, default False
        If True and the Maxwell fit succeeds, override the polynomial
        estimate of :math:`\\eta'(0)` with :math:`G_0\\tau`.  Ignored when
        ``fit_maxwell_tail=False``.
    n_maxwell_tail_points : int, default 200
        Number of log-spaced time samples in the reported Maxwell tail

    Examples
    --------
    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.frequency_to_time import FrequencyToTime
    >>> exp = TriosRheoReader().data_from_filepath(
    ...     "test_frequency_sweep_tts_ARES.txt", create_composite_datasets=True
    ... )  # doctest: +SKIP
    >>> out = FrequencyToTime()(exp)  # doctest: +SKIP
    >>> out.measurements[0].datasets[0].data_array_names  # doctest: +SKIP
    ['time', 'G_from_eta1', 'G_from_eta2']
    """

    def __init__(
        self,
        units: str = "rad/s",
        lf_poly_order: int = 1,
        lf_fit_decades: float = 0.1,
        hf_poly_order: int = 1,
        hf_fit_decades: float = 1.0,
        hf_ext_decades: float = 1.0,
        interpolation_method: str = "spline",
        csaps_smoothing: float = 0.999,
        resample_points: int = 5000,
        enforce_eta2_zero: bool = True,
        eta2_zero: float = 0.0,
        n_time_points: int = 600,
        experimental_gt: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fit_maxwell_tail: bool = False,
        use_maxwell_eta10: bool = False,
        n_maxwell_tail_points: int = 200,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.units = units
        self.lf_poly_order = lf_poly_order
        self.lf_fit_decades = lf_fit_decades
        self.hf_poly_order = hf_poly_order
        self.hf_fit_decades = hf_fit_decades
        self.hf_ext_decades = hf_ext_decades
        self.interpolation_method = interpolation_method
        self.csaps_smoothing = csaps_smoothing
        self.resample_points = resample_points
        self.enforce_eta2_zero = enforce_eta2_zero
        self.eta2_zero = eta2_zero
        self.n_time_points = n_time_points
        self.experimental_gt = experimental_gt
        self.fit_maxwell_tail = fit_maxwell_tail
        self.use_maxwell_eta10 = use_maxwell_eta10
        self.n_maxwell_tail_points = n_maxwell_tail_points
        self._result: Optional[FrequencyToTimeResult] = None

    @property
    def result(self) -> Optional[FrequencyToTimeResult]:
        """Most recent processed result"""
        return self._result

    def _apply(self, target: Any, **kwargs: Any) -> Any:
        # Detection is based on the presence of the required frequency-domain
        # columns on ``target.datasets[0]``. This makes the transform work for
        # any upstream that leaves ``angular frequency``, ``storage modulus``
        # and ``loss modulus`` on the active dataset — including OWChirp-
        # processed steps and TORCh files that already store pre-computed
        # frequency-domain moduli under non-standard method names.
        if not hasattr(target, "datasets") or len(target.datasets) == 0:
            return target

        dataset = target.datasets[0]
        if not hasattr(dataset, "data_array_names") or not hasattr(dataset, "data_arrays"):
            return target

        names = list(dataset.data_array_names)
        required = ["angular frequency", "storage modulus", "loss modulus"]
        if any(name not in names for name in required):
            return target

        idx_omega = names.index("angular frequency")
        idx_gp = names.index("storage modulus")
        idx_gpp = names.index("loss modulus")

        omega_raw = np.asarray(dataset.data_arrays[idx_omega], dtype=float)
        gp_raw = np.asarray(dataset.data_arrays[idx_gp], dtype=float)
        gpp_raw = np.asarray(dataset.data_arrays[idx_gpp], dtype=float)

        # Numerical gate: require an actually swept ω-axis. Temperature ramps,
        # creep and stress-relaxation steps often carry ``angular frequency``
        # as a constant single-frequency oscillation, which is not a valid
        # source for the frequency → time inversion.
        omega_finite = omega_raw[np.isfinite(omega_raw) & (omega_raw > 0.0)]
        if omega_finite.size < 3:
            return target
        omega_min = float(np.min(omega_finite))
        omega_max = float(np.max(omega_finite))
        if omega_max / omega_min < 2.0:
            return target

        omega_exp, gp_exp, gpp_exp = _sanitize_omega_moduli(
            omega_raw, gp_raw, gpp_raw, self.units
        )
        eta1_exp = gpp_exp / omega_exp
        eta2_exp = gp_exp / omega_exp

        eta10_fit = _estimate_eta10_polynomial(
            omega_exp,
            eta1_exp,
            lf_poly_order=self.lf_poly_order,
            lf_fit_decades=self.lf_fit_decades,
        )

        eta20 = float(self.eta2_zero if self.enforce_eta2_zero else eta2_exp[0])

        # Optional single-mode Maxwell terminal tail (mirrors the iRheoTempo
        # V9 GUI "Overlay Maxwell tail" option, but purely numerical here).
        maxwell_available = False
        maxwell_G0 = float("nan")
        maxwell_tau = float("nan")
        maxwell_wcross = float("nan")
        maxwell_ttail = np.array([], dtype=float)
        maxwell_Gtail = np.array([], dtype=float)
        if self.fit_maxwell_tail:
            (
                maxwell_available,
                maxwell_G0,
                maxwell_tau,
                maxwell_wcross,
                maxwell_ttail,
                maxwell_Gtail,
            ) = _fit_maxwell_tail(
                omega_exp,
                gp_exp,
                gpp_exp,
                n_tail_points=self.n_maxwell_tail_points,
            )
            if (
                maxwell_available
                and self.use_maxwell_eta10
                and np.isfinite(maxwell_G0)
                and np.isfinite(maxwell_tau)
            ):
                eta10_fit = float(maxwell_G0 * maxwell_tau)

        w_interpdata, eta1_interpdata, eta2_interpdata = _build_interpolation_dataset(
            omega_exp,
            eta1_exp,
            eta2_exp,
            eta10_fit=eta10_fit,
            eta20=eta20,
            hf_poly_order=self.hf_poly_order,
            hf_fit_decades=self.hf_fit_decades,
            hf_ext_decades=self.hf_ext_decades,
        )

        pp1, pp2 = _build_interpolants(
            w_interpdata,
            eta1_interpdata,
            eta2_interpdata,
            interpolation_method=self.interpolation_method,
            csaps_smoothing=self.csaps_smoothing,
        )

        w_proc, eta1_proc, eta2_proc = _resample(
            w_interpdata,
            pp1,
            pp2,
            eta10_fit=eta10_fit,
            eta20=eta20,
            resample_points=self.resample_points,
        )

        t_report, g_eta1, g_eta2 = _compute_gt_v8(
            w_proc,
            eta1_proc,
            eta2_proc,
            wmin_exp=float(np.min(omega_exp)),
            wmax_exp=float(np.max(omega_exp)),
            n_time_points=self.n_time_points,
            eta20=eta20,
        )

        w_raw, delta_a1_raw, delta_a2_raw = _compute_delta_a(omega_exp, eta1_exp, eta2_exp)
        _, delta_a1_proc, delta_a2_proc = _compute_delta_a(w_proc, eta1_proc, eta2_proc)

        g_threshold_const = _compute_constant_threshold(omega_exp, eta1_exp, eta2_exp)

        mrae_eta1 = None
        mrae_eta2 = None
        if self.experimental_gt is not None:
            t_exp, g_exp = self.experimental_gt
            mrae_eta1 = _compute_mrae(t_report, g_eta1, t_exp, g_exp)
            mrae_eta2 = _compute_mrae(t_report, g_eta2, t_exp, g_exp)

        dataset_time = one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[t_report, g_eta1, g_eta2],
            data_array_names=["time", "G_from_eta1", "G_from_eta2"],
            data_array_units=["s", "Pa", "Pa"],
            default_independent_name="time",
            default_dependent_name="G_from_eta1",
            source="Dataset in time space from frequency-domain reconstruction",
        )

        target.datasets = [dataset_time, *list(target.datasets)]

        result = FrequencyToTimeResult(
            omega_exp=omega_exp,
            eta1_exp=eta1_exp,
            eta2_exp=eta2_exp,
            eta10_fit=eta10_fit,
            W_interpdata=w_interpdata,
            eta1_interpdata=eta1_interpdata,
            eta2_interpdata=eta2_interpdata,
            W_proc=w_proc,
            eta1_proc=eta1_proc,
            eta2_proc=eta2_proc,
            t_report=t_report,
            G_eta1=g_eta1,
            G_eta2=g_eta2,
            delta_a1_raw=delta_a1_raw,
            delta_a2_raw=delta_a2_raw,
            delta_a1_proc=delta_a1_proc,
            delta_a2_proc=delta_a2_proc,
            W_raw=w_raw,
            G_threshold_const=g_threshold_const,
            mrae_eta1=mrae_eta1,
            mrae_eta2=mrae_eta2,
            dataset=dataset_time,
            maxwell_tail_available=maxwell_available,
            maxwell_G0=maxwell_G0,
            maxwell_tau=maxwell_tau,
            maxwell_wcross=maxwell_wcross,
            maxwell_ttail=maxwell_ttail,
            maxwell_Gtail=maxwell_Gtail,
        )

        self._result = result
        try:
            if hasattr(target, "details") and isinstance(target.details, dict):
                target.details["frequency_to_time_result"] = result
        except Exception:
            pass

        return target
