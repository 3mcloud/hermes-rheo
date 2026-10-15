"""Continuous and discrete relaxation spectra via pyReSpect.

This transform wraps the two open-source pyReSpect solvers behind a
single hermes-rheo ``MeasurementTransform`` interface:

* :mod:`pyrespect_time` --- given :math:`G(t)` from a stress-relaxation
  experiment, recover the continuous relaxation spectrum
  :math:`H(s)` via Tikhonov regularisation with Bayesian
  :math:`\\lambda` selection [Shanbhag2019]_ [TakehShanbhag2013]_ and
  a discrete Prony series :math:`(g_i, \\tau_i)` by AIC minimisation.

* :mod:`pyrespect_freq` --- given oscillatory moduli
  :math:`G'(\\omega), G''(\\omega)`, recover the same
  :math:`H(s)` / :math:`(g_i, \\tau_i)` [Shanbhag2020]_.

Both solvers are vendored in :mod:`hermes_rheo._vendor` (MIT licence,
see ``_vendor/README.md``) and need no separate installation.

Given the following inverse problems:

.. math::

   G(t) = G_\\infty + \\int_{-\\infty}^{\\infty} H(s) e^{-t/s} \\, d\\ln s

.. math::

   G'(\\omega) = G_\\infty + \\int_{-\\infty}^{\\infty}
       \\frac{\\omega^2 s^2}{1 + \\omega^2 s^2} H(s)\\, d\\ln s

.. math::

   G''(\\omega) = \\int_{-\\infty}^{\\infty}
       \\frac{\\omega s}{1 + \\omega^2 s^2} H(s)\\, d\\ln s

the transform returns two composite datasets prepended onto the input
measurement:

1. ``continuous relaxation spectrum`` --- ``s`` (s), ``H`` (log CRS,
   dimensionless), ``h`` (Pa; :math:`h = \\exp(H)`) and, when
   available, ``dH`` (log CRS error band from the frequency branch).
2. ``discrete relaxation spectrum`` --- ``discrete relaxation time``
   (s), ``g`` (Pa) and, when available, ``dtau`` (s; the time branch's
   NLLS uncertainty estimate).

The full solver output is stashed on
``target.details["relaxation_spectrum_result"]`` as a
:class:`RelaxationSpectrumResult` dataclass for downstream code (plots,
model fits, tutorials).

References
----------
.. [Shanbhag2019] Shanbhag, S., *pyReSpect: A Computer Program to
   Extract Discrete and Continuous Spectra from Stress Relaxation
   Experiments*, Macromol. Theory Simul. **28**, 1900005 (2019).
   DOI: `10.1002/mats.201900005
   <https://doi.org/10.1002/mats.201900005>`_.

.. [Shanbhag2020] Shanbhag, S., *Relaxation spectra using nonlinear
   Tikhonov regularization with a Bayesian criterion*, Rheologica Acta
   **59**, 509 (2020).
   DOI: `10.1007/s00397-020-01212-w
   <https://doi.org/10.1007/s00397-020-01212-w>`_.

.. [TakehShanbhag2013] Takeh, A. and Shanbhag, S., *A computer program
   to extract the continuous and discrete relaxation spectra from
   dynamic viscoelastic measurements*, Applied Rheology **23**, 24628
   (2013).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from piblin.transform.abc.measurement_transform import MeasurementTransform
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset


__all__ = ["RelaxationSpectrum", "RelaxationSpectrumResult"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOMAIN_TIME = "time"
_DOMAIN_FREQUENCY = "frequency"
_VALID_DOMAINS = (_DOMAIN_TIME, _DOMAIN_FREQUENCY)

#: Aliases accepted for the relaxation-modulus column in the time branch.
#: The first hit wins.  ``G_from_eta1`` / ``G_from_eta2`` let the transform
#: chain seamlessly after
#: :class:`~hermes_rheo.transforms.frequency_to_time.FrequencyToTime`.
_RELAXATION_ALIASES = (
    "G_from_eta1",
    "G_from_eta2",
    "relaxation modulus",
    "G_relax",
    "modulus",
)

_TIME_ALIASES = ("time", "step time")


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------


def _require_pyrespect(domain: str):
    """Return the vendored pyReSpect module for *domain*."""
    if domain == _DOMAIN_TIME:
        from hermes_rheo._vendor import pyrespect_time

        return pyrespect_time
    if domain == _DOMAIN_FREQUENCY:
        from hermes_rheo._vendor import pyrespect_freq

        return pyrespect_freq
    raise ValueError(f"Unknown domain {domain!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RelaxationSpectrumResult:
    """Unified container for pyReSpect-time and pyReSpect-freq outputs.

    Attributes
    ----------
    domain : {"time", "frequency"}
        Which pyReSpect branch produced the result.
    config : dict
        Snapshot of the ``ReSpectConfig`` values actually passed to
        pyReSpect.  Useful for reproducibility and for stashing on the
        measurement ``details``.

    Continuous relaxation spectrum:

    s : np.ndarray, shape (ns,)
        Relaxation-time grid in seconds.
    H : np.ndarray, shape (ns,)
        Log-CRS :math:`H(s) = \\ln h(s)` (dimensionless).
    h : np.ndarray, shape (ns,)
        Spectrum :math:`h(s) = \\exp(H(s))` in Pa.
    lam_C : float
        Optimal Tikhonov :math:`\\lambda`.
    G0 : float
        Fitted plateau modulus, or 0.0 when ``plateau=False``.
    G_fit_cont : np.ndarray
        Predicted :math:`G(t)` (time branch) or concatenated
        :math:`[G' \\| G'']` (frequency branch) from the continuous
        spectrum, in Pa.
    lam, rho, eta, log_P, H_lam, dH : np.ndarray or None
        L-curve / evidence / snapshot diagnostics.  Some are only
        populated by one of the two branches --- see
        :class:`~pyrespect_time.continuous.ContinuousResult` and
        :class:`~pyrespect_freq.continuous.ContinuousResult`.

    Discrete relaxation spectrum:

    g : np.ndarray, shape (N_opt,)
        Discrete mode weights :math:`g_i` in Pa.
    tau : np.ndarray, shape (N_opt,)
        Discrete relaxation times :math:`\\tau_i` in seconds.
    dtau : np.ndarray or None
        NLLS uncertainty estimates for :math:`\\tau_i` (time branch
        only); ``None`` for the frequency branch.
    N_opt : int
        Optimal number of discrete modes.
    G_fit_disc : np.ndarray
        Predicted response from the discrete spectrum.
    G0_disc : float
    wt_base, AIC_bst, N_bst : np.ndarray or None
        AIC-scan diagnostics.

    dataset_continuous, dataset_discrete : OneDimensionalCompositeDataset
        The two composite datasets prepended onto the input
        measurement's ``.datasets`` list.
    """

    domain: str
    config: dict

    # Continuous spectrum
    s: np.ndarray
    H: np.ndarray
    h: np.ndarray
    lam_C: float
    G0: float
    G_fit_cont: np.ndarray

    lam: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    eta: Optional[np.ndarray] = None
    log_P: Optional[np.ndarray] = None
    H_lam: Optional[np.ndarray] = None
    dH: Optional[np.ndarray] = None

    # Discrete spectrum
    g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    tau: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dtau: Optional[np.ndarray] = None
    N_opt: int = 0
    G_fit_disc: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    G0_disc: float = 0.0
    wt_base: Optional[np.ndarray] = None
    AIC_bst: Optional[np.ndarray] = None
    N_bst: Optional[np.ndarray] = None

    # Datasets stored back on the target measurement
    dataset_continuous: Optional[one_d_composite_dataset.OneDimensionalCompositeDataset] = None
    dataset_discrete: Optional[one_d_composite_dataset.OneDimensionalCompositeDataset] = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _find_first(names: Tuple[str, ...], catalogue: list) -> Optional[str]:
    for candidate in names:
        if candidate in catalogue:
            return candidate
    return None


def _sanitize_time(t: np.ndarray, g_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).ravel()
    g_t = np.asarray(g_t, dtype=float).ravel()
    mask = np.isfinite(t) & np.isfinite(g_t) & (t > 0.0) & (g_t > 0.0)
    t = t[mask]
    g_t = g_t[mask]
    if t.size < 5:
        raise ValueError(
            "RelaxationSpectrum(domain='time') requires at least 5 finite "
            "positive samples of the relaxation modulus."
        )
    order = np.argsort(t)
    t = t[order]
    g_t = g_t[order]
    # Deduplicate coincident time samples (some rheometers repeat step-time
    # values across segments); pyReSpect requires strictly monotonic input.
    t, idx = np.unique(t, return_index=True)
    g_t = g_t[idx]
    return t, g_t


def _sanitize_frequency(
    omega: np.ndarray, gp: np.ndarray, gpp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = np.asarray(omega, dtype=float).ravel()
    gp = np.asarray(gp, dtype=float).ravel()
    gpp = np.asarray(gpp, dtype=float).ravel()
    mask = (
        np.isfinite(omega)
        & np.isfinite(gp)
        & np.isfinite(gpp)
        & (omega > 0.0)
        & (gp > 0.0)
        & (gpp > 0.0)
    )
    omega = omega[mask]
    gp = gp[mask]
    gpp = gpp[mask]
    if omega.size < 5:
        raise ValueError(
            "RelaxationSpectrum(domain='frequency') requires at least 5 "
            "finite positive samples of (omega, G', G'')."
        )
    order = np.argsort(omega)
    omega = omega[order]
    gp = gp[order]
    gpp = gpp[order]
    omega, idx = np.unique(omega, return_index=True)
    gp = gp[idx]
    gpp = gpp[idx]
    return omega, gp, gpp


def _config_kwargs() -> Tuple[str, ...]:
    """The full set of ``ReSpectConfig`` kwargs shared by both branches."""
    return (
        "ns",
        "plateau",
        "freq_end",
        "lam_min",
        "lam_max",
        "lam_C",
        "lam_density",
        "SmFacLam",
        "max_num_modes",
        "delta_base_weight_dist",
        "min_tau_spacing",
        "resample",
        "n_resample",
    )


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------


class RelaxationSpectrum(MeasurementTransform):
    """Continuous and discrete relaxation spectra via pyReSpect.

    Wraps :class:`pyrespect_time.ReSpect` (``domain="time"``) and
    :class:`pyrespect_freq.ReSpect` (``domain="frequency"``) behind a
    single hermes-rheo transform.  Both solvers are vendored and imported
    on first use.

    Parameters
    ----------
    domain : {"time", "frequency"}, default "frequency"
        Which pyReSpect branch to invoke.  ``"time"`` operates on the
        relaxation modulus :math:`G(t)`; ``"frequency"`` operates on the
        oscillatory moduli :math:`G'(\\omega), G''(\\omega)`.
    plateau : bool, default False
        Fit a non-zero plateau modulus :math:`G_\\infty`.  Set to
        ``True`` for viscoelastic solids.
    ns : int, default 100
        Discretisation points on the :math:`s` axis.
    freq_end : {"lenient", "neutral", "strict"}, default "lenient"
        Extent of the :math:`s`-axis window relative to the data window.
    lam_min, lam_max : float
        L-curve search bounds for :math:`\\lambda`.
    lam_C : float or None
        Pre-specified :math:`\\lambda`; skips the L-curve when set.
    SmFacLam : float, default 0.0
        Smoothness nudge in :math:`[-1, 1]`.
    max_num_modes : int or None
        Cap on the number of discrete Maxwell modes.
    delta_base_weight_dist : float, default 0.2
        Step size for the AIC scan.
    min_tau_spacing : float, default 1.25
        Minimum :math:`\\tau_{i+1}/\\tau_i` before merging.
    resample : bool, default True
        Resample the input onto a geometric grid before fitting.
    n_resample : int, default 100
        Number of points for the geometric-grid resample.
    lam_density : int, default 2
        :math:`\\lambda` grid density per decade.  Frequency branch only
        (silently ignored in the time branch).
    time_name : str, default "time"
        Column name of the time axis in the input dataset.  Falls back
        to ``"step time"`` when the requested name is not present.
    relaxation_name : str, default "G_from_eta1"
        Column name of :math:`G(t)` in the input dataset.  Falls back
        to :data:`_RELAXATION_ALIASES` (``G_from_eta2``,
        ``relaxation modulus``, ``G_relax``, ``modulus``) so the
        transform composes cleanly with
        :class:`~hermes_rheo.transforms.frequency_to_time.FrequencyToTime`
        and with the TRIOS reader's ``modulus`` column.
    omega_name : str, default "angular frequency"
        Column name of the frequency axis in the input dataset.
    storage_name, loss_name : str
        Column names of :math:`G'` and :math:`G''` in the input
        dataset (defaults ``"storage modulus"``, ``"loss modulus"``).

    Examples
    --------
    Frequency-domain example on a real TRIOS frequency sweep:

    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.relaxation_spectrum import RelaxationSpectrum
    >>> exp = TriosRheoReader().data_from_filepath(
    ...     "frequency_sweep.txt", create_composite_datasets=True
    ... )  # doctest: +SKIP
    >>> out = RelaxationSpectrum(domain="frequency")(exp)  # doctest: +SKIP

    Time-domain example on synthetic multi-mode Maxwell data:

    >>> import numpy as np
    >>> from hermes_rheo.synthetic import SyntheticRheometry
    >>> model = SyntheticRheometry.multi_mode_maxwell(
    ...     g=[1e5, 1e4], tau=[0.1, 1.0]
    ... )
    >>> t = np.geomspace(1e-3, 1e2, 200)
    >>> meas = model.to_measurement(experiment="stress_relaxation", x=t)
    >>> out = RelaxationSpectrum(
    ...     domain="time",
    ...     relaxation_name="relaxation modulus",
    ... )(meas)  # doctest: +SKIP
    """

    def __init__(
        self,
        domain: str = _DOMAIN_FREQUENCY,
        *,
        plateau: bool = False,
        ns: int = 100,
        freq_end: str = "lenient",
        lam_min: float = 1e-10,
        lam_max: float = 1e3,
        lam_C: Optional[float] = None,
        SmFacLam: float = 0.0,
        max_num_modes: Optional[int] = None,
        delta_base_weight_dist: float = 0.2,
        min_tau_spacing: float = 1.25,
        resample: bool = True,
        n_resample: int = 100,
        lam_density: int = 2,
        time_name: str = "time",
        relaxation_name: str = "G_from_eta1",
        omega_name: str = "angular frequency",
        storage_name: str = "storage modulus",
        loss_name: str = "loss modulus",
    ) -> None:
        super().__init__()
        if domain not in _VALID_DOMAINS:
            raise ValueError(f"domain must be one of {_VALID_DOMAINS!r}, got {domain!r}")

        self.domain = domain
        self.plateau = plateau
        self.ns = ns
        self.freq_end = freq_end
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.lam_C = lam_C
        self.SmFacLam = SmFacLam
        self.max_num_modes = max_num_modes
        self.delta_base_weight_dist = delta_base_weight_dist
        self.min_tau_spacing = min_tau_spacing
        self.resample = resample
        self.n_resample = n_resample
        self.lam_density = lam_density

        self.time_name = time_name
        self.relaxation_name = relaxation_name
        self.omega_name = omega_name
        self.storage_name = storage_name
        self.loss_name = loss_name

        self._result: Optional[RelaxationSpectrumResult] = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[RelaxationSpectrumResult]:
        """Most recent :class:`RelaxationSpectrumResult` produced by ``_apply``."""
        return self._result

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _apply(self, target: Any, **kwargs: Any) -> Any:
        if not hasattr(target, "datasets") or len(target.datasets) == 0:
            return target

        dataset = target.datasets[0]
        if not hasattr(dataset, "data_array_names") or not hasattr(dataset, "data_arrays"):
            return target

        names = list(dataset.data_array_names)

        if self.domain == _DOMAIN_TIME:
            result = self._apply_time(dataset, names)
        else:
            result = self._apply_frequency(dataset, names)

        if result is None:
            return target

        # Prepend the two new spectrum datasets in front of the existing ones.
        # Continuous first, discrete second, then the original active dataset.
        prepended = [result.dataset_continuous, result.dataset_discrete]
        target.datasets = [ds for ds in prepended if ds is not None] + list(target.datasets)

        self._result = result
        try:
            if hasattr(target, "add_detail") and hasattr(target, "update_detail"):
                # Measurement.details is a deep-copy property; use the proper
                # setters instead of mutating the copy.
                if "relaxation_spectrum_result" in target.detail_names:
                    target.update_detail("relaxation_spectrum_result", result)
                else:
                    target.add_detail("relaxation_spectrum_result", result)
            elif hasattr(target, "details") and isinstance(target.details, dict):
                target.details["relaxation_spectrum_result"] = result
        except Exception:  # pragma: no cover - defensive: never fail the transform
            pass

        return target

    # ------------------------------------------------------------------
    # Domain-specific branches
    # ------------------------------------------------------------------

    def _apply_time(self, dataset: Any, names: list) -> Optional[RelaxationSpectrumResult]:
        time_col = self.time_name if self.time_name in names else _find_first(_TIME_ALIASES, names)
        modulus_col = (
            self.relaxation_name
            if self.relaxation_name in names
            else _find_first(_RELAXATION_ALIASES, names)
        )
        if time_col is None or modulus_col is None:
            return None  # silent no-op — required columns not present

        t_raw = np.asarray(dataset.data_arrays[names.index(time_col)], dtype=float)
        g_raw = np.asarray(dataset.data_arrays[names.index(modulus_col)], dtype=float)
        try:
            t, g_t = _sanitize_time(t_raw, g_raw)
        except ValueError:
            return None

        pr_mod = _require_pyrespect(_DOMAIN_TIME)
        kwarg_names = _config_kwargs()
        config = self._build_config(pr_mod, kwarg_names)
        solver = pr_mod.ReSpect(config)
        solver.fit(source=(t, g_t), resample=self.resample)

        cont = solver.continuous
        disc = solver.discrete
        return self._pack_result(
            domain=_DOMAIN_TIME,
            config=self._config_snapshot(config, kwarg_names),
            cont=cont,
            disc=disc,
        )

    def _apply_frequency(self, dataset: Any, names: list) -> Optional[RelaxationSpectrumResult]:
        for required in (self.omega_name, self.storage_name, self.loss_name):
            if required not in names:
                return None  # silent no-op

        omega_raw = np.asarray(dataset.data_arrays[names.index(self.omega_name)], dtype=float)
        gp_raw = np.asarray(dataset.data_arrays[names.index(self.storage_name)], dtype=float)
        gpp_raw = np.asarray(dataset.data_arrays[names.index(self.loss_name)], dtype=float)
        try:
            omega, gp, gpp = _sanitize_frequency(omega_raw, gp_raw, gpp_raw)
        except ValueError:
            return None

        # Refuse to run on effectively single-frequency oscillations (e.g. a
        # temperature ramp carrying a constant ω column).
        if float(omega.max()) / float(omega.min()) < 2.0:
            return None

        pr_mod = _require_pyrespect(_DOMAIN_FREQUENCY)
        kwarg_names = _config_kwargs()
        config = self._build_config(pr_mod, kwarg_names)
        solver = pr_mod.ReSpect(config)
        solver.fit(source=(omega, gp, gpp))

        cont = solver.continuous
        disc = solver.discrete
        return self._pack_result(
            domain=_DOMAIN_FREQUENCY,
            config=self._config_snapshot(config, kwarg_names),
            cont=cont,
            disc=disc,
        )

    # ------------------------------------------------------------------
    # Config plumbing
    # ------------------------------------------------------------------

    def _build_config(self, pr_mod: Any, kwarg_names: Tuple[str, ...]):
        payload = {name: getattr(self, name) for name in kwarg_names}
        return pr_mod.ReSpectConfig(**payload)

    def _config_snapshot(self, config: Any, kwarg_names: Tuple[str, ...]) -> dict:
        return {name: getattr(config, name) for name in kwarg_names}

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _pack_result(
        self,
        domain: str,
        config: dict,
        cont: Any,
        disc: Any,
    ) -> RelaxationSpectrumResult:
        # Both pyReSpect variants store the CRS on s / H; the freq variant
        # uses ``lamC`` / ``logP`` while the time variant uses ``lam_C`` /
        # ``log_P``.  Read whichever is present.
        s = np.asarray(cont.s, dtype=float)
        H = np.asarray(cont.H, dtype=float)
        h = np.exp(H)
        lam_C = float(getattr(cont, "lam_C", getattr(cont, "lamC", float("nan"))))
        G0_cont = float(getattr(cont, "G0", 0.0) or 0.0)
        G_fit_cont = np.asarray(cont.G_fit, dtype=float)

        lam = getattr(cont, "lam", None)
        rho = getattr(cont, "rho", None)
        eta = getattr(cont, "eta", None)
        log_P = getattr(cont, "log_P", getattr(cont, "logP", None))
        H_lam = getattr(cont, "H_lam", None)
        dH = getattr(cont, "dH", None)

        # Discrete result — attribute names diverge between the two branches.
        if disc is None:
            g = np.array([], dtype=float)
            tau = np.array([], dtype=float)
            dtau = None
            N_opt = 0
            G_fit_disc = np.array([], dtype=float)
            G0_disc = 0.0
            wt_base = None
            AIC_bst = None
            N_bst = None
        else:
            g = np.asarray(getattr(disc, "g", []), dtype=float)
            tau = np.asarray(getattr(disc, "tau", []), dtype=float)
            dtau_attr = getattr(disc, "dtau", None)
            dtau = None if dtau_attr is None else np.asarray(dtau_attr, dtype=float)
            N_opt = int(getattr(disc, "N_opt", getattr(disc, "N", g.size)))
            G_fit_disc = np.asarray(getattr(disc, "G_fit", []), dtype=float)
            G0_disc = float(getattr(disc, "G0", 0.0) or 0.0)
            wt_base = getattr(disc, "wt_base", getattr(disc, "wtBase", None))
            AIC_bst = getattr(disc, "AIC_bst", getattr(disc, "AICbst", None))
            N_bst = getattr(disc, "N_bst", getattr(disc, "nzNbst", None))
            wt_base = None if wt_base is None else np.asarray(wt_base, dtype=float)
            AIC_bst = None if AIC_bst is None else np.asarray(AIC_bst, dtype=float)
            N_bst = None if N_bst is None else np.asarray(N_bst, dtype=float)

            # Sort discrete modes by ascending tau so their storage matches
            # SyntheticRheometry.multi_mode_maxwell and other tools.
            if tau.size > 1:
                order = np.argsort(tau)
                tau = tau[order]
                g = g[order]
                if dtau is not None and dtau.size == tau.size:
                    dtau = dtau[order]

        dataset_cont = self._build_continuous_dataset(s, H, h, dH, domain)
        dataset_disc = self._build_discrete_dataset(tau, g, dtau, domain)

        return RelaxationSpectrumResult(
            domain=domain,
            config=config,
            s=s,
            H=H,
            h=h,
            lam_C=lam_C,
            G0=G0_cont,
            G_fit_cont=G_fit_cont,
            lam=None if lam is None else np.asarray(lam, dtype=float),
            rho=None if rho is None else np.asarray(rho, dtype=float),
            eta=None if eta is None else np.asarray(eta, dtype=float),
            log_P=None if log_P is None else np.asarray(log_P, dtype=float),
            H_lam=None if H_lam is None else np.asarray(H_lam, dtype=float),
            dH=None if dH is None else np.asarray(dH, dtype=float),
            g=g,
            tau=tau,
            dtau=dtau,
            N_opt=N_opt,
            G_fit_disc=G_fit_disc,
            G0_disc=G0_disc,
            wt_base=wt_base,
            AIC_bst=AIC_bst,
            N_bst=N_bst,
            dataset_continuous=dataset_cont,
            dataset_discrete=dataset_disc,
        )

    # ------------------------------------------------------------------
    # Dataset builders
    # ------------------------------------------------------------------

    def _build_continuous_dataset(
        self,
        s: np.ndarray,
        H: np.ndarray,
        h: np.ndarray,
        dH: Optional[np.ndarray],
        domain: str,
    ) -> one_d_composite_dataset.OneDimensionalCompositeDataset:
        arrays = [s, H, h]
        names = ["relaxation time", "H", "h"]
        units = ["s", "", "Pa"]
        if dH is not None and dH.shape == s.shape:
            arrays.append(dH)
            names.append("dH")
            units.append("")
        return one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=arrays,
            data_array_names=names,
            data_array_units=units,
            default_independent_name="relaxation time",
            default_dependent_name="h",
            source=f"Continuous relaxation spectrum (pyReSpect-{domain})",
        )

    def _build_discrete_dataset(
        self,
        tau: np.ndarray,
        g: np.ndarray,
        dtau: Optional[np.ndarray],
        domain: str,
    ) -> Optional[one_d_composite_dataset.OneDimensionalCompositeDataset]:
        if tau.size == 0:
            return None
        arrays = [tau, g]
        names = ["discrete relaxation time", "g"]
        units = ["s", "Pa"]
        if dtau is not None and dtau.shape == tau.shape:
            arrays.append(dtau)
            names.append("dtau")
            units.append("s")
        return one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=arrays,
            data_array_names=names,
            data_array_units=units,
            default_independent_name="discrete relaxation time",
            default_dependent_name="g",
            source=f"Discrete relaxation spectrum (pyReSpect-{domain})",
        )
