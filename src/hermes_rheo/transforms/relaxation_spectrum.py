"""Continuous and discrete relaxation spectra via pyReSpect.

Two vendored pyReSpect solvers are exposed through one
``MeasurementTransform``:

* ``domain="time"``: pyReSpect-time [Shanbhag2019]_ computes the continuous
  spectrum :math:`H(s)` from a relaxation modulus :math:`G(t)` by Tikhonov
  regularisation with Bayesian selection of the regularisation parameter,
  and a discrete spectrum :math:`(g_i, \\tau_i)` by AIC minimisation.
* ``domain="frequency"``: pyReSpect-freq [Shanbhag2020]_ [TakehShanbhag2013]_
  computes the same quantities from :math:`G'(\\omega)` and
  :math:`G''(\\omega)`.

The solvers are vendored in :mod:`hermes_rheo._vendor` (MIT licence, see
``_vendor/README.md``). The inverse problems are

.. math::

   G(t) = G_\\infty + \\int_{-\\infty}^{\\infty} H(s) e^{-t/s} \\, d\\ln s

.. math::

   G'(\\omega) = G_\\infty + \\int_{-\\infty}^{\\infty}
       \\frac{\\omega^2 s^2}{1 + \\omega^2 s^2} H(s)\\, d\\ln s

.. math::

   G''(\\omega) = \\int_{-\\infty}^{\\infty}
       \\frac{\\omega s}{1 + \\omega^2 s^2} H(s)\\, d\\ln s

The transform prepends two composite datasets to the measurement:

1. ``relaxation time`` (s), ``H`` (log spectrum, dimensionless),
   ``h = exp(H)`` (Pa) and, for the frequency branch, ``dH``.
2. ``discrete relaxation time`` (s), ``g`` (Pa) and, for the time branch,
   ``dtau`` (s).

The full solver output is stored in
``details["relaxation_spectrum_result"]`` as a
:class:`RelaxationSpectrumResult`; the method, domain, solver settings and
references are stored in ``details["relaxation_spectrum"]``.

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
   (2013). DOI: `10.3933/applrheol-23-24628
   <https://doi.org/10.3933/applrheol-23-24628>`_.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset as one_d_composite_dataset
from piblin.transform.abc.measurement_transform import MeasurementTransform

__all__ = ["RelaxationSpectrum", "RelaxationSpectrumResult"]


_DOMAIN_TIME = "time"
_DOMAIN_FREQUENCY = "frequency"
_VALID_DOMAINS = (_DOMAIN_TIME, _DOMAIN_FREQUENCY)

# Fallback column names for the relaxation modulus; the first match wins.
# ``G_from_eta1`` / ``G_from_eta2`` are written by FrequencyToTime.
_RELAXATION_ALIASES = (
    "G_from_eta1",
    "G_from_eta2",
    "relaxation modulus",
    "G_relax",
    "modulus",
)

_TIME_ALIASES = ("time", "step time")

_REFERENCES = {
    _DOMAIN_TIME: (
        "Shanbhag, Macromol. Theory Simul. 28, 1900005 (2019), doi:10.1002/mats.201900005",
    ),
    _DOMAIN_FREQUENCY: (
        "Shanbhag, Rheol. Acta 59, 509 (2020), doi:10.1007/s00397-020-01212-w",
        "Takeh and Shanbhag, Appl. Rheol. 23, 24628 (2013), doi:10.3933/applrheol-23-24628",
    ),
}


def _load_pyrespect(domain: str):
    """Return the vendored pyReSpect module for *domain*."""
    if domain == _DOMAIN_TIME:
        from hermes_rheo._vendor import pyrespect_time

        return pyrespect_time
    if domain == _DOMAIN_FREQUENCY:
        from hermes_rheo._vendor import pyrespect_freq

        return pyrespect_freq
    raise ValueError(f"Unknown domain {domain!r}")  # pragma: no cover


@dataclass
class RelaxationSpectrumResult:
    """Unified container for pyReSpect-time and pyReSpect-freq outputs.

    Attributes
    ----------
    domain : {"time", "frequency"}
        Which pyReSpect branch produced the result.
    config : dict
        ``ReSpectConfig`` values passed to pyReSpect.

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
        L-curve and evidence diagnostics. Some are populated by only one
        branch; see the ``ContinuousResult`` classes in ``pyrespect_time``
        and ``pyrespect_freq``.

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
        Plateau modulus of the discrete fit, in Pa.
    wt_base, AIC_bst, N_bst : np.ndarray or None
        AIC-scan diagnostics.

    dataset_continuous, dataset_discrete : OneDimensionalCompositeDataset
        The datasets prepended to the measurement; the discrete one is
        ``None`` when no modes were found.
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

    lam: np.ndarray | None = None
    rho: np.ndarray | None = None
    eta: np.ndarray | None = None
    log_P: np.ndarray | None = None
    H_lam: np.ndarray | None = None
    dH: np.ndarray | None = None

    # Discrete spectrum
    g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    tau: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dtau: np.ndarray | None = None
    N_opt: int = 0
    G_fit_disc: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    G0_disc: float = 0.0
    wt_base: np.ndarray | None = None
    AIC_bst: np.ndarray | None = None
    N_bst: np.ndarray | None = None

    # Datasets stored back on the target measurement
    dataset_continuous: one_d_composite_dataset.OneDimensionalCompositeDataset | None = None
    dataset_discrete: one_d_composite_dataset.OneDimensionalCompositeDataset | None = None


def _first_present(candidates: tuple[str, ...], available: list) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _clean_time_series(t: np.ndarray, g_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    # Repeated time values (e.g. across segments) break pyReSpect's
    # requirement of strictly increasing input.
    t, idx = np.unique(t, return_index=True)
    g_t = g_t[idx]
    return t, g_t


def _clean_frequency_series(
    omega: np.ndarray, gp: np.ndarray, gpp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ``ReSpectConfig`` fields shared by both branches.
_CONFIG_FIELDS = (
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


class RelaxationSpectrum(MeasurementTransform):
    """Continuous and discrete relaxation spectra via pyReSpect.

    Uses ``pyrespect_time`` for ``domain="time"`` [Shanbhag2019]_ and
    ``pyrespect_freq`` for ``domain="frequency"`` [Shanbhag2020]_
    [TakehShanbhag2013]_. If the required columns are missing or fewer
    than five valid points remain, the measurement is returned unchanged.

    Parameters
    ----------
    domain : {"time", "frequency"}, default "frequency"
        ``"time"`` uses the relaxation modulus :math:`G(t)`; ``"frequency"``
        uses :math:`G'(\\omega)` and :math:`G''(\\omega)`.
    plateau : bool, default False
        Fit a plateau modulus :math:`G_\\infty` (viscoelastic solids).
    ns : int, default 100
        Discretisation points on the :math:`s` axis.
    freq_end : {"lenient", "neutral", "strict"}, default "lenient"
        Width of the :math:`s` window relative to the data window.
    lam_min, lam_max : float
        Search bounds for the regularisation parameter :math:`\\lambda`.
    lam_C : float or None
        Fixed :math:`\\lambda`; skips the :math:`\\lambda` search when set.
    SmFacLam : float, default 0.0
        Smoothing shift of the selected :math:`\\lambda`, in :math:`[-1, 1]`.
    max_num_modes : int or None
        Maximum number of discrete modes.
    delta_base_weight_dist : float, default 0.2
        Step size of the base-weight scan in the AIC search.
    min_tau_spacing : float, default 1.25
        Minimum ratio :math:`\\tau_{i+1}/\\tau_i`; closer modes are merged.
    resample : bool, default True
        Resample the input onto a geometric grid before fitting (applied
        in the time branch only).
    n_resample : int, default 100
        Number of points of the resampled grid.
    lam_density : int, default 2
        :math:`\\lambda` grid points per decade.
    time_name : str, default "time"
        Time column; falls back to ``"step time"``.
    relaxation_name : str, default "G_from_eta1"
        :math:`G(t)` column. Falls back to ``G_from_eta1`` / ``G_from_eta2``
        (written by
        :class:`~hermes_rheo.transforms.frequency_to_time.FrequencyToTime`),
        ``relaxation modulus``, ``G_relax`` and ``modulus``.
    omega_name : str, default "angular frequency"
        Angular-frequency column (frequency branch).
    storage_name, loss_name : str
        :math:`G'` and :math:`G''` columns (defaults ``"storage modulus"``,
        ``"loss modulus"``).

    Notes
    -----
    Input is expected in SI units (s, rad/s, Pa). Non-positive and
    non-finite samples are dropped. In the frequency branch the data must
    span at least a factor of 2 in :math:`\\omega`.

    Examples
    --------
    Frequency sweep from a TRIOS file:

    >>> from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
    >>> from hermes_rheo.transforms.relaxation_spectrum import RelaxationSpectrum
    >>> exp = TriosRheoReader().data_from_filepath(
    ...     "frequency_sweep.txt", create_composite_datasets=True
    ... )  # doctest: +SKIP
    >>> out = RelaxationSpectrum(domain="frequency")(exp)  # doctest: +SKIP

    Relaxation modulus simulated with
    :class:`hermes_rheo.lve_models.MultiModeMaxwellModel`:

    >>> import numpy as np
    >>> from hermes_rheo.lve_models import MultiModeMaxwellModel, Protocols
    >>> model = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
    >>> t = np.geomspace(1e-3, 1e2, 200)
    >>> meas = Protocols.stress_relaxation(model, t)
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
        lam_C: float | None = None,
        SmFacLam: float = 0.0,
        max_num_modes: int | None = None,
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

        self._result: RelaxationSpectrumResult | None = None

    @property
    def result(self) -> RelaxationSpectrumResult | None:
        """Result of the most recent application, or ``None``."""
        return self._result

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

        prepended = [result.dataset_continuous, result.dataset_discrete]
        target.datasets = [ds for ds in prepended if ds is not None] + list(target.datasets)

        self._result = result
        provenance = {
            "method": f"pyReSpect-{result.domain}",
            "domain": result.domain,
            "settings": dict(result.config),
            "references": list(_REFERENCES[result.domain]),
        }
        for key, value in (
            ("relaxation_spectrum_result", result),
            ("relaxation_spectrum", provenance),
        ):
            # Measurement.details returns a copy, so use the setters.
            target.update_detail(key, value)

        return target

    def _apply_time(self, dataset: Any, names: list) -> RelaxationSpectrumResult | None:
        time_col = (
            self.time_name if self.time_name in names else _first_present(_TIME_ALIASES, names)
        )
        modulus_col = (
            self.relaxation_name
            if self.relaxation_name in names
            else _first_present(_RELAXATION_ALIASES, names)
        )
        if time_col is None or modulus_col is None:
            return None

        t_raw = np.asarray(dataset.data_arrays[names.index(time_col)], dtype=float)
        g_raw = np.asarray(dataset.data_arrays[names.index(modulus_col)], dtype=float)
        try:
            t, g_t = _clean_time_series(t_raw, g_raw)
        except ValueError:
            return None

        pr_mod = _load_pyrespect(_DOMAIN_TIME)
        config = self._build_config(pr_mod)
        solver = pr_mod.ReSpect(config)
        solver.fit(source=(t, g_t), resample=self.resample)
        return self._pack_result(
            _DOMAIN_TIME, self._config_snapshot(config), solver.continuous, solver.discrete
        )

    def _apply_frequency(self, dataset: Any, names: list) -> RelaxationSpectrumResult | None:
        for required in (self.omega_name, self.storage_name, self.loss_name):
            if required not in names:
                return None

        omega_raw = np.asarray(dataset.data_arrays[names.index(self.omega_name)], dtype=float)
        gp_raw = np.asarray(dataset.data_arrays[names.index(self.storage_name)], dtype=float)
        gpp_raw = np.asarray(dataset.data_arrays[names.index(self.loss_name)], dtype=float)
        try:
            omega, gp, gpp = _clean_frequency_series(omega_raw, gp_raw, gpp_raw)
        except ValueError:
            return None

        # Skip near-single-frequency data (e.g. a temperature ramp at fixed omega).
        if float(omega.max()) / float(omega.min()) < 2.0:
            return None

        pr_mod = _load_pyrespect(_DOMAIN_FREQUENCY)
        config = self._build_config(pr_mod)
        solver = pr_mod.ReSpect(config)
        solver.fit(source=(omega, gp, gpp))
        return self._pack_result(
            _DOMAIN_FREQUENCY, self._config_snapshot(config), solver.continuous, solver.discrete
        )

    def _build_config(self, pr_mod: Any):
        return pr_mod.ReSpectConfig(**{name: getattr(self, name) for name in _CONFIG_FIELDS})

    @staticmethod
    def _config_snapshot(config: Any) -> dict:
        return {name: getattr(config, name) for name in _CONFIG_FIELDS}

    def _pack_result(
        self,
        domain: str,
        config: dict,
        cont: Any,
        disc: Any,
    ) -> RelaxationSpectrumResult:
        # The two solvers use different attribute names (lam_C / lamC,
        # log_P / logP, ...); read whichever exists.
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

            # Ascending tau, as in lve_models.MultiModeMaxwellModel.
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

    def _build_continuous_dataset(
        self,
        s: np.ndarray,
        H: np.ndarray,
        h: np.ndarray,
        dH: np.ndarray | None,
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
        dtau: np.ndarray | None,
        domain: str,
    ) -> one_d_composite_dataset.OneDimensionalCompositeDataset | None:
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
