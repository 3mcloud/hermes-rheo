"""Rheological-model fitting of storage and loss moduli.

Two complementary fitting engines are exposed:

* :class:`MultiModeMaxwellModel` — a generalized Maxwell (Prony series)
  model fitted by non-negative least squares (default) or multistart
  nonlinear optimisation, with an optional rubbery-plateau modulus.
* :class:`FitModel` — a pipeline step that fits either the built-in
  generalized Maxwell model or one of the fractional/classical
  viscoelastic models provided by `pyRheo
  <https://github.com/mirandi1/pyRheo>`_ (Miranda-Valdez *et al.*,
  Digital Discovery 4, 1075, 2025; arXiv:2412.15941) to a single
  frequency sweep or a TTS master curve.

pyRheo is an optional dependency (GPL-3.0, not distributed on PyPI) and is
imported only when a pyRheo model is requested.  Install it with::

    pip install git+https://github.com/mirandi1/pyRheo.git
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, nnls

_TINY = np.finfo(float).tiny


class MultiModeMaxwellModel:
    r"""Generalized Maxwell (Prony series) model for :math:`G'(\omega)`, :math:`G''(\omega)`.

    The generalized Maxwell model represents the linear viscoelastic
    response as a sum of :math:`N` Maxwell elements (a spring
    :math:`G_i` in series with a dashpot, relaxation time
    :math:`\tau_i`) optionally in parallel with an equilibrium
    (rubbery-plateau) spring :math:`G_e`:

    .. math::

        G'(\omega)  &= G_e + \sum_{i=1}^{N} G_i
                       \frac{(\omega\tau_i)^2}{1 + (\omega\tau_i)^2}, \\
        G''(\omega) &= \sum_{i=1}^{N} G_i
                       \frac{\omega\tau_i}{1 + (\omega\tau_i)^2}.

    Two fitting strategies are provided:

    ``"nnls"`` (default)
        The relaxation times :math:`\tau_i` are held fixed on a
        logarithmic grid spanning the data and the mode moduli
        :math:`G_i \ge 0` are recovered by a single **non-negative
        least-squares** solve.  This is the standard, fast and robust
        way to fit a Prony series and requires no random restarts.
    ``"nonlinear"``
        Both :math:`G_i` and :math:`\tau_i` are optimised in log-space
        by multistart local minimisation (the historical behaviour), for
        compact fits with a small number of free modes.

    Parameters
    ----------
    n_modes : int, optional
        Number of Maxwell modes.  For ``mode="nnls"`` it defaults to
        ``modes_per_decade`` per frequency decade; for
        ``mode="nonlinear"`` it must be given.
    mode : {"nnls", "nonlinear"}, optional
        Fitting strategy.  Defaults to ``"nnls"``.
    equilibrium_modulus : bool, optional
        If *True*, fit an additional rubbery-plateau modulus
        :math:`G_e \ge 0` (useful for cross-linked/elastomeric
        materials).  Defaults to *False*.
    tau_range : tuple of float, optional
        ``(tau_min, tau_max)`` for the fixed relaxation-time grid
        (``mode="nnls"``).  Defaults to
        :math:`(1/\omega_{\max},\ 1/\omega_{\min})`.
    modes_per_decade : int, optional
        Density of the relaxation-time grid when ``n_modes`` is not given
        (``mode="nnls"``).  Defaults to ``2``.
    weighting : {"relative", "none"}, optional
        ``"relative"`` (default) weights each residual by ``1/|data|`` so
        the fit is balanced across the many decades typical of moduli;
        ``"none"`` uses ordinary residuals.
    cost_function : {"RSS", "MSE", "BIC"}, optional
        Reported goodness-of-fit metric (and the objective for
        ``mode="nonlinear"``).  Defaults to ``"RSS"``.
    initial_guesses, num_initial_guesses, minimization_algorithm, bounds, initial_guess
        Options for ``mode="nonlinear"`` (see Notes).

    Attributes
    ----------
    G_modes : numpy.ndarray
        Fitted mode moduli :math:`G_i` (Pa).
    tau_modes : numpy.ndarray
        Relaxation times :math:`\tau_i` (s).
    equilibrium_modulus_value : float
        Fitted :math:`G_e` (0 when ``equilibrium_modulus`` is *False*).
    parameters : numpy.ndarray
        Interleaved ``[G1, eta1, ...]`` with :math:`\eta_i = G_i \tau_i`
        (kept for backward compatibility).

    Examples
    --------
    >>> import numpy as np
    >>> from hermes_rheo.transforms.fit_model import MultiModeMaxwellModel
    >>> omega = np.logspace(-2, 2, 60)
    >>> model = MultiModeMaxwellModel(equilibrium_modulus=True).fit(
    ...     omega, *MultiModeMaxwellModel().fit(omega, omega, omega).predict(omega)
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        n_modes=None,
        mode="nnls",
        equilibrium_modulus=False,
        tau_range=None,
        modes_per_decade=2,
        weighting="relative",
        initial_guesses="random",
        cost_function="RSS",
        bounds="auto",
        num_initial_guesses=500,
        minimization_algorithm="Powell",
        initial_guess=None,
    ):
        self.n_modes = n_modes
        self.mode = str(mode).lower()
        self.equilibrium_modulus = bool(equilibrium_modulus)
        self.tau_range = tau_range
        self.modes_per_decade = modes_per_decade
        self.weighting = str(weighting).lower()
        self.cost_function = cost_function.upper()
        self.bounds = bounds
        self.num_initial_guesses = num_initial_guesses
        self.minimization_algorithm = minimization_algorithm
        self.initial_guesses = initial_guesses
        self.initial_guess = initial_guess

        # Fitted quantities
        self.G_modes = None
        self.tau_modes = None
        self.equilibrium_modulus_value = 0.0
        self.parameters = None
        self.rss = self.mse = self.bic = self.cost = None
        self.model = "GeneralizedMaxwell"

        self._y_true = None
        self._y_pred = None

    # ------------------------------------------------------------------
    # Forward model (vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _moduli(omega, Gs, taus, Ge=0.0):
        """Vectorised generalized-Maxwell moduli.

        Parameters
        ----------
        omega : array_like, shape (N,)
        Gs, taus : array_like, shape (M,)
        Ge : float

        Returns
        -------
        G_prime, G_double_prime : numpy.ndarray, shape (N,)
        """
        omega = np.asarray(omega, dtype=float)
        Gs = np.asarray(Gs, dtype=float)
        taus = np.asarray(taus, dtype=float)
        wt = np.outer(omega, taus)          # (N, M)
        denom = 1.0 + wt * wt
        g_prime = (Gs * wt * wt / denom).sum(axis=1) + Ge
        g_double_prime = (Gs * wt / denom).sum(axis=1)
        return g_prime, g_double_prime

    def predict(self, omega):
        """Predict ``(G', G'')`` at *omega* from the fitted parameters."""
        if self.G_modes is None:
            raise ValueError("Model parameters have not been fitted yet.")
        return self._moduli(omega, self.G_modes, self.tau_modes,
                            self.equilibrium_modulus_value)

    # ------------------------------------------------------------------
    # Relaxation-time grid + weights
    # ------------------------------------------------------------------

    def _make_taus(self, omega):
        """Return a fixed log-spaced relaxation-time grid for NNLS."""
        if self.tau_range is not None:
            tau_min, tau_max = self.tau_range
        else:
            tau_min = 1.0 / np.max(omega)
            tau_max = 1.0 / np.min(omega)
        n = self.n_modes
        if n is None:
            decades = max(np.log10(tau_max / tau_min), 1.0)
            n = int(np.ceil(self.modes_per_decade * decades)) + 1
        return np.logspace(np.log10(tau_min), np.log10(tau_max), int(n))

    def _weights(self, data):
        if self.weighting == "relative":
            return 1.0 / np.maximum(np.abs(data), _TINY)
        return np.ones_like(data)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _n_params(self):
        if self.mode == "nnls":
            n = len(self.tau_modes) if self.tau_modes is not None else (self.n_modes or 0)
            return n + (1 if self.equilibrium_modulus else 0)
        return 2 * (self.n_modes or 0)

    def _compute_metrics(self, residuals):
        residuals = np.asarray(residuals, dtype=float)
        rss = float(np.sum(residuals ** 2))
        mse = float(np.mean(residuals ** 2))
        n = residuals.size
        k = self._n_params()
        bic = n * np.log(max(rss, _TINY) / n) + k * np.log(n)
        return rss, mse, bic

    def _finalise(self, omega, Gp_data, Gpp_data):
        """Store predictions and goodness-of-fit metrics after a fit."""
        Gp_pred, Gpp_pred = self.predict(omega)
        self._y_true = np.concatenate([Gp_data, Gpp_data])
        self._y_pred = np.concatenate([Gp_pred, Gpp_pred])
        self.parameters = np.empty(2 * len(self.G_modes))
        self.parameters[0::2] = self.G_modes
        self.parameters[1::2] = self.G_modes * self.tau_modes  # eta = G*tau
        self.rss, self.mse, self.bic = self._compute_metrics(self._y_pred - self._y_true)
        self.cost = {"RSS": self.rss, "MSE": self.mse, "BIC": self.bic}[self.cost_function]
        return self

    # ------------------------------------------------------------------
    # Non-negative least-squares (NNLS) (linear Prony) fit 
    # ------------------------------------------------------------------

    def _fit_nnls(self, omega, Gp_data, Gpp_data):
        taus = self._make_taus(omega)
        wt = np.outer(omega, taus)
        denom = 1.0 + wt * wt
        a_prime = wt * wt / denom            # (N, M)  -> G'
        a_double = wt / denom                # (N, M)  -> G''

        if self.equilibrium_modulus:
            # G_e contributes to G' (column of ones) and not to G''.
            a_prime = np.hstack([a_prime, np.ones((len(omega), 1))])
            a_double = np.hstack([a_double, np.zeros((len(omega), 1))])

        design = np.vstack([a_prime, a_double])
        target = np.concatenate([Gp_data, Gpp_data])

        weights = np.concatenate([self._weights(Gp_data), self._weights(Gpp_data)])
        design_w = design * weights[:, None]
        target_w = target * weights

        coeffs, _ = nnls(design_w, target_w)

        if self.equilibrium_modulus:
            self.G_modes = coeffs[:-1]
            self.equilibrium_modulus_value = float(coeffs[-1])
        else:
            self.G_modes = coeffs
            self.equilibrium_modulus_value = 0.0
        self.tau_modes = taus
        return self._finalise(omega, Gp_data, Gpp_data)

    # ------------------------------------------------------------------
    # Nonlinear (free tau) fit
    # ------------------------------------------------------------------

    def _predict_from_log_params(self, omega, params_log):
        Gs = np.exp(params_log[0::2])
        taus = np.exp(params_log[1::2]) / Gs  # params store [logG, log(eta)] -> tau = eta/G
        return self._moduli(omega, Gs, taus, self.equilibrium_modulus_value)

    def _objective(self, params_log, omega, Gp_data, Gpp_data, w_p, w_pp):
        Gp_pred, Gpp_pred = self._predict_from_log_params(omega, params_log)
        residuals = np.concatenate([(Gp_pred - Gp_data) * w_p,
                                    (Gpp_pred - Gpp_data) * w_pp])
        rss, mse, bic = self._compute_metrics(residuals)
        return {"RSS": rss, "MSE": mse, "BIC": bic}[self.cost_function]

    def _make_initial_guess(self, omega, Gp_data):
        if self.initial_guess is not None:
            return np.asarray(self.initial_guess, dtype=float)
        n = self.n_modes
        g0 = np.log(np.full(n, max(np.mean(Gp_data) / n, 1e-12)))
        eta0 = np.log(np.full(n, max(np.mean(Gp_data) / np.mean(omega), 1e-12)))
        x0 = np.empty(2 * n)
        x0[0::2], x0[1::2] = g0, eta0
        if str(self.initial_guesses).lower() != "random":
            return x0
        x0 = x0.copy()
        x0[0::2] += np.random.uniform(-2, 2, size=n)
        x0[1::2] += np.random.uniform(-2, 2, size=n)
        return x0

    def _fit_nonlinear(self, omega, Gp_data, Gpp_data):
        if self.n_modes is None:
            raise ValueError("n_modes must be specified for mode='nonlinear'.")
        w_p = self._weights(Gp_data)
        w_pp = self._weights(Gpp_data)
        bounds = None if self.bounds == "auto" else self.bounds
        n_trials = self.num_initial_guesses if str(self.initial_guesses).lower() == "random" else 1

        best = None
        best_val = np.inf
        for _ in range(n_trials):
            x0 = self._make_initial_guess(omega, Gp_data)
            res = minimize(self._objective, x0,
                           args=(omega, Gp_data, Gpp_data, w_p, w_pp),
                           method=self.minimization_algorithm, bounds=bounds)
            if res.fun < best_val:
                best_val, best = res.fun, res
        if best is None:
            raise RuntimeError("Optimization failed for MultiModeMaxwellModel.")

        Gs = np.exp(best.x[0::2])
        self.G_modes = Gs
        self.tau_modes = np.exp(best.x[1::2]) / Gs
        return self._finalise(omega, Gp_data, Gpp_data)

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------

    def fit(self, omega, Gp_data, Gpp_data):
        """Fit the model to storage/loss modulus data.

        Parameters
        ----------
        omega : array_like
            Angular frequencies (rad/s).
        Gp_data, Gpp_data : array_like
            Storage and loss moduli (Pa).

        Returns
        -------
        MultiModeMaxwellModel
            ``self``, fitted.
        """
        omega = np.asarray(omega, dtype=float)
        Gp_data = np.asarray(Gp_data, dtype=float)
        Gpp_data = np.asarray(Gpp_data, dtype=float)
        if self.mode == "nnls":
            return self._fit_nnls(omega, Gp_data, Gpp_data)
        if self.mode == "nonlinear":
            return self._fit_nonlinear(omega, Gp_data, Gpp_data)
        raise ValueError("mode must be 'nnls' or 'nonlinear'.")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_parameters(self):
        if self.G_modes is None:
            raise ValueError("Model parameters have not been fitted yet.")
        n = len(self.G_modes)
        print(f"Generalized Maxwell ({n} modes, mode='{self.mode}') parameters:")
        if self.equilibrium_modulus:
            print(f"  Equilibrium modulus G_e = {self.equilibrium_modulus_value:.3e} Pa")
        for i, (G, tau) in enumerate(zip(self.G_modes, self.tau_modes), 1):
            print(f"  Mode {i}: G = {G:.3e} Pa, tau = {tau:.3e} s, eta = {G * tau:.3e} Pa.s")

    def print_error(self):
        if self._y_true is None or self._y_pred is None:
            raise AttributeError("No stored prediction data available for error calculation.")
        absolute_error = np.abs(self._y_true - self._y_pred)
        denom = np.where(self._y_true == 0, 1.0, self._y_true)
        mean_percentage_error = float(np.mean(absolute_error / denom) * 100.0)
        print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
        print(f"Cost ({self.cost_function}): {self.cost}")
        return self.cost


_PYRHEO_INSTALL_HINT = (
    "FitModel requires the optional dependency pyRheo for {} fits. "
    "Install it with:\n\n"
    "    pip install git+https://github.com/mirandi1/pyRheo.git\n\n"
    "or use model='custom', model_name='MultiModeMaxwell' for the built-in "
    "generalized Maxwell model."
)


def _load_pyrheo_model(domain):
    """Return the default pyRheo model class for *domain*."""
    try:
        if domain == "relaxation":
            from pyRheo.relaxation_model import RelaxationModel as model_class
        else:
            from pyRheo.oscillation_model import SAOSModel as model_class
    except ImportError as exc:
        raise ImportError(_PYRHEO_INSTALL_HINT.format(domain)) from exc
    return model_class


def _is_default_saos(model):
    return model is None or getattr(model, "__name__", None) == "SAOSModel"


class FitModel:
    """Fit a rheological model to storage/loss moduli from many input types.

    ``FitModel`` fits :math:`G'(\\omega)` and :math:`G''(\\omega)` with either
    a `pyRheo <https://github.com/mirandi1/pyRheo>`_ constitutive model
    (fractional and classical viscoelastic models) or the built-in
    :class:`MultiModeMaxwellModel` (generalized Maxwell / Prony series).

    The moduli can come from:

    * a **single dataset** — an isothermal frequency sweep or one
      temperature of a TTS experiment (pass a ``Measurement``,
      ``MeasurementSet``/``ExperimentSet`` — one dataset selected by
      ``measurement_index`` — or a dataset directly); or
    * a **master curve** — a
      :class:`~hermes_rheo.transforms.automated_mastercurve.CompositeMasterCurve`
      (the two moduli curves are interpolated onto a common grid).

        For ``domain="relaxation"``, the transform instead fits a time-domain
        relaxation curve :math:`G(t)` using pyRheo
        :class:`pyRheo.relaxation_model.RelaxationModel`.

    Because it accepts either, ``FitModel`` can be used stand-alone on a
    single sweep *or* chained after master-curve construction, e.g.
    ``OWChirpAnalysis() + CompositeMasterCurve(...) + FitModel(...)``.

    Parameters
    ----------
    model : callable, {"custom"} or None, optional
        A pyRheo model class or the string ``"custom"`` together with
        ``model_name="MultiModeMaxwell"`` for the built-in generalized
        Maxwell model.  ``None`` (default) selects pyRheo's
        :class:`~pyRheo.oscillation_model.SAOSModel` for
        ``domain="oscillation"`` and
        :class:`~pyRheo.relaxation_model.RelaxationModel` for
        ``domain="relaxation"``; pyRheo is then imported on first use.
    model_name : str, optional
        pyRheo model name (e.g. ``"FractionalZener"``) or
        ``"MultiModeMaxwell"``.  Defaults to ``"FractionalZener"``.
    n_modes, mode, equilibrium_modulus : optional
        Forwarded to :class:`MultiModeMaxwellModel` when
        ``model="custom"``.
    measurement_index : int, optional
        Which measurement to fit when the input is an
        ``ExperimentSet``/``MeasurementSet``.  Defaults to ``0``.
    dataset_index : int, optional
        Which dataset within the selected measurement to fit.  Defaults
        to ``0``.
    x_name, storage_name, loss_name : str, optional
        Data-array names for angular frequency and the two moduli.
    n_interp : int, optional
        Number of points on the common grid used when fitting a master
        curve.  Defaults to ``100``.
    domain : {"oscillation", "relaxation"}, optional
        Fitting domain. ``"oscillation"`` (default) fits
        :math:`G'(\omega), G''(\omega)`. ``"relaxation"`` fits
        :math:`G(t)`.
    time_name, relaxation_name : str, optional
        Data-array names used in ``domain="relaxation"`` mode.

    References
    ----------
    .. [pyRheo] Miranda-Valdez, I. Y. *et al.*, *pyRheo: an open-source
       Python package for complex rheology*, Digital Discovery 4, 1075
       (2025).  DOI: `10.1039/D4DD00378K
       <https://doi.org/10.1039/D4DD00378K>`_ —
       preprint `arXiv:2412.15941 <https://arxiv.org/abs/2412.15941>`_.
    """

    def __init__(
        self,
        model=None,
        model_name="FractionalZener",
        n_modes=8,
        mode="nnls",
        equilibrium_modulus=False,
        initial_guesses="random",
        cost_function="RSS",
        bounds="auto",
        num_initial_guesses=500,
        minimization_algorithm="L-BFGS-B",
        measurement_index=0,
        dataset_index=0,
        x_name="angular frequency",
        storage_name="storage modulus",
        loss_name="loss modulus",
        domain="oscillation",
        time_name="time",
        relaxation_name="G_from_eta1",
        n_interp=100,
    ):
        self.model = model
        self.model_name = model_name
        self.n_modes = n_modes
        self.mode = mode
        self.equilibrium_modulus = equilibrium_modulus
        self.initial_guesses = initial_guesses
        self.cost_function = cost_function
        self.bounds = bounds
        self.num_initial_guesses = num_initial_guesses
        self.minimization_algorithm = minimization_algorithm
        self.measurement_index = measurement_index
        self.dataset_index = dataset_index
        self.x_name = x_name
        self.storage_name = storage_name
        self.loss_name = loss_name
        self.domain = str(domain).lower()
        self.time_name = time_name
        self.relaxation_name = relaxation_name
        self.n_interp = n_interp
        self.fitted_model = None
        self.fit_params = None
        self.omega_common = None
        self.G_prime = None
        self.G_double_prime = None
        self.t_common = None
        self.G_relax = None

    # ------------------------------------------------------------------
    # Input adapters
    # ------------------------------------------------------------------

    @staticmethod
    def _xy(dataset, x_name, y_name):
        """Return ``(x, y)`` for two named arrays without mutating *dataset*."""
        ds = copy.deepcopy(dataset)
        ds.switch_coordinates(independent_name=x_name, dependent_name=y_name)
        return np.asarray(ds.x_values, dtype=float), np.asarray(ds.y_values, dtype=float)

    def _select_dataset(self, target):
        """Locate a single composite dataset from a variety of input objects."""
        if hasattr(target, "switch_coordinates"):          # a dataset already
            return target
        if hasattr(target, "measurements"):                # Experiment/MeasurementSet
            measurements = list(target.measurements)
            return measurements[self.measurement_index].datasets[self.dataset_index]
        if hasattr(target, "datasets"):                    # a Measurement
            return target.datasets[self.dataset_index]
        raise TypeError(
            f"FitModel could not locate a dataset to fit in a "
            f"{type(target).__name__}."
        )

    def _extract_from_measurements(self, target):
        dataset = self._select_dataset(target)
        omega, g_prime = self._xy(dataset, self.x_name, self.storage_name)
        _, g_double_prime = self._xy(dataset, self.x_name, self.loss_name)
        mask = np.isfinite(omega) & np.isfinite(g_prime) & np.isfinite(g_double_prime)
        order = np.argsort(omega[mask])
        return omega[mask][order], g_prime[mask][order], g_double_prime[mask][order]

    def _extract_from_composite(self, composite):
        for q in (self.storage_name, self.loss_name):
            if q not in composite.available_quantities:
                raise ValueError(
                    f"FitModel requires '{q}' in the CompositeMasterCurve quantities."
                )
        mc_storage = composite.get_mastercurve(self.storage_name)
        mc_loss = composite.get_mastercurve(self.loss_name)

        omega_s = np.concatenate([np.exp(a) for a in mc_storage.xtransformed])
        gp_s = np.concatenate([np.exp(a) for a in mc_storage.ytransformed])
        omega_l = np.concatenate([np.exp(a) for a in mc_loss.xtransformed])
        gpp_l = np.concatenate([np.exp(a) for a in mc_loss.ytransformed])

        os_, ol_ = np.argsort(omega_s), np.argsort(omega_l)
        omega_s, gp_s = omega_s[os_], gp_s[os_]
        omega_l, gpp_l = omega_l[ol_], gpp_l[ol_]

        omega_min = max(np.min(omega_s), np.min(omega_l))
        omega_max = min(np.max(omega_s), np.max(omega_l))
        omega_common = np.logspace(np.log10(omega_min), np.log10(omega_max), self.n_interp)

        interp_storage = interp1d(omega_s, gp_s, kind="linear",
                                  bounds_error=False, fill_value="extrapolate")
        interp_loss = interp1d(omega_l, gpp_l, kind="linear",
                               bounds_error=False, fill_value="extrapolate")
        return omega_common, interp_storage(omega_common), interp_loss(omega_common)

    def _extract_moduli(self, target):
        """Return ``(omega, G', G'')`` from a master curve or a single dataset."""
        if hasattr(target, "get_mastercurve") and hasattr(target, "available_quantities"):
            return self._extract_from_composite(target)
        return self._extract_from_measurements(target)

    def _extract_relaxation(self, target):
        """Return ``(t, G(t))`` from a single dataset selected from *target*."""
        dataset = self._select_dataset(target)
        t, g_relax = self._xy(dataset, self.time_name, self.relaxation_name)
        # pyRheo relaxation models build log-space initial guesses from the
        # observed modulus range, so non-positive points must be excluded.
        mask = np.isfinite(t) & np.isfinite(g_relax) & (t > 0) & (g_relax > 0)
        if np.count_nonzero(mask) < 5:
            raise ValueError(
                "FitModel(domain='relaxation') requires at least 5 finite positive "
                "(time, G(t)) points"
            )
        order = np.argsort(t[mask])
        return t[mask][order], g_relax[mask][order]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def __call__(self, target):
        if self.domain == "relaxation":
            t_common, g_relax = self._extract_relaxation(target)

            # An explicit SAOSModel is swapped for RelaxationModel so that
            # code written for oscillation fits keeps working here.
            if _is_default_saos(self.model):
                model_class = _load_pyrheo_model("relaxation")
            elif self.model == "custom" and self.model_name == "MultiModeMaxwell":
                raise ValueError(
                    "MultiModeMaxwell custom model is only valid for domain='oscillation'"
                )
            else:
                model_class = self.model

            model = model_class(
                model=self.model_name,
                initial_guesses=self.initial_guesses,
                cost_function=self.cost_function,
                bounds=self.bounds,
                num_initial_guesses=self.num_initial_guesses,
                minimization_algorithm=self.minimization_algorithm,
            )
            model.fit(t_common, g_relax)

            self.fitted_model = model
            self.fit_params = getattr(model, "parameters", None)
            self.t_common = t_common
            self.G_relax = g_relax
            self.omega_common = None
            self.G_prime = None
            self.G_double_prime = None
            return self

        omega_common, g_prime, g_double_prime = self._extract_moduli(target)

        if self.model == "custom" and self.model_name == "MultiModeMaxwell":
            model = MultiModeMaxwellModel(
                n_modes=self.n_modes,
                mode=self.mode,
                equilibrium_modulus=self.equilibrium_modulus,
                initial_guesses=self.initial_guesses,
                cost_function=self.cost_function,
                bounds=self.bounds,
                num_initial_guesses=self.num_initial_guesses,
                minimization_algorithm=self.minimization_algorithm,
            )
        else:
            model_class = (_load_pyrheo_model("oscillation")
                           if self.model is None else self.model)
            model = model_class(
                model=self.model_name,
                initial_guesses=self.initial_guesses,
                cost_function=self.cost_function,
                bounds=self.bounds,
                num_initial_guesses=self.num_initial_guesses,
                minimization_algorithm=self.minimization_algorithm,
            )

        model.fit(omega_common, g_prime, g_double_prime)

        self.fitted_model = model
        self.fit_params = getattr(model, "parameters", None)
        self.omega_common = omega_common
        self.G_prime = g_prime
        self.G_double_prime = g_double_prime
        self.t_common = None
        self.G_relax = None
        return self

    def apply_to(self, target, make_copy=False, **kwargs):
        return self.__call__(target)

    @property
    def model_fit_params(self):
        return self.fit_params

    @property
    def model_fit_object(self):
        return self.fitted_model

    @property
    def model_fit_plot(self):
        return self.plot()

    def plot(self, *args, **kwargs):
        """Plot the fitted moduli against the data.

        Delegates to
        :func:`hermes_rheo.visualization.fit.plot_model_fit`.  Accepts
        ``show_residuals``, ``ax``, ``figsize``, ``xlabel``, ``ylabel``,
        ``title``, ``xlim``, ``ylim`` and, for backward compatibility,
        ``savefig``/``format``/``dpi``.  Returns ``(figure, axes)``.
        """
        from hermes_rheo.visualization.fit import plot_model_fit

        savefig = kwargs.pop("savefig", None)
        fmt = kwargs.pop("format", "png")
        dpi = kwargs.pop("dpi", 300)
        show = kwargs.pop("show", True)

        fig, ax = plot_model_fit(self, show=False, **kwargs)
        if savefig is not None:
            fig.savefig(savefig, format=fmt, dpi=dpi)
        if show:
            plt.show()
        return fig, ax

    def print_parameters(self, *args, **kwargs):
        if self.fitted_model is not None and hasattr(self.fitted_model, 'print_parameters'):
            return self.fitted_model.print_parameters(*args, **kwargs)
        else:
            raise AttributeError("No fitted model available or print_parameters method not found.")

    def print_error(self, *args, **kwargs):
        if self.fitted_model is not None and hasattr(self.fitted_model, 'print_error'):
            return self.fitted_model.print_error(*args, **kwargs)
        else:
            raise AttributeError("No fitted model available or print_error method not found.")