"""Model fitting of linear viscoelastic moduli.

* :class:`MultiModeMaxwellModel`: generalized Maxwell (Prony series) fit
  of :math:`G'(\\omega)` and :math:`G''(\\omega)` by non-negative least
  squares (default) or multistart nonlinear optimisation. This class *fits*
  data; to *simulate* a multimode Maxwell material use
  :class:`hermes_rheo.lve_models.MultiModeMaxwellModel`.
* :class:`FitModel`: piblin transform that fits the above or a classical or
  fractional model from pyRheo to one frequency sweep, a master curve, or
  (``domain="relaxation"``) a relaxation curve :math:`G(t)`. It returns a
  piblin ``Measurement`` holding measured and fitted data, with the fitted
  parameters and goodness of fit in ``details``.

Classical models (Maxwell, Kelvin-Voigt, Zener) combine springs and
dashpots. Fractional models replace the dashpot by a spring-pot, whose
stress is proportional to a fractional derivative of the strain of order
:math:`\\alpha` (0 = elastic, 1 = viscous). They describe the power-law
response of soft materials with few parameters.

pyRheo is an optional dependency (GPL-3.0, not distributed on PyPI) and is
imported only when a pyRheo model is requested. Install it with::

    pip install git+https://github.com/mirandi1/pyRheo.git

References
----------
.. [Bonfanti2020] Bonfanti, A., Kaplan, J. L., Charras, G., Kabla, A.,
   *Fractional viscoelastic models for power-law materials*, Soft Matter
   16(26), 6002-6020 (2020). DOI: `10.1039/d0sm00354a
   <https://doi.org/10.1039/d0sm00354a>`_.
.. [Song2023] Song, J., Holten-Andersen, N., McKinley, G. H.,
   *Non-Maxwellian viscoelastic stress relaxations in soft matter*, Soft
   Matter 19(41), 7885-7906 (2023). DOI: `10.1039/d3sm00736g
   <https://doi.org/10.1039/d3sm00736g>`_.
.. [pyRheo] Miranda-Valdez, I. Y., Niinisto, J., Makinen, T., Lejon, J.,
   Koivisto, J., Alava, M. J., *pyRheo: an open-source Python package for
   complex rheology*, Digital Discovery 4(4), 1075-1082 (2025). DOI:
   `10.1039/d5dd00021a <https://doi.org/10.1039/d5dd00021a>`_.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
from piblin.data.data_collections import measurement as _measurement
from piblin.data.datasets.abc.split_datasets import (
    one_dimensional_composite_dataset as _composite,
)
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform
from scipy.interpolate import interp1d
from scipy.optimize import minimize, nnls

from hermes_rheo import __version__

_TINY = np.finfo(float).tiny


class MultiModeMaxwellModel:
    r"""Fit a generalized Maxwell (Prony series) model to :math:`G'(\omega)`, :math:`G''(\omega)`.

    This class fits data. To simulate a multimode Maxwell material see
    :class:`hermes_rheo.lve_models.MultiModeMaxwellModel`.

    The model is a sum of :math:`N` Maxwell elements (modulus :math:`G_i`,
    relaxation time :math:`\tau_i`) in parallel with an optional equilibrium
    spring :math:`G_e`:

    .. math::

        G'(\omega)  &= G_e + \sum_{i=1}^{N} G_i
                       \frac{(\omega\tau_i)^2}{1 + (\omega\tau_i)^2}, \\
        G''(\omega) &= \sum_{i=1}^{N} G_i
                       \frac{\omega\tau_i}{1 + (\omega\tau_i)^2}.

    Fit modes:

    ``"nnls"`` (default)
        The :math:`\tau_i` are fixed on a log-spaced grid and the
        :math:`G_i \ge 0` follow from one non-negative least-squares solve.
    ``"nonlinear"``
        :math:`G_i` and :math:`\tau_i` are optimised in log space by
        multistart local minimisation.

    Parameters
    ----------
    n_modes : int, optional
        Number of modes. For ``mode="nnls"`` the default is
        ``modes_per_decade`` per decade of the :math:`\tau` grid; required
        for ``mode="nonlinear"``.
    mode : {"nnls", "nonlinear"}, optional
        Fit mode. Default ``"nnls"``.
    equilibrium_modulus : bool, optional
        Fit :math:`G_e \ge 0` (``mode="nnls"`` only). Default False.
    tau_range : tuple of float, optional
        ``(tau_min, tau_max)`` of the :math:`\tau` grid for ``"nnls"``.
        Default :math:`(1/\omega_{\max}, 1/\omega_{\min})`.
    modes_per_decade : int, optional
        Grid density when ``n_modes`` is None. Default 2.
    weighting : {"relative", "none"}, optional
        ``"relative"`` (default) weights residuals by ``1/|data|``.
    cost_function : {"RSS", "MSE", "BIC"}, optional
        Reported fit metric, and the objective for ``"nonlinear"``.
        Default ``"RSS"``.
    initial_guesses, num_initial_guesses, minimization_algorithm, bounds, initial_guess
        Options for ``mode="nonlinear"``: ``initial_guesses="random"``
        draws ``num_initial_guesses`` perturbed starts and keeps the best;
        ``bounds="auto"`` means unbounded; ``initial_guess`` is an explicit
        start vector ``[log G_1, log eta_1, ...]``.

    Attributes
    ----------
    G_modes : numpy.ndarray
        Fitted :math:`G_i` (Pa).
    tau_modes : numpy.ndarray
        Fitted :math:`\tau_i` (s).
    equilibrium_modulus_value : float
        Fitted :math:`G_e` (Pa); 0 if not fitted.
    parameters : numpy.ndarray
        Interleaved ``[G_1, eta_1, G_2, eta_2, ...]`` with
        :math:`\eta_i = G_i \tau_i` (Pa s).
    rss, mse, bic, cost : float
        Fit metrics on the concatenated :math:`G'` and :math:`G''` residuals;
        ``cost`` is the one selected by ``cost_function``.

    Examples
    --------
    >>> import numpy as np
    >>> omega = np.logspace(-2, 2, 60)
    >>> g_prime, g_double_prime = MultiModeMaxwellModel._moduli(omega, [1e3], [1.0])
    >>> model = MultiModeMaxwellModel().fit(omega, g_prime, g_double_prime)  # doctest: +SKIP
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

    @staticmethod
    def _moduli(omega, Gs, taus, Ge=0.0):
        """Return ``(G', G'')`` of a generalized Maxwell model.

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
        wt = np.outer(omega, taus)
        denom = 1.0 + wt * wt
        g_prime = (Gs * wt * wt / denom).sum(axis=1) + Ge
        g_double_prime = (Gs * wt / denom).sum(axis=1)
        return g_prime, g_double_prime

    def predict(self, omega):
        """Predict ``(G', G'')`` at *omega* from the fitted parameters."""
        if self.G_modes is None:
            raise ValueError("Model parameters have not been fitted yet.")
        return self._moduli(omega, self.G_modes, self.tau_modes, self.equilibrium_modulus_value)

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
        """Return residual weights for *data* according to ``weighting``."""
        if self.weighting == "relative":
            return 1.0 / np.maximum(np.abs(data), _TINY)
        return np.ones_like(data)

    def _n_params(self):
        if self.mode == "nnls":
            n = len(self.tau_modes) if self.tau_modes is not None else (self.n_modes or 0)
            return n + (1 if self.equilibrium_modulus else 0)
        return 2 * (self.n_modes or 0)

    def _compute_metrics(self, residuals):
        residuals = np.asarray(residuals, dtype=float)
        rss = float(np.sum(residuals**2))
        mse = float(np.mean(residuals**2))
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

    def _fit_nnls(self, omega, Gp_data, Gpp_data):
        """Fit ``G_i`` (and ``G_e``) on a fixed tau grid by NNLS."""
        taus = self._make_taus(omega)
        wt = np.outer(omega, taus)
        denom = 1.0 + wt * wt
        a_prime = wt * wt / denom
        a_double = wt / denom

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

    def _predict_from_log_params(self, omega, params_log):
        Gs = np.exp(params_log[0::2])
        taus = np.exp(params_log[1::2]) / Gs  # params store [logG, log(eta)] -> tau = eta/G
        return self._moduli(omega, Gs, taus, self.equilibrium_modulus_value)

    def _objective(self, params_log, omega, Gp_data, Gpp_data, w_p, w_pp):
        Gp_pred, Gpp_pred = self._predict_from_log_params(omega, params_log)
        residuals = np.concatenate([(Gp_pred - Gp_data) * w_p, (Gpp_pred - Gpp_data) * w_pp])
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
        """Fit ``G_i`` and ``tau_i`` by multistart minimisation in log space."""
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
            res = minimize(
                self._objective,
                x0,
                args=(omega, Gp_data, Gpp_data, w_p, w_pp),
                method=self.minimization_algorithm,
                bounds=bounds,
            )
            if res.fun < best_val:
                best_val, best = res.fun, res
        if best is None:
            raise RuntimeError("Optimization failed for MultiModeMaxwellModel.")

        Gs = np.exp(best.x[0::2])
        self.G_modes = Gs
        self.tau_modes = np.exp(best.x[1::2]) / Gs
        return self._finalise(omega, Gp_data, Gpp_data)

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

    def print_parameters(self):
        """Print the fitted ``G_e`` (if any) and each mode's G, tau and eta."""
        if self.G_modes is None:
            raise ValueError("Model parameters have not been fitted yet.")
        n = len(self.G_modes)
        print(f"Generalized Maxwell ({n} modes, mode='{self.mode}') parameters:")
        if self.equilibrium_modulus:
            print(f"  Equilibrium modulus G_e = {self.equilibrium_modulus_value:.3e} Pa")
        for i, (G, tau) in enumerate(zip(self.G_modes, self.tau_modes), 1):
            print(f"  Mode {i}: G = {G:.3e} Pa, tau = {tau:.3e} s, eta = {G * tau:.3e} Pa.s")

    def print_error(self):
        """Print the mean percentage error and the cost; return the cost."""
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


_REFERENCES = [
    "Bonfanti, Kaplan, Charras, Kabla, Soft Matter 16(26), 6002-6020 (2020), "
    "doi:10.1039/d0sm00354a",
    "Song, Holten-Andersen, McKinley, Soft Matter 19(41), 7885-7906 (2023), doi:10.1039/d3sm00736g",
    "Miranda-Valdez et al., Digital Discovery 4(4), 1075-1082 (2025), doi:10.1039/d5dd00021a",
]


def _is_default_saos(model):
    return model is None or getattr(model, "__name__", None) == "SAOSModel"


class FitModel(MeasurementSetTransform):
    r"""Fit a rheological model to linear viscoelastic data.

    Fits :math:`G'(\omega)` and :math:`G''(\omega)` (or, with
    ``domain="relaxation"``, :math:`G(t)`) with a classical or fractional
    model from pyRheo, or with the built-in :class:`MultiModeMaxwellModel`
    (a Prony series, :math:`G(t) = G_e + \sum_i G_i e^{-t/\tau_i}`).
    Fractional models use spring-pots of order :math:`\alpha`, so a few
    parameters describe power-law relaxation. Parameters are found by
    minimising a cost (RSS, MSE or BIC) of the model against the data from
    many random starts. The simulator of the same name,
    :class:`hermes_rheo.lve_models.MultiModeMaxwellModel`, generates data
    instead.

    The input may be a ``Measurement``, a ``MeasurementSet`` or
    ``ExperimentSet`` (``measurement_index`` picks one measurement), a
    dataset, or a
    :class:`~hermes_rheo.transforms.automated_mastercurve.CompositeMasterCurve`
    whose storage and loss curves are interpolated onto a common log grid.
    ``domain="relaxation"`` fits one dataset, such as the output of
    :class:`~hermes_rheo.transforms.frequency_to_time.FrequencyToTime`.

    Returns
    -------
    piblin.data.data_collections.measurement.Measurement
        One :class:`OneDimensionalCompositeDataset`. Oscillation arrays:
        ``angular frequency`` (rad/s), ``storage modulus``, ``loss modulus``,
        ``fitted storage modulus``, ``fitted loss modulus`` (Pa). Relaxation
        arrays: ``time`` (s), ``relaxation modulus``,
        ``fitted relaxation modulus`` (Pa). ``conditions`` hold ``fit_model``
        and ``fitted``. ``details`` hold ``model``, ``parameters``,
        ``goodness_of_fit`` (cost, metric, RSS and R-squared; parameter
        uncertainties are not estimated), ``method``, ``domain``,
        ``references`` and ``software``.

    The fitted model stays available as :attr:`fitted_model`,
    :attr:`fit_params` and :attr:`fit_measurement`, and the fitted data as
    ``omega_common``, ``G_prime``, ``G_double_prime`` or ``t_common``,
    ``G_relax``.

    Parameters
    ----------
    model : callable, {"custom"} or None, optional
        pyRheo model class, or ``"custom"`` with
        ``model_name="MultiModeMaxwell"`` for the built-in fit. None
        (default) selects ``pyRheo.oscillation_model.SAOSModel`` or, for
        ``domain="relaxation"``, ``pyRheo.relaxation_model.RelaxationModel``.
    model_name : str, optional
        pyRheo model name (e.g. ``"FractionalZener"``) or
        ``"MultiModeMaxwell"``. Default ``"FractionalZener"``.
    n_modes, mode, equilibrium_modulus : optional
        Passed to :class:`MultiModeMaxwellModel` for ``model="custom"``.
    initial_guesses, cost_function, bounds, num_initial_guesses, minimization_algorithm : optional
        Optimiser options passed to the fitted model.
    measurement_index : int, optional
        Measurement to fit in a set. Default 0.
    dataset_index : int, optional
        Dataset to fit within the measurement. Default 0.
    x_name, storage_name, loss_name : str, optional
        Names of the angular frequency, storage and loss modulus arrays.
    n_interp : int, optional
        Points of the common grid for master curves. Default 100.
    domain : {"oscillation", "relaxation"}, optional
        Fit :math:`G'(\omega), G''(\omega)` (default) or :math:`G(t)`.
    time_name, relaxation_name : str, optional
        Array names for ``domain="relaxation"``. Default ``"time"`` and
        ``"G_from_eta1"`` (the output of ``FrequencyToTime``).

    References
    ----------
    .. [Bonfanti2020] Bonfanti, A., Kaplan, J. L., Charras, G., Kabla, A.,
       *Fractional viscoelastic models for power-law materials*, Soft Matter
       16(26), 6002-6020 (2020). DOI: `10.1039/d0sm00354a
       <https://doi.org/10.1039/d0sm00354a>`_.
    .. [Song2023] Song, J., Holten-Andersen, N., McKinley, G. H.,
       *Non-Maxwellian viscoelastic stress relaxations in soft matter*, Soft
       Matter 19(41), 7885-7906 (2023). DOI: `10.1039/d3sm00736g
       <https://doi.org/10.1039/d3sm00736g>`_.
    .. [pyRheo] Miranda-Valdez, I. Y., Niinisto, J., Makinen, T., Lejon, J.,
       Koivisto, J., Alava, M. J., *pyRheo: an open-source Python package for
       complex rheology*, Digital Discovery 4(4), 1075-1082 (2025). DOI:
       `10.1039/d5dd00021a <https://doi.org/10.1039/d5dd00021a>`_.
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        self.fit_measurement = None

    @staticmethod
    def _xy(dataset, x_name, y_name):
        """Return ``(x, y)`` for two named arrays without mutating *dataset*."""
        ds = copy.deepcopy(dataset)
        ds.switch_coordinates(independent_name=x_name, dependent_name=y_name)
        return np.asarray(ds.x_values, dtype=float), np.asarray(ds.y_values, dtype=float)

    def _select_dataset(self, target):
        """Locate a single composite dataset from a variety of input objects."""
        if hasattr(target, "switch_coordinates"):  # a dataset already
            return target
        if hasattr(target, "measurements"):  # Experiment/MeasurementSet
            measurements = list(target.measurements)
            return measurements[self.measurement_index].datasets[self.dataset_index]
        if hasattr(target, "datasets"):  # a Measurement
            return target.datasets[self.dataset_index]
        raise TypeError(f"FitModel could not locate a dataset to fit in a {type(target).__name__}.")

    def _extract_from_measurements(self, target):
        """Return finite ``(omega, G', G'')`` sorted by frequency."""
        dataset = self._select_dataset(target)
        omega, g_prime = self._xy(dataset, self.x_name, self.storage_name)
        _, g_double_prime = self._xy(dataset, self.x_name, self.loss_name)
        mask = np.isfinite(omega) & np.isfinite(g_prime) & np.isfinite(g_double_prime)
        order = np.argsort(omega[mask])
        return omega[mask][order], g_prime[mask][order], g_double_prime[mask][order]

    def _extract_from_composite(self, composite):
        """Return ``(omega, G', G'')`` on a common log grid from a master curve."""
        for q in (self.storage_name, self.loss_name):
            if q not in composite.available_quantities:
                raise ValueError(f"FitModel requires '{q}' in the CompositeMasterCurve quantities.")
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

        interp_storage = interp1d(
            omega_s, gp_s, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        interp_loss = interp1d(
            omega_l, gpp_l, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
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

    def _build_model(self):
        """Return an unfitted model object for the current settings."""
        if self.domain == "relaxation":
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
        elif self.model == "custom" and self.model_name == "MultiModeMaxwell":
            return MultiModeMaxwellModel(
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
            model_class = _load_pyrheo_model("oscillation") if self.model is None else self.model
        return model_class(
            model=self.model_name,
            initial_guesses=self.initial_guesses,
            cost_function=self.cost_function,
            bounds=self.bounds,
            num_initial_guesses=self.num_initial_guesses,
            minimization_algorithm=self.minimization_algorithm,
        )

    @staticmethod
    def _plain(value):
        """Convert numpy values to plain Python for ``details``."""
        if isinstance(value, dict):
            return {str(k): FitModel._plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [FitModel._plain(v) for v in np.asarray(value).tolist()]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _parameter_details(self, model):
        """Return ``(parameters, goodness_of_fit)`` dicts for the fitted model."""
        if isinstance(model, MultiModeMaxwellModel):
            params = {
                "G_modes": model.G_modes,
                "tau_modes": model.tau_modes,
                "equilibrium_modulus": model.equilibrium_modulus_value,
            }
            gof = {
                "cost_function": model.cost_function,
                "cost": model.cost,
                "rss": model.rss,
                "mse": model.mse,
                "bic": model.bic,
            }
        else:
            raw = dict(model.get_parameters()) if hasattr(model, "get_parameters") else {}
            cost = raw.pop("Cost", getattr(model, "cost_", None))
            metric = raw.pop("Cost Metric", self.cost_function)
            params = raw
            gof = {"cost_function": metric, "cost": cost}
        return self._plain(params), self._plain(gof)

    def _to_measurement(self, model, x, y, y_fit, x_names, y_names, units, source):
        """Build the output measurement from measured and fitted arrays."""
        params, gof = self._parameter_details(model)
        residual = np.concatenate([np.ravel(a) - np.ravel(b) for a, b in zip(y, y_fit)])
        measured = np.concatenate([np.ravel(a) for a in y])
        ss_tot = float(np.sum((measured - measured.mean()) ** 2))
        gof["r_squared"] = float(1.0 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else None
        gof["uncertainties"] = "not estimated"

        name = getattr(model, "model", None) or self.model_name
        names = [x_names[0], *y_names]
        dataset = _composite.OneDimensionalCompositeDataset(
            data_arrays=[x, *y, *y_fit],
            data_array_names=names + [f"fitted {n}" for n in y_names],
            data_array_units=units,
            default_independent_name=x_names[0],
            default_dependent_name=y_names[0],
            source=source,
        )
        details = {
            "model": name,
            "parameters": params,
            "goodness_of_fit": gof,
            "method": f"{type(model).__name__} fit, cost {gof['cost_function']}",
            "domain": self.domain,
            "grid": {
                "variable": x_names[0],
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "points": int(np.size(x)),
            },
            "references": list(_REFERENCES),
            "software": f"hermes-rheo {__version__}",
        }
        return _measurement.Measurement(
            datasets=[dataset],
            conditions={"fit_model": name, "fitted": True},
            details=details,
        )

    def _fit(self, target):
        """Fit *target* and return the output ``Measurement``."""
        model = self._build_model()
        if self.domain == "relaxation":
            t, g_relax = self._extract_relaxation(target)
            model.fit(t, g_relax)
            g_fit = np.asarray(model.predict(t), dtype=float)
            out = self._to_measurement(
                model,
                t,
                [g_relax],
                [g_fit],
                ["time"],
                ["relaxation modulus"],
                ["s", "Pa", "Pa"],
                f"{self.model_name} fit of relaxation modulus",
            )
            omega = g_prime = g_double_prime = None
            self.t_common, self.G_relax = t, g_relax
        else:
            omega, g_prime, g_double_prime = self._extract_moduli(target)
            model.fit(omega, g_prime, g_double_prime)
            gp_fit, gpp_fit = (np.asarray(v, dtype=float) for v in model.predict(omega))
            out = self._to_measurement(
                model,
                omega,
                [g_prime, g_double_prime],
                [gp_fit, gpp_fit],
                ["angular frequency"],
                ["storage modulus", "loss modulus"],
                ["rad/s", "Pa", "Pa", "Pa", "Pa"],
                f"{self.model_name} fit of storage and loss modulus",
            )
            self.t_common = self.G_relax = None

        self.fitted_model = model
        self.fit_params = getattr(model, "parameters", None)
        if self.fit_params is None and hasattr(model, "get_parameters"):
            self.fit_params = model.get_parameters()
        self.omega_common, self.G_prime, self.G_double_prime = omega, g_prime, g_double_prime
        self.fit_measurement = out
        return out

    def _apply(self, target, **kwargs):
        return self._fit(target)

    def apply_to(self, target, make_copy=False, **kwargs):
        """Fit *target* and return a piblin ``Measurement``.

        *target* may also be a ``CompositeMasterCurve``. *make_copy* is
        ignored because the input is never modified.
        """
        return self._fit(target)

    def __call__(self, target, make_copy=False, **kwargs):
        """See :meth:`apply_to`."""
        return self._fit(target)

    @property
    def model_fit_params(self):
        """Fitted parameters of the model (alias of :attr:`fit_params`)."""
        return self.fit_params

    @property
    def model_fit_object(self):
        """Fitted model object (alias of :attr:`fitted_model`)."""
        return self.fitted_model

    @property
    def model_fit_plot(self):
        """Plot the fit; see :meth:`plot`."""
        return self.plot()

    def plot(self, *args, **kwargs):
        """Plot the fitted moduli against the data.

        Calls :func:`hermes_rheo.visualization.fit.plot_model_fit`, which
        accepts ``show_residuals``, ``ax``, ``figsize``, ``xlabel``,
        ``ylabel``, ``title``, ``xlim`` and ``ylim``. ``savefig``,
        ``format``, ``dpi`` and ``show`` are handled here. Returns
        ``(figure, axes)``.
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
        """Print the parameters of the fitted model."""
        if self.fitted_model is not None and hasattr(self.fitted_model, "print_parameters"):
            return self.fitted_model.print_parameters(*args, **kwargs)
        else:
            raise AttributeError("No fitted model available or print_parameters method not found.")

    def print_error(self, *args, **kwargs):
        """Print the error of the fitted model."""
        if self.fitted_model is not None and hasattr(self.fitted_model, "print_error"):
            return self.fitted_model.print_error(*args, **kwargs)
        else:
            raise AttributeError("No fitted model available or print_error method not found.")
