"""Models for the temperature dependence of time--temperature-superposition shift factors.

This module fits the horizontal shift factors :math:`a_T` produced by
:class:`~hermes_rheo.transforms.automated_mastercurve.AutomatedMasterCurve`
(exposed as ``master_curve.hparams[0]`` for the states
``master_curve.states``) to classical constitutive descriptions of their
temperature dependence:

* the **Williams--Landel--Ferry** (WLF) equation, and
* the **Arrhenius** equation.

The design mirrors :class:`~hermes_rheo.transforms.fit_model.FitModel`: the
pipeline-style :class:`ShiftFactorModel` wraps a lightweight model object
(:class:`WLFModel` or :class:`ArrheniusModel`) and can be applied directly to
a master-curve object.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize

# Universal gas constant (J mol^-1 K^-1).
_R = 8.314462618
_LN10 = np.log(10.0)
_CELSIUS_TO_KELVIN = 273.15


class WLFModel:
    r"""Williams--Landel--Ferry description of shift-factor temperature dependence.

    Fits :math:`\log_{10} a_T` as a function of temperature with

    .. math::

        \log_{10} a_T(T) = -\frac{C_1 (T - T_\mathrm{ref})}
                                  {C_2 + (T - T_\mathrm{ref})}

    where :math:`C_1` (dimensionless) and :math:`C_2` (in the temperature
    unit of the data, conventionally :math:`^\circ\mathrm{C}`) are the WLF
    constants and :math:`T_\mathrm{ref}` is the reference temperature at
    which :math:`a_T = 1`.

    Parameters
    ----------
    reference_temperature : float, optional
        Reference temperature.  If ``None`` (default) it is taken as the
        state whose shift factor is closest to unity.

    References
    ----------
    .. [WLF1955] Williams, M. L., Landel, R. F. and Ferry, J. D., *The
       temperature dependence of relaxation mechanisms in amorphous polymers
       and other glass-forming liquids*, J. Am. Chem. Soc. **77**(14),
       3701-3707 (1955).  DOI: `10.1021/ja01619a008
       <https://doi.org/10.1021/ja01619a008>`_.
    """

    model = "WLF"

    def __init__(self, reference_temperature=None):
        self.reference_temperature = reference_temperature
        self.C1 = None
        self.C2 = None
        self.parameters = None
        self._temperatures = None
        self._log_shift_factors = None
        self._log_shift_pred = None
        self.rss = None

    @staticmethod
    def _equation(temperature, c1, c2, reference_temperature):
        delta = temperature - reference_temperature
        return -c1 * delta / (c2 + delta)

    def fit(self, temperatures, log_shift_factors, reference_temperature=None):
        """Fit the WLF constants to shift-factor data.

        Parameters
        ----------
        temperatures : array_like
            Temperatures of each state (conventionally in Celsius).
        log_shift_factors : array_like
            Base-10 logarithm of the horizontal shift factors, ``log10(a_T)``.
        reference_temperature : float, optional
            Overrides the value supplied at construction.

        Returns
        -------
        WLFModel
            ``self``, with fitted attributes populated.
        """
        temperatures = np.asarray(temperatures, dtype=float)
        log_shift_factors = np.asarray(log_shift_factors, dtype=float)
        mask = np.isfinite(temperatures) & np.isfinite(log_shift_factors)
        temperatures, log_shift_factors = temperatures[mask], log_shift_factors[mask]

        if reference_temperature is not None:
            self.reference_temperature = reference_temperature
        if self.reference_temperature is None:
            self.reference_temperature = float(
                temperatures[np.argmin(np.abs(log_shift_factors))]
            )

        t_ref = self.reference_temperature

        # Constrain the WLF constants to be physically positive (C1 > 0, C2 > 0)
        # and use a multi-start bounded minimisation.  The WLF equation is
        # ill-conditioned (C1 and C2 trade off), so a single local solve can get
        # trapped at the degenerate C1 = 0 (flat) solution on near-Arrhenius
        # data; multiple starts reliably recover the true minimum.
        def _sse(params):
            c1, c2 = params
            return float(
                np.sum((log_shift_factors - self._equation(temperatures, c1, c2, t_ref)) ** 2)
            )

        guesses = [
            [17.44, 51.6],   # universal WLF constants
            [10.0, 98.0],
            [5.0, 100.0],
            [50.0, 200.0],
            [1.0, 50.0],
        ]
        best = None
        for guess in guesses:
            res = minimize(
                _sse, guess,
                bounds=[(0.0, None), (1e-6, None)],
                method="SLSQP",
            )
            if best is None or res.fun < best.fun:
                best = res

        self.C1, self.C2 = float(best.x[0]), float(best.x[1])
        self.parameters = {"C1": self.C1, "C2": self.C2, "reference_temperature": t_ref}
        self._temperatures = temperatures
        self._log_shift_factors = log_shift_factors
        self._log_shift_pred = self._equation(temperatures, self.C1, self.C2, t_ref)
        self.rss = float(np.sum((log_shift_factors - self._log_shift_pred) ** 2))
        return self

    def predict(self, temperatures):
        """Return ``log10(a_T)`` predicted by the fitted WLF equation."""
        temperatures = np.asarray(temperatures, dtype=float)
        return self._equation(temperatures, self.C1, self.C2, self.reference_temperature)

    def print_parameters(self):
        """Print the fitted WLF constants."""
        print(
            f"WLF: C1 = {self.C1:.4g}, C2 = {self.C2:.4g}, "
            f"Tref = {self.reference_temperature:.4g}"
        )


class ArrheniusModel:
    r"""Arrhenius description of shift-factor temperature dependence.

    Fits :math:`\log_{10} a_T` as a function of temperature with

    .. math::

        \log_{10} a_T(T) = \frac{E_a}{R \ln 10}
                           \left(\frac{1}{T} - \frac{1}{T_\mathrm{ref}}\right)

    with temperatures in kelvin, universal gas constant
    :math:`R = 8.314\ \mathrm{J\,mol^{-1}\,K^{-1}}`, and activation energy
    :math:`E_a` (in :math:`\mathrm{J\,mol^{-1}}`).

    Parameters
    ----------
    reference_temperature : float, optional
        Reference temperature (same unit as the input data).  If ``None``
        (default) it is taken as the state whose shift factor is closest to
        unity.
    temperature_unit : {"C", "K"}, optional
        Unit of the input temperatures.  Celsius (``"C"``, the default) is
        converted to kelvin internally.

    References
    ----------
    .. [Arrhenius1889] Arrhenius, S., *Über die Reaktionsgeschwindigkeit bei
       der Inversion von Rohrzucker durch Säuren*, Z. Phys. Chem. **4**,
       226-248 (1889).
    """

    model = "Arrhenius"

    def __init__(self, reference_temperature=None, temperature_unit="C"):
        if temperature_unit not in ("C", "K"):
            raise ValueError("temperature_unit must be 'C' or 'K'")
        self.reference_temperature = reference_temperature
        self.temperature_unit = temperature_unit
        self.activation_energy = None
        self.parameters = None
        self._temperatures = None
        self._log_shift_factors = None
        self._log_shift_pred = None
        self.rss = None

    def _to_kelvin(self, temperatures):
        if self.temperature_unit == "C":
            return temperatures + _CELSIUS_TO_KELVIN
        return temperatures

    def fit(self, temperatures, log_shift_factors, reference_temperature=None):
        """Fit the activation energy to shift-factor data.

        Parameters
        ----------
        temperatures : array_like
            Temperatures of each state, in the unit given by
            ``temperature_unit``.
        log_shift_factors : array_like
            Base-10 logarithm of the horizontal shift factors, ``log10(a_T)``.
        reference_temperature : float, optional
            Overrides the value supplied at construction.

        Returns
        -------
        ArrheniusModel
            ``self``, with fitted attributes populated.
        """
        temperatures = np.asarray(temperatures, dtype=float)
        log_shift_factors = np.asarray(log_shift_factors, dtype=float)
        mask = np.isfinite(temperatures) & np.isfinite(log_shift_factors)
        temperatures, log_shift_factors = temperatures[mask], log_shift_factors[mask]

        if reference_temperature is not None:
            self.reference_temperature = reference_temperature
        if self.reference_temperature is None:
            self.reference_temperature = float(
                temperatures[np.argmin(np.abs(log_shift_factors))]
            )

        t_kelvin = self._to_kelvin(temperatures)
        t_ref_kelvin = self._to_kelvin(np.asarray(self.reference_temperature, dtype=float))
        x = 1.0 / t_kelvin - 1.0 / t_ref_kelvin

        def _model(_x, ea):
            return (ea / (_R * _LN10)) * _x

        popt, _ = curve_fit(_model, x, log_shift_factors, p0=[1.0e5], maxfev=100000)
        self.activation_energy = float(popt[0])
        self.parameters = {
            "activation_energy": self.activation_energy,
            "reference_temperature": self.reference_temperature,
        }
        self._temperatures = temperatures
        self._log_shift_factors = log_shift_factors
        self._log_shift_pred = _model(x, self.activation_energy)
        self.rss = float(np.sum((log_shift_factors - self._log_shift_pred) ** 2))
        return self

    def predict(self, temperatures):
        """Return ``log10(a_T)`` predicted by the fitted Arrhenius equation."""
        temperatures = np.asarray(temperatures, dtype=float)
        t_kelvin = self._to_kelvin(temperatures)
        t_ref_kelvin = self._to_kelvin(np.asarray(self.reference_temperature, dtype=float))
        x = 1.0 / t_kelvin - 1.0 / t_ref_kelvin
        return (self.activation_energy / (_R * _LN10)) * x

    def print_parameters(self):
        """Print the fitted activation energy."""
        print(
            f"Arrhenius: Ea = {self.activation_energy:.4g} J/mol "
            f"({self.activation_energy / 1000.0:.4g} kJ/mol), "
            f"Tref = {self.reference_temperature:.4g}"
        )


_MODELS = {"WLF": WLFModel, "Arrhenius": ArrheniusModel}


class ShiftFactorModel:
    r"""Pipeline step that fits a temperature model to master-curve shift factors.

    ``ShiftFactorModel`` reads the horizontal shift factors :math:`a_T`
    (``master_curve.hparams[0]``) and their states
    (``master_curve.states``) from a master curve produced by
    :class:`~hermes_rheo.transforms.automated_mastercurve.AutomatedMasterCurve`,
    converts them to :math:`\log_{10} a_T`, and fits either the WLF or the
    Arrhenius equation.  It can also be applied to a pair of arrays
    ``(temperatures, shift_factors)``.

    Parameters
    ----------
    model : {"WLF", "Arrhenius"}, optional
        Which temperature model to fit.  Defaults to ``"WLF"``.
    reference_temperature : float, optional
        Reference temperature at which :math:`a_T = 1`.  If ``None``
        (default) the state closest to :math:`a_T = 1` is used.
    temperature_unit : {"C", "K"}, optional
        Unit of the state temperatures (used by the Arrhenius model).
        Defaults to ``"C"``.

    Examples
    --------
    >>> from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve
    >>> from hermes_rheo.transforms.shift_factor_models import ShiftFactorModel
    >>> mc = AutomatedMasterCurve(y='storage modulus').apply_to(experiment)  # doctest: +SKIP
    >>> wlf = ShiftFactorModel(model='WLF')(mc)  # doctest: +SKIP
    >>> wlf.print_parameters()  # doctest: +SKIP
    """

    def __init__(self, model="WLF", reference_temperature=None, temperature_unit="C"):
        if model not in _MODELS:
            raise ValueError(f"model must be one of {sorted(_MODELS)}")
        self.model = model
        self.reference_temperature = reference_temperature
        self.temperature_unit = temperature_unit
        self.fitted_model = None
        self.parameters = None
        self.temperatures = None
        self.log_shift_factors = None

    @staticmethod
    def _extract(target):
        """Return ``(temperatures, log10(a_T))`` from a master curve or arrays."""
        if hasattr(target, "hparams") and hasattr(target, "states"):
            states = np.asarray(target.states, dtype=float)
            shift_factors = np.asarray(target.hparams[0], dtype=float)
        else:
            states, shift_factors = target
            states = np.asarray(states, dtype=float)
            shift_factors = np.asarray(shift_factors, dtype=float)
        with np.errstate(divide="ignore"):
            log_shift = np.log10(shift_factors)
        return states, log_shift

    def __call__(self, target):
        temperatures, log_shift_factors = self._extract(target)
        mask = np.isfinite(temperatures) & np.isfinite(log_shift_factors)
        temperatures, log_shift_factors = temperatures[mask], log_shift_factors[mask]

        model_cls = _MODELS[self.model]
        if self.model == "Arrhenius":
            model = model_cls(
                reference_temperature=self.reference_temperature,
                temperature_unit=self.temperature_unit,
            )
        else:
            model = model_cls(reference_temperature=self.reference_temperature)
        model.fit(temperatures, log_shift_factors)

        self.fitted_model = model
        self.parameters = model.parameters
        self.reference_temperature = model.reference_temperature
        self.temperatures = temperatures
        self.log_shift_factors = log_shift_factors
        return self

    def apply_to(self, target, make_copy=False, **kwargs):
        """Apply the model to *target* (kept for pipeline symmetry with FitModel)."""
        return self.__call__(target)

    def predict(self, temperatures):
        """Return ``log10(a_T)`` predicted by the fitted model."""
        if self.fitted_model is None:
            raise RuntimeError("ShiftFactorModel has not been fitted yet.")
        return self.fitted_model.predict(temperatures)

    def plot(self, *args, **kwargs):
        """Plot ``log10(a_T)`` versus temperature with the fitted curve.

        Delegates to
        :func:`hermes_rheo.visualization.fit.plot_shift_factors` and
        returns ``(figure, axes)``.  Pass ``show=False`` to suppress the
        implicit ``plt.show``.
        """
        from hermes_rheo.visualization.fit import plot_shift_factors

        show = kwargs.pop("show", True)
        fig, ax = plot_shift_factors(self, show=False, **kwargs)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig, ax

    def print_parameters(self):
        """Print the fitted model parameters."""
        if self.fitted_model is None:
            raise RuntimeError("ShiftFactorModel has not been fitted yet.")
        self.fitted_model.print_parameters()
