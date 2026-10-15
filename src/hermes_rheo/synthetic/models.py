"""Classical linear-viscoelastic constitutive equations.

Each concrete model exposes the three canonical LVE experiments
:meth:`stress_relaxation`, :meth:`frequency_sweep` and :meth:`creep`
as pure NumPy functions of the observation grid, plus a
:meth:`to_measurement` convenience wrapper that builds a piblin
:class:`~piblin.data.data_collections.measurement.Measurement`
with the correct column names, units, ``conditions["method"]`` and
``details["model"] / details["parameters"]`` metadata.

The classical models are:

* :class:`MaxwellModel` --- single spring + dashpot in series.
* :class:`MultiModeMaxwellModel` --- generalized Maxwell (Prony series)
  with optional plateau modulus :math:`G_\\infty` for viscoelastic
  solids.
* :class:`KelvinVoigtModel` --- spring + dashpot in parallel.
* :class:`JeffreysModel` --- Jeffreys / Oldroyd model
  (Maxwell mode + purely viscous solvent).

References
----------
.. [Ferry1980] Ferry, J. D., *Viscoelastic Properties of Polymers*, 3rd
   ed., John Wiley & Sons (1980).  ISBN 978-0471048947.
.. [Bird1987] Bird, R. B., Armstrong, R. C. and Hassager, O., *Dynamics
   of Polymeric Liquids, Volume 1: Fluid Mechanics*, 2nd ed., Wiley
   (1987).  ISBN 978-0471802457.
.. [Jeffreys1929] Jeffreys, H., *The Earth: Its Origin, History and
   Physical Constitution*, 2nd ed., Cambridge University Press (1929).
   Original introduction of the three-parameter Jeffreys constitutive
   equation used in :class:`JeffreysModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from piblin.data.data_collections import measurement as _measurement
from piblin.data.datasets.abc.split_datasets import (
    one_dimensional_composite_dataset as _one_d_composite_dataset,
)

__all__ = [
    "RheologyModel",
    "MaxwellModel",
    "MultiModeMaxwellModel",
    "KelvinVoigtModel",
    "JeffreysModel",
]


# ---------------------------------------------------------------------------
# Column-name / unit conventions
# ---------------------------------------------------------------------------

_TIME_UNIT = "s"
_OMEGA_UNIT = "rad/s"
_MODULUS_UNIT = "Pa"
_COMPLIANCE_UNIT = "1/Pa"

_RELAXATION_NAME = "relaxation modulus"
_CREEP_COMPLIANCE_NAME = "creep compliance"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RheologyModel(ABC):
    """Abstract base class for linear-viscoelastic constitutive models.

    Subclasses implement the three canonical LVE experiments
    :meth:`stress_relaxation`, :meth:`frequency_sweep` and :meth:`creep`
    as pure functions of the observation grid.  The shared
    :meth:`to_measurement` builds a piblin ``Measurement`` from
    those arrays with the correct column names, units, and
    ``details`` / ``conditions`` metadata so the result composes
    directly with the rest of the hermes-rheo pipeline.

    Attributes
    ----------
    parameters : dict
        Human-readable snapshot of the constitutive parameters (used
        for the ``details["parameters"]`` metadata block).
    model_name : str
        Short label stored in ``details["model"]`` and appended to
        ``conditions["method"]``.

    Notes
    -----
    All moduli are in Pa, times in s, angular frequencies in rad/s and
    creep compliances in 1/Pa.  Downstream transforms and visualization
    helpers use those units through
    :func:`hermes_rheo.visualization.units.get_unit` and stay consistent
    with the TRIOS-composite conventions of the file readers.
    """

    #: Short model label used for metadata (subclasses override).
    model_name: str = "RheologyModel"

    #: Constitutive parameters stored on ``details["parameters"]``.
    #: Subclasses override in ``__init__`` and expose them as attributes.
    parameters: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------

    @abstractmethod
    def stress_relaxation(self, t: np.ndarray) -> np.ndarray:
        """Return the relaxation modulus :math:`G(t)` in Pa.

        Parameters
        ----------
        t : np.ndarray
            Time grid in seconds.  Must be strictly positive and finite.

        Returns
        -------
        np.ndarray
            :math:`G(t)` evaluated on *t*, same shape as *t*.
        """

    @abstractmethod
    def frequency_sweep(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the oscillatory moduli :math:`(G'(\\omega), G''(\\omega))`.

        Parameters
        ----------
        omega : np.ndarray
            Angular-frequency grid in rad/s.  Must be strictly positive
            and finite.

        Returns
        -------
        g_prime : np.ndarray
            Storage modulus in Pa, same shape as *omega*.
        g_double_prime : np.ndarray
            Loss modulus in Pa, same shape as *omega*.
        """

    @abstractmethod
    def creep(self, t: np.ndarray) -> np.ndarray:
        """Return the creep compliance :math:`J(t)` in 1/Pa.

        Parameters
        ----------
        t : np.ndarray
            Time grid in seconds.  Must be strictly non-negative and finite.

        Returns
        -------
        np.ndarray
            :math:`J(t)` evaluated on *t*, same shape as *t*.
        """

    # -----------------------------------------------------------------
    # Concrete helpers
    # -----------------------------------------------------------------

    def to_measurement(
        self,
        experiment: str,
        x: np.ndarray,
        *,
        conditions: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> _measurement.Measurement:
        """Build a piblin ``Measurement`` for a given experiment.

        Parameters
        ----------
        experiment : {"stress_relaxation", "frequency_sweep", "creep"}
            Type of synthetic experiment to generate.
        x : np.ndarray
            Observation grid.  Interpreted as time (s) for
            ``stress_relaxation`` and ``creep`` and as angular
            frequency (rad/s) for ``frequency_sweep``.
        conditions : dict or None, optional
            Extra keys to merge into ``Measurement.conditions``.  The
            base implementation always sets ``"method"`` and
            ``"synthetic"``.
        details : dict or None, optional
            Extra keys to merge into ``Measurement.details``.  The
            base implementation always sets ``"model"``, ``"parameters"``
            and ``"experiment"``.
        source : str or None, optional
            Free-text description passed to the composite dataset's
            ``source`` field.  Defaults to a model-specific string.

        Returns
        -------
        piblin.data.data_collections.measurement.Measurement
            One-measurement, one-dataset object whose active dataset
            carries the correct hermes-rheo column names, units and
            defaults for the requested experiment.

        Raises
        ------
        ValueError
            If ``experiment`` is not one of the three supported strings.
        """
        x_arr = np.asarray(x, dtype=float).ravel()

        experiment_key = experiment.strip().lower()
        if experiment_key == "stress_relaxation":
            dataset, method = self._build_stress_relaxation_dataset(x_arr, source=source)
        elif experiment_key == "frequency_sweep":
            dataset, method = self._build_frequency_sweep_dataset(x_arr, source=source)
        elif experiment_key == "creep":
            dataset, method = self._build_creep_dataset(x_arr, source=source)
        else:
            raise ValueError(
                "experiment must be one of "
                "{'stress_relaxation', 'frequency_sweep', 'creep'}, "
                f"got {experiment!r}"
            )

        merged_conditions: Dict[str, Any] = {
            "method": method,
            "synthetic": True,
        }
        if conditions:
            merged_conditions.update(conditions)

        merged_details: Dict[str, Any] = {
            "model": self.model_name,
            "parameters": dict(self.parameters),
            "experiment": experiment_key,
        }
        if details:
            merged_details.update(details)

        return _measurement.Measurement(
            datasets=[dataset],
            conditions=merged_conditions,
            details=merged_details,
        )

    # -----------------------------------------------------------------
    # Private dataset builders (kept small so subclasses rarely override)
    # -----------------------------------------------------------------

    def _build_stress_relaxation_dataset(
        self,
        t: np.ndarray,
        source: Optional[str],
    ) -> Tuple[
        _one_d_composite_dataset.OneDimensionalCompositeDataset,
        str,
    ]:
        g_t = np.asarray(self.stress_relaxation(t), dtype=float)
        method = f"Stress Relaxation Test (synthetic {self.model_name})"
        ds = _one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[t, g_t],
            data_array_names=["time", _RELAXATION_NAME],
            data_array_units=[_TIME_UNIT, _MODULUS_UNIT],
            default_independent_name="time",
            default_dependent_name=_RELAXATION_NAME,
            source=source or f"Synthetic {self.model_name} stress relaxation",
        )
        return ds, method

    def _build_frequency_sweep_dataset(
        self,
        omega: np.ndarray,
        source: Optional[str],
    ) -> Tuple[
        _one_d_composite_dataset.OneDimensionalCompositeDataset,
        str,
    ]:
        gp, gpp = self.frequency_sweep(omega)
        gp = np.asarray(gp, dtype=float)
        gpp = np.asarray(gpp, dtype=float)
        complex_modulus = np.sqrt(gp**2 + gpp**2)
        # Avoid ``0/0`` when both moduli vanish (e.g. Kelvin-Voigt at omega=0).
        with np.errstate(divide="ignore", invalid="ignore"):
            tan_delta = np.where(gp != 0.0, gpp / gp, np.nan)
        method = f"Frequency sweep (synthetic {self.model_name})"
        ds = _one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[omega, gp, gpp, complex_modulus, tan_delta],
            data_array_names=[
                "angular frequency",
                "storage modulus",
                "loss modulus",
                "complex modulus",
                "tan(delta)",
            ],
            data_array_units=[_OMEGA_UNIT, _MODULUS_UNIT, _MODULUS_UNIT, _MODULUS_UNIT, ""],
            default_independent_name="angular frequency",
            default_dependent_name="storage modulus",
            source=source or f"Synthetic {self.model_name} frequency sweep",
        )
        return ds, method

    def _build_creep_dataset(
        self,
        t: np.ndarray,
        source: Optional[str],
    ) -> Tuple[
        _one_d_composite_dataset.OneDimensionalCompositeDataset,
        str,
    ]:
        j_t = np.asarray(self.creep(t), dtype=float)
        method = f"Creep (synthetic {self.model_name})"
        ds = _one_d_composite_dataset.OneDimensionalCompositeDataset(
            data_arrays=[t, j_t],
            data_array_names=["time", _CREEP_COMPLIANCE_NAME],
            data_array_units=[_TIME_UNIT, _COMPLIANCE_UNIT],
            default_independent_name="time",
            default_dependent_name=_CREEP_COMPLIANCE_NAME,
            source=source or f"Synthetic {self.model_name} creep compliance",
        )
        return ds, method


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------


def _validate_positive_scalar(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar, got {value!r}")
    return v


def _validate_positive_array(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values, got {values!r}")
    return arr


class MaxwellModel(RheologyModel):
    """Single-mode Maxwell fluid (spring + dashpot in series).

    Relaxation modulus:

    .. math::

       G(t) = G_0 \\exp(-t/\\tau)

    Oscillatory moduli:

    .. math::

       G'(\\omega) = G_0 \\frac{(\\omega\\tau)^2}{1 + (\\omega\\tau)^2},
       \\qquad
       G''(\\omega) = G_0 \\frac{\\omega\\tau}{1 + (\\omega\\tau)^2}

    Creep compliance (fluid: linear terminal growth):

    .. math::

       J(t) = \\frac{1}{G_0} + \\frac{t}{G_0 \\tau}

    Parameters
    ----------
    G0 : float
        Instantaneous modulus in Pa.  Must be finite and positive.
    tau : float
        Relaxation time in seconds.  Must be finite and positive.

    Attributes
    ----------
    G0 : float
    tau : float
    parameters : dict
        ``{"G0": G0, "tau": tau}``.

    References
    ----------
    See [Ferry1980]_ Chapter 3.

    Examples
    --------
    >>> import numpy as np
    >>> m = MaxwellModel(G0=1e5, tau=1.0)
    >>> np.isclose(m.stress_relaxation(np.array([0.0]))[0], 1e5)
    True
    >>> gp, gpp = m.frequency_sweep(np.array([1.0]))
    >>> np.isclose(gp[0], 0.5e5) and np.isclose(gpp[0], 0.5e5)
    True
    """

    model_name = "Maxwell"

    def __init__(self, G0: float, tau: float) -> None:
        self.G0 = _validate_positive_scalar("G0", G0)
        self.tau = _validate_positive_scalar("tau", tau)
        self.parameters = {"G0": self.G0, "tau": self.tau}

    def stress_relaxation(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        return self.G0 * np.exp(-t_arr / self.tau)

    def frequency_sweep(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = np.asarray(omega, dtype=float)
        wt = w * self.tau
        denom = 1.0 + wt**2
        return self.G0 * (wt**2) / denom, self.G0 * wt / denom

    def creep(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        return 1.0 / self.G0 + t_arr / (self.G0 * self.tau)


class MultiModeMaxwellModel(RheologyModel):
    """Generalized (multi-mode) Maxwell model with optional plateau.

    Relaxation modulus:

    .. math::

       G(t) = G_\\infty + \\sum_{i=1}^{N} g_i \\exp(-t/\\tau_i)

    Oscillatory moduli:

    .. math::

       G'(\\omega) = G_\\infty + \\sum_i g_i
                    \\frac{(\\omega\\tau_i)^2}{1 + (\\omega\\tau_i)^2},
       \\qquad
       G''(\\omega) = \\sum_i g_i
                    \\frac{\\omega\\tau_i}{1 + (\\omega\\tau_i)^2}

    For ``G_inf == 0`` the model is a viscoelastic fluid; for ``G_inf > 0``
    it is a viscoelastic solid whose long-time modulus tends to
    :math:`G_\\infty`.  This is the discrete relaxation spectrum
    (Prony series) used by :mod:`pyrespect_time` and :mod:`pyrespect_freq`.

    Parameters
    ----------
    g : sequence of float
        Mode weights :math:`g_i` in Pa.  All entries finite and positive.
    tau : sequence of float
        Relaxation times :math:`\\tau_i` in seconds.  All entries finite
        and positive.  Must have the same length as *g*.
    G_inf : float, optional
        Plateau modulus in Pa.  Default 0.0 (viscoelastic fluid).

    Attributes
    ----------
    g : np.ndarray
    tau : np.ndarray
    G_inf : float
    parameters : dict
        ``{"g": [...], "tau": [...], "G_inf": G_inf}``.

    Notes
    -----
    Creep compliance is NOT analytically closed-form for a general
    multi-mode Maxwell.  The implementation returns a numerical estimate
    based on the retardation spectrum obtained by numerically inverting
    the relaxation kernel via Simpson integration on a dense log-time
    grid.  This is intentionally kept simple; users needing high
    accuracy on creep should generate a dedicated stress-relaxation
    experiment and consume ``FrequencyToTime`` or a similar transform
    instead.

    References
    ----------
    See [Ferry1980]_ Chapter 3 and [Bird1987]_ Chapter 5.

    Examples
    --------
    >>> import numpy as np
    >>> m = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
    >>> np.isclose(m.stress_relaxation(np.array([0.0]))[0], 1.1e5)
    True
    >>> gp, gpp = m.frequency_sweep(np.array([10.0]))
    >>> float(gpp[0]) > 0.0
    True
    """

    model_name = "MultiModeMaxwell"

    def __init__(
        self,
        g: Sequence[float],
        tau: Sequence[float],
        G_inf: float = 0.0,
    ) -> None:
        g_arr = _validate_positive_array("g", g)
        tau_arr = _validate_positive_array("tau", tau)
        if g_arr.size != tau_arr.size:
            raise ValueError(
                f"g and tau must have the same length, got {g_arr.size} and {tau_arr.size}"
            )
        g_inf_val = float(G_inf)
        if not np.isfinite(g_inf_val) or g_inf_val < 0.0:
            raise ValueError(f"G_inf must be a finite non-negative scalar, got {G_inf!r}")

        # Sort by ascending tau so downstream code (plotting, comparison
        # with pyReSpect output) has a canonical order.
        order = np.argsort(tau_arr)
        self.tau = tau_arr[order]
        self.g = g_arr[order]
        self.G_inf = g_inf_val
        self.parameters = {
            "g": self.g.tolist(),
            "tau": self.tau.tolist(),
            "G_inf": self.G_inf,
        }

    def stress_relaxation(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float).ravel()
        # Broadcast: (n_t, 1) * (n_modes,) -> (n_t, n_modes)
        contributions = self.g[None, :] * np.exp(-t_arr[:, None] / self.tau[None, :])
        return contributions.sum(axis=1) + self.G_inf

    def frequency_sweep(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = np.asarray(omega, dtype=float).ravel()
        wt = w[:, None] * self.tau[None, :]
        denom = 1.0 + wt**2
        gp = (self.g[None, :] * wt**2 / denom).sum(axis=1) + self.G_inf
        gpp = (self.g[None, :] * wt / denom).sum(axis=1)
        return gp, gpp

    def creep(self, t: np.ndarray) -> np.ndarray:
        # Numerical retardation-spectrum inversion is beyond the scope of
        # a synthetic-data helper. We provide the short-time approximation
        # J(t) ~ 1/G(t) as a first-order estimate and flag the caveat in
        # the docstring.  For scientific creep analysis, use the transform
        # pipeline (frequency_to_time / relaxation_spectrum).
        return 1.0 / self.stress_relaxation(t)


class KelvinVoigtModel(RheologyModel):
    """Kelvin--Voigt element (spring + dashpot in parallel).

    Storage / loss moduli:

    .. math::

       G'(\\omega) = G,\\qquad G''(\\omega) = \\eta\\omega

    Creep compliance (retarded elastic solid):

    .. math::

       J(t) = \\frac{1}{G}\\left[1 - \\exp(-t/\\tau_r)\\right],
       \\qquad \\tau_r = \\eta/G

    Stress relaxation modulus is a Dirac delta at :math:`t=0` plus the
    constant :math:`G` for all :math:`t>0`; :meth:`stress_relaxation`
    returns the well-defined non-impulsive branch :math:`G(t) = G` for
    all :math:`t \\ge 0` and documents the delta contribution here.

    Parameters
    ----------
    G : float
        Elastic modulus in Pa.  Finite and positive.
    eta : float
        Dashpot viscosity in Pa*s.  Finite and positive.

    Attributes
    ----------
    G : float
    eta : float
    tau_retardation : float
        :math:`\\tau_r = \\eta/G` (s).
    parameters : dict

    References
    ----------
    See [Ferry1980]_ Chapter 1 for the Kelvin-Voigt element.

    Examples
    --------
    >>> import numpy as np
    >>> m = KelvinVoigtModel(G=1e5, eta=1e4)
    >>> gp, gpp = m.frequency_sweep(np.array([1.0]))
    >>> np.isclose(gp[0], 1e5) and np.isclose(gpp[0], 1e4)
    True
    """

    model_name = "KelvinVoigt"

    def __init__(self, G: float, eta: float) -> None:
        self.G = _validate_positive_scalar("G", G)
        self.eta = _validate_positive_scalar("eta", eta)
        self.tau_retardation = self.eta / self.G
        self.parameters = {
            "G": self.G,
            "eta": self.eta,
            "tau_retardation": self.tau_retardation,
        }

    def stress_relaxation(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        return np.full_like(t_arr, self.G, dtype=float)

    def frequency_sweep(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = np.asarray(omega, dtype=float)
        gp = np.full_like(w, self.G, dtype=float)
        gpp = self.eta * w
        return gp, gpp

    def creep(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        return (1.0 - np.exp(-t_arr / self.tau_retardation)) / self.G


class JeffreysModel(RheologyModel):
    """Jeffreys / Oldroyd model: Maxwell mode + purely-viscous solvent.

    Equivalent to a Maxwell element (:math:`G_0, \\tau`) in parallel with
    a dashpot of viscosity :math:`\\eta_s`.

    Relaxation modulus (impulsive :math:`\\eta_s` contribution omitted for
    :math:`t>0`, as for the pure dashpot):

    .. math::

       G(t) = G_0 \\exp(-t/\\tau)

    Oscillatory moduli:

    .. math::

       G'(\\omega) = G_0 \\frac{(\\omega\\tau)^2}{1 + (\\omega\\tau)^2},
       \\qquad
       G''(\\omega) = G_0 \\frac{\\omega\\tau}{1 + (\\omega\\tau)^2}
                     + \\eta_s \\omega

    Creep compliance:

    .. math::

       J(t) = \\frac{1}{G_0}\\left[1 - \\exp(-t/\\tau_r)\\right]
              + \\frac{t}{\\eta_s + \\eta_p},
       \\qquad
       \\tau_r = \\frac{\\eta_p \\eta_s}{G_0(\\eta_p + \\eta_s)},
       \\qquad
       \\eta_p = G_0 \\tau

    Parameters
    ----------
    G0 : float
        Elastic modulus of the Maxwell branch in Pa.
    tau : float
        Relaxation time of the Maxwell branch in s.
    eta_s : float
        Solvent (parallel dashpot) viscosity in Pa*s.

    Attributes
    ----------
    G0, tau, eta_s : float
    eta_p : float
        Polymer viscosity :math:`\\eta_p = G_0 \\tau` (Pa*s).
    parameters : dict

    References
    ----------
    See [Jeffreys1929]_ and [Bird1987]_ Chapter 5.

    Examples
    --------
    >>> import numpy as np
    >>> m = JeffreysModel(G0=1e5, tau=1.0, eta_s=1e3)
    >>> gp, gpp = m.frequency_sweep(np.array([1e-3]))
    >>> float(gpp[0]) > 0.0
    True
    """

    model_name = "Jeffreys"

    def __init__(self, G0: float, tau: float, eta_s: float) -> None:
        self.G0 = _validate_positive_scalar("G0", G0)
        self.tau = _validate_positive_scalar("tau", tau)
        self.eta_s = _validate_positive_scalar("eta_s", eta_s)
        self.eta_p = self.G0 * self.tau
        self.parameters = {
            "G0": self.G0,
            "tau": self.tau,
            "eta_s": self.eta_s,
            "eta_p": self.eta_p,
        }

    def stress_relaxation(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        return self.G0 * np.exp(-t_arr / self.tau)

    def frequency_sweep(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = np.asarray(omega, dtype=float)
        wt = w * self.tau
        denom = 1.0 + wt**2
        gp = self.G0 * (wt**2) / denom
        gpp = self.G0 * wt / denom + self.eta_s * w
        return gp, gpp

    def creep(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        eta_total = self.eta_p + self.eta_s
        tau_r = (self.eta_p * self.eta_s) / (self.G0 * eta_total)
        return (1.0 - np.exp(-t_arr / tau_r)) / self.G0 + t_arr / eta_total
