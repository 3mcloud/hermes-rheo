"""Classical linear viscoelastic models.

* :class:`MaxwellModel`: spring and dashpot in series.
* :class:`MultiModeMaxwellModel`: parallel Maxwell modes (Prony series),
  optionally with an equilibrium modulus.
* :class:`KelvinVoigtModel`: spring and dashpot in parallel.
* :class:`JeffreysModel`: a Maxwell mode in parallel with a dashpot.

References
----------
.. [Ferry1980] Ferry, J. D., *Viscoelastic Properties of Polymers*, 3rd ed.,
   Wiley (1980). ISBN 978-0471048947.
.. [Bird1987] Bird, R. B., Armstrong, R. C. and Hassager, O., *Dynamics of
   Polymeric Liquids, Vol. 1*, 2nd ed., Wiley (1987). ISBN 978-0471802457.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hermes_rheo.lve_models.base import LVEModel, positive

__all__ = [
    "MaxwellModel",
    "MultiModeMaxwellModel",
    "KelvinVoigtModel",
    "JeffreysModel",
]


class MaxwellModel(LVEModel):
    r"""Maxwell fluid: spring :math:`G` and dashpot :math:`\eta = G\tau` in series.

    .. math::

       G(t) = G e^{-t/\tau}, \qquad
       J(t) = \frac{1}{G}\left(1 + \frac{t}{\tau}\right), \qquad
       G^*(s) = \frac{G \tau s}{1 + \tau s}

    Parameters
    ----------
    G : float
        Modulus in Pa.
    tau : float
        Relaxation time in s.

    Examples
    --------
    >>> m = MaxwellModel(G=1e5, tau=1.0)
    >>> gp, gpp = m.saos(np.array([1.0]))
    >>> bool(np.isclose(gp[0], 5e4) and np.isclose(gpp[0], 5e4))
    True
    """

    name = "Maxwell"
    references = ("Ferry 1980, ch. 3",)

    def __init__(self, G: float, tau: float) -> None:
        self.G = positive("G", G)
        self.tau = positive("tau", tau)
        self.parameters = {"G": self.G, "tau": self.tau}

    def complex_modulus(self, s):
        return self.G * self.tau * s / (1.0 + self.tau * s)

    def stress_relaxation(self, t):
        return self.G * np.exp(-np.asarray(t, dtype=float) / self.tau)

    def creep(self, t):
        return (1.0 + np.asarray(t, dtype=float) / self.tau) / self.G


class MultiModeMaxwellModel(LVEModel):
    r"""Parallel Maxwell modes with an optional equilibrium modulus.

    .. math::

       G(t) = G_\infty + \sum_i g_i e^{-t/\tau_i}, \qquad
       G^*(s) = G_\infty + \sum_i \frac{g_i \tau_i s}{1 + \tau_i s}

    The creep compliance has no simple closed form and is computed by
    numerical Laplace inversion (:math:`t > 0`; at :math:`t = 0` it is
    :math:`J(0) = 1/G(0)`).

    Parameters
    ----------
    g : sequence of float
        Mode strengths in Pa.
    tau : sequence of float
        Relaxation times in s, same length as *g*.
    G_inf : float, optional
        Equilibrium modulus in Pa. ``0`` (default) gives a fluid.

    Examples
    --------
    >>> m = MultiModeMaxwellModel(g=[1e5, 1e4], tau=[0.1, 1.0])
    >>> bool(np.isclose(m.stress_relaxation(np.array([0.0]))[0], 1.1e5))
    True
    """

    name = "MultiModeMaxwell"
    references = ("Ferry 1980, ch. 3", "Bird et al. 1987, ch. 5")

    def __init__(self, g: Sequence[float], tau: Sequence[float], G_inf: float = 0.0) -> None:
        g_arr = np.asarray(g, dtype=float).ravel()
        tau_arr = np.asarray(tau, dtype=float).ravel()
        if g_arr.size == 0 or g_arr.size != tau_arr.size:
            raise ValueError("g and tau must be non-empty and have the same length")
        for label, arr in (("g", g_arr), ("tau", tau_arr)):
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError(f"{label} must contain only finite positive values")
        G_inf = float(G_inf)
        if not np.isfinite(G_inf) or G_inf < 0.0:
            raise ValueError(f"G_inf must be finite and non-negative, got {G_inf!r}")
        order = np.argsort(tau_arr)
        self.g, self.tau, self.G_inf = g_arr[order], tau_arr[order], G_inf
        self.parameters = {"g": self.g.tolist(), "tau": self.tau.tolist(), "G_inf": G_inf}

    def complex_modulus(self, s):
        s = np.asarray(s)
        ts = s[..., None] * self.tau
        return self.G_inf + (self.g * ts / (1.0 + ts)).sum(axis=-1)

    def stress_relaxation(self, t):
        t = np.asarray(t, dtype=float)
        return self.G_inf + (self.g * np.exp(-t[..., None] / self.tau)).sum(axis=-1)

    def creep(self, t):
        t = np.asarray(t, dtype=float)
        positive_t = t > 0.0
        out = np.full(t.shape, 1.0 / (self.G_inf + self.g.sum()))
        if positive_t.any():
            out[positive_t] = super().creep(t[positive_t])
        return out


class KelvinVoigtModel(LVEModel):
    r"""Kelvin-Voigt solid: spring :math:`G` and dashpot :math:`\eta` in parallel.

    .. math::

       J(t) = \frac{1}{G}\left(1 - e^{-t/\tau_r}\right), \quad
       \tau_r = \frac{\eta}{G}, \qquad
       G^*(s) = G + \eta s

    The relaxation modulus is :math:`G + \eta\,\delta(t)`. For :math:`t > 0`
    :meth:`stress_relaxation` returns the constant :math:`G`.

    Parameters
    ----------
    G : float
        Modulus in Pa.
    eta : float
        Viscosity in Pa s.
    """

    name = "KelvinVoigt"
    references = ("Ferry 1980, ch. 1",)

    def __init__(self, G: float, eta: float) -> None:
        self.G = positive("G", G)
        self.eta = positive("eta", eta)
        self.tau_r = self.eta / self.G
        self.parameters = {"G": self.G, "eta": self.eta}

    def complex_modulus(self, s):
        return self.G + self.eta * s

    def stress_relaxation(self, t):
        return np.full(np.shape(t), self.G, dtype=float)

    def creep(self, t):
        return -np.expm1(-np.asarray(t, dtype=float) / self.tau_r) / self.G


class JeffreysModel(LVEModel):
    r"""Jeffreys model: a Maxwell mode in parallel with a dashpot.

    With :math:`\eta_p = G\tau` and :math:`\eta = \eta_p + \eta_s`:

    .. math::

       G^*(s) = \eta_s s + \frac{G\tau s}{1 + \tau s}, \qquad
       J(t) = \frac{t}{\eta}
              + \frac{\tau\,\eta_p}{\eta^2}\left(1 - e^{-t/\tau_r}\right), \quad
       \tau_r = \frac{\eta_s \tau}{\eta}

    The relaxation modulus is :math:`G e^{-t/\tau} + \eta_s\,\delta(t)`. For
    :math:`t > 0` :meth:`stress_relaxation` returns the first term.

    Parameters
    ----------
    G : float
        Modulus of the Maxwell mode in Pa.
    tau : float
        Relaxation time of the Maxwell mode in s.
    eta_s : float
        Solvent viscosity in Pa s.
    """

    name = "Jeffreys"
    references = ("Jeffreys 1929", "Bird et al. 1987, ch. 5")

    def __init__(self, G: float, tau: float, eta_s: float) -> None:
        self.G = positive("G", G)
        self.tau = positive("tau", tau)
        self.eta_s = positive("eta_s", eta_s)
        self.eta_p = self.G * self.tau
        self.parameters = {"G": self.G, "tau": self.tau, "eta_s": self.eta_s}

    def complex_modulus(self, s):
        return self.eta_s * s + self.G * self.tau * s / (1.0 + self.tau * s)

    def stress_relaxation(self, t):
        return self.G * np.exp(-np.asarray(t, dtype=float) / self.tau)

    def creep(self, t):
        t = np.asarray(t, dtype=float)
        eta = self.eta_p + self.eta_s
        tau_r = self.eta_s * self.tau / eta
        return t / eta + (self.tau * self.eta_p / eta**2) * -np.expm1(-t / tau_r)
