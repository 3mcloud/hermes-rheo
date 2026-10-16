"""Fractional linear viscoelastic models.

Built from the springpot (Scott Blair element), whose complex modulus is
:math:`G^*(s) = V s^\\alpha` with :math:`0 < \\alpha < 1`. Elements in series
add compliances (:math:`1/G^*`); elements in parallel add moduli.

Parameter names follow Bonfanti et al. (2020) and pyRheo, so values can be
exchanged directly. Quasi-moduli :math:`V, G, K` have units Pa s^order.

* :class:`SpringPotModel`
* :class:`FractionalMaxwellGelModel`: spring and springpot in series.
* :class:`FractionalMaxwellLiquidModel`: springpot and dashpot in series.
* :class:`FractionalMaxwellModel`: two springpots in series.
* :class:`FractionalKelvinVoigtModel`: two springpots in parallel.
* :class:`FractionalZenerModel`: a springpot in parallel with a fractional
  Maxwell model.

Stress relaxation and creep are obtained by numerical Laplace inversion of
the complex modulus (see :func:`hermes_rheo.lve_models.base.invert_laplace`),
except where an exact expression is given. They require :math:`t > 0`.

References
----------
.. [Schiessel1995] Schiessel, H., Metzler, R., Blumen, A. and Nonnenmacher,
   T. F., *Generalized viscoelastic models: their fractional equations with
   solutions*, J. Phys. A **28**, 6567 (1995). DOI: 10.1088/0305-4470/28/23/012.
.. [Jaishankar2013] Jaishankar, A. and McKinley, G. H., *Power-law rheology in
   the bulk and at the interface*, Proc. R. Soc. A **469**, 20120284 (2013).
   DOI: 10.1098/rspa.2012.0284.
.. [Bonfanti2020] Bonfanti, A., Kaplan, J. L., Charras, G. and Kabla, A.,
   *Fractional viscoelastic models for power-law materials*, Soft Matter
   **16**(26), 6002--6020 (2020). DOI: 10.1039/d0sm00354a.
.. [Song2023] Song, J., Holten-Andersen, N. and McKinley, G. H.,
   *Non-Maxwellian viscoelastic stress relaxations in soft matter*, Soft
   Matter **19**(41), 7885--7906 (2023). DOI: 10.1039/d3sm00736g.
.. [MirandaValdez2025] Miranda-Valdez, I. Y., Niinisto, H., Makinen, T. P.,
   Lejon, E., Koivisto, J. and Alava, M. J., *pyRheo: an open-source Python
   package for complex rheology*, Digital Discovery **4**(4), 1075--1082
   (2025). DOI: 10.1039/d5dd00021a.
.. [Mainardi2010] Mainardi, F., *Fractional Calculus and Waves in Linear
   Viscoelasticity*, Imperial College Press (2010). DOI: 10.1142/p614.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

from hermes_rheo.lve_models.base import LVEModel, positive

__all__ = [
    "SpringPotModel",
    "FractionalMaxwellGelModel",
    "FractionalMaxwellLiquidModel",
    "FractionalMaxwellModel",
    "FractionalKelvinVoigtModel",
    "FractionalZenerModel",
]

_REFS = ("Schiessel et al. 1995", "Jaishankar and McKinley 2013", "Bonfanti et al. 2020")


def _order(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or not 0.0 < v < 1.0:
        raise ValueError(f"{name} must be in the open interval (0, 1), got {value!r}")
    return v


def _series(a, b):
    return a * b / (a + b)


class SpringPotModel(LVEModel):
    r"""Springpot: power-law element.

    .. math::

       G(t) = \frac{V t^{-\alpha}}{\Gamma(1-\alpha)}, \qquad
       J(t) = \frac{t^{\alpha}}{V\,\Gamma(1+\alpha)}, \qquad
       G^*(s) = V s^{\alpha}

    Parameters
    ----------
    V : float
        Quasi-modulus in Pa s^alpha.
    alpha : float
        Order, in (0, 1).
    """

    name = "SpringPot"
    references = _REFS

    def __init__(self, V: float, alpha: float) -> None:
        self.V = positive("V", V)
        self.alpha = _order("alpha", alpha)
        self.parameters = {"V": self.V, "alpha": self.alpha}

    def complex_modulus(self, s):
        return self.V * s**self.alpha

    def stress_relaxation(self, t):
        t = _positive_time(t)
        return self.V * t**-self.alpha / gamma(1.0 - self.alpha)

    def creep(self, t):
        t = _positive_time(t)
        return t**self.alpha / (self.V * gamma(1.0 + self.alpha))


def _positive_time(t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("t must be strictly positive for this model")
    return t


class FractionalMaxwellGelModel(LVEModel):
    r"""Spring :math:`G` in series with a springpot :math:`(V, \alpha)`.

    .. math::

       G^*(s) = \frac{G V s^{\alpha}}{G + V s^{\alpha}}

    The relaxation modulus is :math:`G\,E_{\alpha}(-(t/\tau_c)^{\alpha})` with
    :math:`\tau_c = (V/G)^{1/\alpha}` (Mittag-Leffler function).

    Parameters
    ----------
    G : float
        Modulus in Pa.
    V : float
        Quasi-modulus in Pa s^alpha.
    alpha : float
        Order, in (0, 1).
    """

    name = "FractionalMaxwellGel"
    references = _REFS

    def __init__(self, G: float, V: float, alpha: float) -> None:
        self.G = positive("G", G)
        self.V = positive("V", V)
        self.alpha = _order("alpha", alpha)
        self.parameters = {"G": self.G, "V": self.V, "alpha": self.alpha}

    def complex_modulus(self, s):
        return _series(self.G, self.V * s**self.alpha)


class FractionalMaxwellLiquidModel(LVEModel):
    r"""Springpot :math:`(G, \beta)` in series with a dashpot :math:`\eta`.

    .. math::

       G^*(s) = \frac{G s^{\beta}\,\eta s}{G s^{\beta} + \eta s}

    Parameters
    ----------
    G : float
        Quasi-modulus in Pa s^beta.
    eta : float
        Viscosity in Pa s.
    beta : float
        Order, in (0, 1).
    """

    name = "FractionalMaxwellLiquid"
    references = _REFS

    def __init__(self, G: float, eta: float, beta: float) -> None:
        self.G = positive("G", G)
        self.eta = positive("eta", eta)
        self.beta = _order("beta", beta)
        self.parameters = {"G": self.G, "eta": self.eta, "beta": self.beta}

    def complex_modulus(self, s):
        return _series(self.G * s**self.beta, self.eta * s)


class FractionalMaxwellModel(LVEModel):
    r"""Springpot :math:`(G, \beta)` in series with springpot :math:`(V, \alpha)`.

    .. math::

       G^*(s) = \frac{G s^{\beta}\,V s^{\alpha}}{G s^{\beta} + V s^{\alpha}},
       \qquad \alpha > \beta

    Parameters
    ----------
    G : float
        Quasi-modulus in Pa s^beta.
    V : float
        Quasi-modulus in Pa s^alpha.
    alpha, beta : float
        Orders in (0, 1) with ``alpha > beta``.
    """

    name = "FractionalMaxwell"
    references = _REFS

    def __init__(self, G: float, V: float, alpha: float, beta: float) -> None:
        self.G = positive("G", G)
        self.V = positive("V", V)
        self.alpha = _order("alpha", alpha)
        self.beta = _order("beta", beta)
        if self.alpha <= self.beta:
            raise ValueError("alpha must be greater than beta")
        self.parameters = {"G": self.G, "V": self.V, "alpha": self.alpha, "beta": self.beta}

    def complex_modulus(self, s):
        return _series(self.G * s**self.beta, self.V * s**self.alpha)


class FractionalKelvinVoigtModel(LVEModel):
    r"""Springpot :math:`(G, \beta)` in parallel with springpot :math:`(V, \alpha)`.

    .. math::

       G^*(s) = G s^{\beta} + V s^{\alpha}, \qquad
       G(t) = \frac{G t^{-\beta}}{\Gamma(1-\beta)}
              + \frac{V t^{-\alpha}}{\Gamma(1-\alpha)}

    Parameters
    ----------
    G : float
        Quasi-modulus in Pa s^beta.
    V : float
        Quasi-modulus in Pa s^alpha.
    alpha, beta : float
        Orders in (0, 1).
    """

    name = "FractionalKelvinVoigt"
    references = _REFS

    def __init__(self, G: float, V: float, alpha: float, beta: float) -> None:
        self.G = positive("G", G)
        self.V = positive("V", V)
        self.alpha = _order("alpha", alpha)
        self.beta = _order("beta", beta)
        self.parameters = {"G": self.G, "V": self.V, "alpha": self.alpha, "beta": self.beta}

    def complex_modulus(self, s):
        return self.G * s**self.beta + self.V * s**self.alpha

    def stress_relaxation(self, t):
        t = _positive_time(t)
        return self.G * t**-self.beta / gamma(1.0 - self.beta) + self.V * t**-self.alpha / gamma(
            1.0 - self.alpha
        )


class FractionalZenerModel(LVEModel):
    r"""Springpot :math:`(K, \kappa)` in parallel with a fractional Maxwell model.

    .. math::

       G^*(s) = K s^{\kappa}
                + \frac{G s^{\beta}\,V s^{\alpha}}{G s^{\beta} + V s^{\alpha}},
       \qquad \alpha > \beta

    Parameters
    ----------
    G : float
        Quasi-modulus in Pa s^beta.
    V : float
        Quasi-modulus in Pa s^alpha.
    K : float
        Quasi-modulus in Pa s^kappa.
    alpha, beta, kappa : float
        Orders in (0, 1) with ``alpha > beta``.
    """

    name = "FractionalZener"
    references = _REFS

    def __init__(
        self, G: float, V: float, K: float, alpha: float, beta: float, kappa: float
    ) -> None:
        self.G = positive("G", G)
        self.V = positive("V", V)
        self.K = positive("K", K)
        self.alpha = _order("alpha", alpha)
        self.beta = _order("beta", beta)
        self.kappa = _order("kappa", kappa)
        if self.alpha <= self.beta:
            raise ValueError("alpha must be greater than beta")
        self.parameters = {
            "G": self.G,
            "V": self.V,
            "K": self.K,
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
        }

    def complex_modulus(self, s):
        series = _series(self.G * s**self.beta, self.V * s**self.alpha)
        return self.K * s**self.kappa + series
