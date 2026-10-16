Linear viscoelastic models
==========================

:mod:`hermes_rheo.lve_models` simulates SAOS, creep and stress relaxation
for classical and fractional linear viscoelastic (LVE) models. Each
experiment returns a piblin
:class:`~piblin.data.data_collections.measurement.Measurement` with the same
column names and units as real data, so it works with every hermes-rheo
transform.

Each model defines its complex modulus :math:`G^*(s)`:

- SAOS: :math:`G^*(i\omega) = G' + iG''`
- Stress relaxation: :math:`G(t) = \mathcal{L}^{-1}[G^*(s)/s]`
- Creep: :math:`J(t) = \mathcal{L}^{-1}[1/(s\,G^*(s))]`

Closed forms are used where they exist. Otherwise the Laplace inversion is
numerical (fixed Talbot). Models with a parallel dashpot (Kelvin–Voigt,
Jeffreys) have a :math:`\delta(t)` term in :math:`G(t)`; only the
:math:`t>0` part is returned.

Models
------

.. list-table::
   :header-rows: 1

   * - Class
     - Parameters
   * - ``MaxwellModel``
     - ``G``, ``tau``
   * - ``MultiModeMaxwellModel``
     - ``g``, ``tau``, ``G_inf``
   * - ``KelvinVoigtModel``
     - ``G``, ``eta``
   * - ``JeffreysModel``
     - ``G``, ``tau``, ``eta_s``
   * - ``SpringPotModel``
     - ``V``, ``alpha``
   * - ``FractionalMaxwellGelModel``
     - ``G``, ``V``, ``alpha``
   * - ``FractionalMaxwellLiquidModel``
     - ``G``, ``eta``, ``beta``
   * - ``FractionalMaxwellModel``
     - ``G``, ``V``, ``alpha``, ``beta`` (``alpha > beta``)
   * - ``FractionalKelvinVoigtModel``
     - ``G``, ``V``, ``alpha``, ``beta``
   * - ``FractionalZenerModel``
     - ``G``, ``V``, ``K``, ``alpha``, ``beta``, ``kappa``

Fractional orders lie in the open interval (0, 1). Parameter names follow
pyRheo.

Example
-------

.. code-block:: python

   import numpy as np
   from hermes_rheo.lve_models import FractionalMaxwellGelModel, Protocols

   model = FractionalMaxwellGelModel(G=1e4, V=5e3, alpha=0.6)
   saos = Protocols.saos(model, np.logspace(-2, 2, 40))
   creep = Protocols.creep(model, np.logspace(-2, 2, 40))
   relax = Protocols.stress_relaxation(model, np.logspace(-2, 2, 40))

Metadata
--------

``conditions["method"]`` names the protocol and model, and
``conditions["simulated"]`` is ``True``. ``details`` holds ``model``,
``parameters``, ``protocol``, ``grid`` (variable, min, max, points),
``references`` and ``software``.

``Protocols.experiment_set(model, [...])`` builds several experiments in one
call.

Validation
----------

The tests check closed forms against the inverted :math:`G^*`, the identity
:math:`\int_0^t G(t-u)J(u)\,du = t`, the Mittag-Leffler series, and the
Maxwell limit of the fractional gel as :math:`\alpha \to 1`. pyRheo is used
only as an optional cross-check; it is GPL-3.0 and is not a dependency.

References
----------

- Bonfanti, A., Kaplan, J. L., Charras, G. and Kabla, A., *Fractional
  viscoelastic models for power-law materials*, *Soft Matter* **16**(26),
  6002--6020 (2020). DOI: `10.1039/d0sm00354a <https://doi.org/10.1039/d0sm00354a>`_.
- Schiessel, H., Metzler, R., Blumen, A., Nonnenmacher, T. F. "Generalized
  viscoelastic models: their fractional equations with solutions."
  *J. Phys. A* 28, 6567 (1995).
- Garrappa, R. "Numerical evaluation of two and three parameter
  Mittag-Leffler functions." *SIAM J. Numer. Anal.* 53, 1350 (2015).
- Talbot, A. "The accurate numerical inversion of Laplace transforms."
  *J. Inst. Math. Appl.* 23, 97 (1979).
- Song, J., Holten-Andersen, N. and McKinley, G. H., *Non-Maxwellian viscoelastic stress relaxations in soft matter*, Soft Matter **19**(41), 7885–7906 (2023). DOI: 10.1039/d3sm00736g.
- Miranda-Valdez, I. Y., Niinisto, H., Makinen, T. P., Lejon, E., Koivisto, J. and Alava, M. J., *pyRheo: an open-source Python package for complex rheology*, Digital Discovery **4**(4), 1075–1082 (2025). DOI: 10.1039/d5dd00021a.
- pyRheo source: https://github.com/3mcloud/pyRheo

API
---

.. automodule:: hermes_rheo.lve_models
   :members:
   :imported-members:
