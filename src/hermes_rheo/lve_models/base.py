"""Base class for linear viscoelastic (LVE) models.

A model is defined by its complex modulus :math:`G^*(s)` as a function of the
Laplace variable :math:`s`. From it:

* SAOS: :math:`G^*(\\omega) = G^*(s = i\\omega)`
* stress relaxation: :math:`G(t) = \\mathcal{L}^{-1}[G^*(s)/s]`
* creep: :math:`J(t) = \\mathcal{L}^{-1}[1/(s\\,G^*(s))]`

The inverse Laplace transforms use the fixed Talbot method [Abate2004]_
unless a subclass provides an exact expression.

Units: time in s, angular frequency in rad/s, moduli in Pa, compliance in 1/Pa.

References
----------
.. [Abate2004] Abate, J. and Valko, P. P., *Multi-precision Laplace transform
   inversion*, Int. J. Numer. Meth. Eng. **60**, 979-993 (2004).
   DOI: 10.1002/nme.995.
.. [Tschoegl1989] Tschoegl, N. W., *The Phenomenological Theory of Linear
   Viscoelastic Behavior*, Springer (1989). DOI: 10.1007/978-3-642-73602-5.
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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from piblin.data.data_collections import measurement as _measurement
from piblin.data.datasets.abc.split_datasets import (
    one_dimensional_composite_dataset as _composite,
)

from hermes_rheo import __version__

__all__ = ["LVEModel", "invert_laplace"]

PROTOCOLS = ("saos", "creep", "stress_relaxation")

_METHOD_LABELS = {
    "saos": "SAOS",
    "creep": "Creep",
    "stress_relaxation": "Stress relaxation",
}


def invert_laplace(
    F: Callable[[np.ndarray], np.ndarray],
    t: np.ndarray,
    n_terms: int = 16,
) -> np.ndarray:
    """Numerically invert a Laplace transform with the fixed Talbot method.

    Parameters
    ----------
    F : callable
        Transform :math:`F(s)`. Must accept and return complex arrays.
    t : np.ndarray
        Times in s. All values must be positive.
    n_terms : int, optional
        Number of contour points. 16 gives about 1e-9 relative accuracy in
        double precision; larger values lose accuracy to round-off.

    Returns
    -------
    np.ndarray
        :math:`f(t)`, same shape as *t*.
    """
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("t must be strictly positive for this protocol")
    flat = t.ravel()
    r = 2.0 * n_terms / (5.0 * flat)
    theta = np.arange(1, n_terms) * np.pi / n_terms
    cot = 1.0 / np.tan(theta)
    s = r[:, None] * theta[None, :] * (cot + 1j)[None, :]
    weight = 1.0 + 1j * (theta + (theta * cot - 1.0) * cot)
    terms = np.exp(flat[:, None] * s) * F(s) * weight[None, :]
    head = 0.5 * np.real(F(r + 0j)) * np.exp(r * flat)
    out = (r / n_terms) * (head + np.real(terms).sum(axis=1))
    return out.reshape(t.shape)


def positive(name: str, value: float) -> float:
    """Return *value* as a float after checking it is finite and positive."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return v


class LVEModel(ABC):
    """Abstract linear viscoelastic model.

    Subclasses implement :meth:`complex_modulus`. They may override
    :meth:`relaxation_modulus` and :meth:`creep_compliance` with exact
    expressions.

    Attributes
    ----------
    name : str
        Short model name, stored in the measurement metadata.
    parameters : dict
        Model parameters, stored in the measurement metadata.
    references : tuple of str
        Literature for the model, stored in the measurement metadata.
    """

    name: str = "LVEModel"
    references: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}

    @abstractmethod
    def complex_modulus(self, s: np.ndarray) -> np.ndarray:
        """Return :math:`G^*(s)` in Pa for complex Laplace variable *s*."""

    def saos(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the storage and loss moduli :math:`(G', G'')` in Pa.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency in rad/s (positive).
        """
        w = np.asarray(omega, dtype=float)
        if np.any(w <= 0.0):
            raise ValueError("omega must be strictly positive")
        g = self.complex_modulus(1j * w.astype(complex))
        return g.real, g.imag

    def stress_relaxation(self, t: np.ndarray) -> np.ndarray:
        """Return the relaxation modulus :math:`G(t)` in Pa for :math:`t > 0`.

        Any impulsive term at :math:`t = 0` (a parallel dashpot) is excluded.
        """
        return invert_laplace(lambda s: self.complex_modulus(s) / s, t)

    def creep(self, t: np.ndarray) -> np.ndarray:
        """Return the creep compliance :math:`J(t)` in 1/Pa for :math:`t > 0`."""
        return invert_laplace(lambda s: 1.0 / (s * self.complex_modulus(s)), t)

    def to_measurement(
        self,
        protocol: str,
        x: np.ndarray,
        *,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> _measurement.Measurement:
        """Run a simulated experiment and return a piblin ``Measurement``.

        Parameters
        ----------
        protocol : {"saos", "creep", "stress_relaxation"}
            Experiment to simulate.
        x : np.ndarray
            Angular frequency in rad/s for ``"saos"``, time in s otherwise.
        conditions, details : dict, optional
            Extra entries merged into ``Measurement.conditions`` / ``details``.

        Returns
        -------
        piblin.data.data_collections.measurement.Measurement
            One composite one-dimensional dataset. ``conditions`` hold
            ``method`` and ``simulated``. ``details`` hold the model name,
            parameters, protocol, grid, references and software version.
        """
        key = protocol.strip().lower()
        if key not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
        grid = np.asarray(x, dtype=float).ravel()
        if grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("x must be a non-empty array of finite values")

        dataset = getattr(self, f"_dataset_{key}")(grid)
        merged_conditions = {
            "method": f"{_METHOD_LABELS[key]} (simulated {self.name})",
            "simulated": True,
        }
        merged_conditions.update(conditions or {})
        merged_details = {
            "model": self.name,
            "parameters": dict(self.parameters),
            "protocol": key,
            "grid": {
                "variable": "angular frequency" if key == "saos" else "time",
                "min": float(grid.min()),
                "max": float(grid.max()),
                "points": int(grid.size),
            },
            "references": list(self.references),
            "software": f"hermes-rheo {__version__}",
        }
        merged_details.update(details or {})
        return _measurement.Measurement(
            datasets=[dataset],
            conditions=merged_conditions,
            details=merged_details,
        )

    def _dataset_stress_relaxation(
        self, t: np.ndarray
    ) -> _composite.OneDimensionalCompositeDataset:
        g_t = np.asarray(self.stress_relaxation(t), dtype=float)
        return _composite.OneDimensionalCompositeDataset(
            data_arrays=[t, g_t],
            data_array_names=["time", "relaxation modulus"],
            data_array_units=["s", "Pa"],
            default_independent_name="time",
            default_dependent_name="relaxation modulus",
            source=f"Simulated {self.name} stress relaxation",
        )

    def _dataset_creep(self, t: np.ndarray) -> _composite.OneDimensionalCompositeDataset:
        j_t = np.asarray(self.creep(t), dtype=float)
        return _composite.OneDimensionalCompositeDataset(
            data_arrays=[t, j_t],
            data_array_names=["time", "creep compliance"],
            data_array_units=["s", "1/Pa"],
            default_independent_name="time",
            default_dependent_name="creep compliance",
            source=f"Simulated {self.name} creep",
        )

    def _dataset_saos(self, omega: np.ndarray) -> _composite.OneDimensionalCompositeDataset:
        gp, gpp = (np.asarray(v, dtype=float) for v in self.saos(omega))
        return _composite.OneDimensionalCompositeDataset(
            data_arrays=[omega, gp, gpp, np.hypot(gp, gpp), gpp / gp],
            data_array_names=[
                "angular frequency",
                "storage modulus",
                "loss modulus",
                "complex modulus",
                "tan(delta)",
            ],
            data_array_units=["rad/s", "Pa", "Pa", "Pa", ""],
            default_independent_name="angular frequency",
            default_dependent_name="storage modulus",
            source=f"Simulated {self.name} SAOS",
        )
