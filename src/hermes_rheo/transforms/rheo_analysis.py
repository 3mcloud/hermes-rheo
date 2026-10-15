"""Deprecated compatibility shim for :class:`OWChirpAnalysis`.

This module previously contained a large ``RheoAnalysis`` class that mixed
OWChirp / Frequency Chirp FFT signal-processing with a cascade of
``switch_coordinates()`` calls for every other rheological method (Creep,
Amplitude Sweep, Temperature Ramp, …).

Since hermes-rheo 2.0:

* The per-method default coordinates are now set at read time by the file
  readers themselves — see the ``_METHOD_DEFAULTS`` machinery in
  :mod:`hermes_rheo.file_readers.trios_rheo_txt`.
* The OWChirp / Frequency Chirp FFT logic lives in the new focused
  :class:`hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`.

``RheoAnalysis`` is preserved here as a thin subclass that emits a
``DeprecationWarning`` and forwards all behaviour to
:class:`OWChirpAnalysis`.  Existing pipelines using
``RheoAnalysis() + AutomatedMasterCurve() + ...`` continue to work
unchanged.

The alias will be removed in hermes-rheo 3.0.
"""
from __future__ import annotations

import warnings
from typing import Any

from hermes_rheo.transforms.owchirp_analysis import (
    OWChirpAnalysis,
    _canonicalise_method,  # noqa: F401 — re-exported for downstream callers
    _UNIT_PREFIXES,  # noqa: F401 — re-exported for downstream callers
)


class RheoAnalysis(OWChirpAnalysis):
    """Deprecated alias for :class:`OWChirpAnalysis`.

    Prior releases used ``RheoAnalysis`` as a catch-all transform that
    (1) applied OWChirp / Frequency Chirp FFT analysis to chirp steps and
    (2) issued ``switch_coordinates()`` calls to set method-appropriate
    default coordinates on every other kind of measurement.

    Item (2) is now handled directly by the file readers, so ``RheoAnalysis``
    on a non-chirp measurement is a no-op — same as :class:`OWChirpAnalysis`.
    Emit a ``DeprecationWarning`` on construction so downstream code has
    time to migrate to :class:`OWChirpAnalysis`.

    Examples
    --------
    Old (still works, emits a warning):

    >>> from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
    >>> RheoAnalysis()  # doctest: +SKIP

    New:

    >>> from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
    >>> OWChirpAnalysis()  # doctest: +SKIP
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "RheoAnalysis is a deprecated alias for OWChirpAnalysis and will "
            "be removed in a future major release.  Import OWChirpAnalysis from "
            "hermes_rheo.transforms.owchirp_analysis instead.  Per-method "
            "coordinate defaults (Creep, Frequency Sweep, Temperature Ramp, "
            "…) are now set by the reader; no post-read transform is needed.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["RheoAnalysis"]
