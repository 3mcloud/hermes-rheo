"""Deprecated alias of :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`.

``RheoAnalysis`` formerly combined OWCh Fourier analysis with per-method
``switch_coordinates()`` calls. Since hermes-rheo 2.0 the file readers set
the default coordinates for each method (see ``_METHOD_DEFAULTS`` in
:mod:`hermes_rheo.file_readers.trios_rheo_txt`), and the OWCh analysis lives
in :class:`OWChirpAnalysis`. This module keeps a subclass that warns on
construction and otherwise behaves identically. It will be removed in
hermes-rheo 3.0.
"""

from __future__ import annotations

import warnings
from typing import Any

from hermes_rheo.transforms.owchirp_analysis import (
    _UNIT_PREFIXES,  # noqa: F401 — re-exported for downstream callers
    OWChirpAnalysis,
    _canonicalise_method,  # noqa: F401 — re-exported for downstream callers
)


class RheoAnalysis(OWChirpAnalysis):
    """Deprecated alias for :class:`OWChirpAnalysis`.

    Emits a ``DeprecationWarning`` on construction. Measurements that are
    not Arbitrary Wave or Frequency Chirp steps are left unchanged, because
    the readers already set their default coordinates.

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
            "be removed in hermes-rheo 3.0. Import OWChirpAnalysis from "
            "hermes_rheo.transforms.owchirp_analysis instead. Per-method "
            "coordinate defaults are now set by the file reader.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["RheoAnalysis"]
