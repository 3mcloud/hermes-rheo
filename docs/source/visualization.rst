Visualization
=============

This section documents the plotting helpers in
:mod:`hermes_rheo.visualization`.  They operate on the
:class:`~piblin.data.data_collections.experiment_set.ExperimentSet`
/ :class:`~piblin.data.data_collections.measurement_set.MeasurementSet`
objects produced by the readers and transforms.

All helpers work with the SciencePlots ``science + notebook`` style
applied by :func:`~hermes_rheo.visualization.use_science_style`.

Rheology plots
--------------
Van Gurp–Palmen and Cole–Cole diagrams, colour-coded by temperature.

.. automodule:: hermes_rheo.visualization.visualize
   :members: plot_vgp, plot_cole_cole
   :undoc-members:
   :show-inheritance:

OWChirp visualiser
------------------
A convenience object that wraps the per-wave extraction, plotting,
PNG-frame saving, animation and Fourier-domain analysis of a processed
OWChirp experiment (see :class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`).

.. automodule:: hermes_rheo.visualization.owchirp
   :members: OWChirpVisualizer
   :undoc-members:
   :show-inheritance:

Relaxation-spectrum plots
-------------------------
Helpers for the pyReSpect-based
:class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`
transform:

* :func:`~hermes_rheo.visualization.spectrum.plot_relaxation_spectrum`
  --- continuous CRS :math:`h(s)` with optional discrete-mode stems and
  a shaded uncertainty band (frequency branch).
* :func:`~hermes_rheo.visualization.spectrum.plot_spectrum_diagnostics`
  --- L-curve, Bayesian log-evidence, and AIC-scan diagnostics.

.. automodule:: hermes_rheo.visualization.spectrum
   :members: plot_relaxation_spectrum, plot_spectrum_diagnostics
   :undoc-members:
   :show-inheritance:

Style helpers
-------------
Apply or reset the publication-quality Matplotlib style used throughout
the tutorials.

.. automodule:: hermes_rheo.visualization.style
   :members: use_science_style, reset_style
   :undoc-members:
