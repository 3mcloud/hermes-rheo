Transforms
==========

This section documents every transform in :mod:`hermes_rheo.transforms`.
Transforms consume and return
:class:`~piblin.data.data_collections.measurement.Measurement` or
:class:`~piblin.data.data_collections.measurement_set.MeasurementSet`
objects; they compose with the ``+`` operator into pipelines.

For a general introduction see :doc:`background`; for full end-to-end
examples see the :doc:`tutorials` and the
`tutorial_notebooks/ <https://github.com/3mcloud/hermes-rheo/tree/main/tutorial_notebooks>`_
folder.

OWChirp Analysis (recommended)
------------------------------
Time-resolved viscoelastic-moduli extraction from Arbitrary Wave /
Frequency Chirp steps.  Implements the *Optimally Windowed Chirp*
technique of Geri *et al.* (2018) [#ref-Geri2018]_; the
strain-controlled-instrument workflow is documented in
Perego (2025) [#ref-Perego2025]_.

.. note::

   **OWChirp measurement modes.** OWChirp measurements were originally
   only available via the TRIOS *Arbitrary Wave* generator on the
   `TA Instruments <https://www.tainstruments.com/>`_ **ARES-G2** and
   **DHR** series, which export only the transient stress and strain
   signal.  ``OWChirpAnalysis`` was developed to Fourier-transform
   that signal into frequency-domain moduli.

   Alternatives:

   * **ARES-G3** ships a native TRIOS *Frequency Chirp* method that
     exports both transient and pre-computed frequency-domain data.
   * **DHR series** — the same *Frequency Chirp* method can be
     enabled by requesting an activation key from
     `TA Instruments <https://www.tainstruments.com/>`_.
   * **ARES-G2** — the *Arbitrary Wave* macro remains the only option.

   ``OWChirpAnalysis`` handles both step types; for Frequency Chirp
   steps it also applies inertia and compliance compensation using
   the geometry constants exported by TRIOS.

.. automodule:: hermes_rheo.transforms.owchirp_analysis
   :members: OWChirpAnalysis
   :undoc-members:
   :show-inheritance:

RheoAnalysis (deprecated alias)
-------------------------------
Retained for backwards compatibility.  New code should use
:class:`~hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis` directly;
per-method default coordinates for non-chirp measurements (Creep,
Frequency Sweep, Temperature Ramp, …) are now assigned by the reader,
so no post-read transform is required.

.. automodule:: hermes_rheo.transforms.rheo_analysis
   :members: RheoAnalysis
   :undoc-members:
   :show-inheritance:

OWChirp Signal Generation
-------------------------
Helper for authoring TRIOS Arbitrary Wave macros implementing an OWChirp
signal.

.. automodule:: hermes_rheo.transforms.owchirp_generation
   :members: OWChirpGeneration
   :undoc-members:
   :show-inheritance:

Time–Temperature Superposition
------------------------------
Automated master-curve construction via the data-driven Gaussian-process
method of Lennon, McKinley and Swan (2023) [#ref-Lennon2023]_.

.. automodule:: hermes_rheo.transforms.automated_mastercurve
   :members: AutomatedMasterCurve, CompositeMasterCurve, MasterCurveMeasurement, MasterCurveSet
   :undoc-members:
   :show-inheritance:

Rheological Model Fitting
-------------------------
Fits rheological models to master-curve data using
`pyRheo <https://github.com/mirandi1/pyRheo>`_ (Miranda-Valdez *et al.*)
or a built-in multi-mode Maxwell (Prony series) implementation
[#ref-MirandaValdez2025]_. Background on fractional models and non-Maxwellian
relaxation: [#ref-Bonfanti2020]_, [#ref-Song2023]_.
pyRheo is an optional dependency (GPL-3.0, not on PyPI) and is imported
only when one of its models is requested:

.. code-block:: bash

   pip install git+https://github.com/mirandi1/pyRheo.git

``FitModel`` supports both oscillation-domain fitting
(:math:`G'(\omega), G''(\omega)`) and time-domain relaxation fitting
(:math:`G(t)`) via ``domain="relaxation"``.

Calling ``FitModel`` returns a piblin ``Measurement`` holding a
one-dimensional composite dataset with the measured and fitted arrays
(``angular frequency``, ``storage modulus``, ``loss modulus``, ``fitted
storage modulus``, ``fitted loss modulus``; or ``time``, ``relaxation
modulus``, ``fitted relaxation modulus``). Its ``details`` record the model,
fitted parameters, goodness of fit, method, references and software version.
The transform instance keeps ``fitted_model``, ``fit_params``, ``plot()`` and
``print_parameters()`` for convenience.

.. automodule:: hermes_rheo.transforms.fit_model
   :members: FitModel, MultiModeMaxwellModel
   :undoc-members:
   :show-inheritance:

Shift-Factor Models
-------------------
Fit the temperature dependence of the horizontal time-temperature
superposition shift factors :math:`a_T` (produced by
:class:`~hermes_rheo.transforms.automated_mastercurve.AutomatedMasterCurve`)
to the Williams-Landel-Ferry [#ref-WLF1955]_ or Arrhenius [#ref-Arrhenius1889]_
equations.

.. automodule:: hermes_rheo.transforms.shift_factor_models
   :members: ShiftFactorModel, WLFModel, ArrheniusModel
   :undoc-members:
   :show-inheritance:

Temperature Ramp Analysis
-------------------------
Extract Tg and interpolated viscoelastic properties from a
:class:`~piblin.data.ExperimentSet` containing temperature-ramp or
temperature-sweep steps, and report the test frequency read from the data.
Tg is taken at the most prominent ``tan(delta)`` peak (default; further
peaks, e.g. of a second phase, are reported too), at the onset of the
storage-modulus drop (intersection of the glassy and inflection tangents of
log G'), or both (``tg_method``), following [#ref-RH100]_.

.. automodule:: hermes_rheo.transforms.temperature_ramp_analysis
   :members: TemperatureRampAnalysis, TemperatureRampMeasurement, TemperatureRampResult,
      find_tan_delta_peaks, TanDeltaPeaks, storage_modulus_onset, StorageModulusOnset
   :undoc-members:
   :show-inheritance:

Recovery Rheology
-----------------
Oscillatory (OCR) and step-strain (creep-recovery) rheology analyses
following Lee, Weigandt, Kelley and Rogers (2019) [#ref-Lee2019]_.

.. automodule:: hermes_rheo.transforms.recovery_rheology
   :members: RecoveryRheology, RecoveryRheologyMeasurementSet, RecoveryRheologyResult
   :undoc-members:
   :show-inheritance:

Mutation Number
---------------
Computes the mutation number :math:`\mathrm{Mu}` following Jamali & McKinley
(2022) [#ref-Jamali2022]_ and the mutation-number concept of Mours & Winter
(1994) [#ref-MoursWinter1994]_, from successive single-wave or correlated
OWCh (e.g. TORCh) spectra. Each output measurement holds time, mutation number,
angular frequency and, for correlated steps, temperature, so the result can be
passed to ``Kymograph``; its ``details`` store the mean and maximum mutation
number and the calculation settings.

.. automodule:: hermes_rheo.transforms.mutation_number
   :members: MutationNumber, MutationNumberMeasurementSet
   :undoc-members:
   :show-inheritance:

Kymograph
---------
Produces a 2-D kymograph (Temperature × Frequency → viscoelastic
quantity) from a Thermally-Optimised Ramped Chirp (TORCh) sequence —
introduced by Vadillo, Perego *et al.* (2026) [#ref-Vadillo2026]_. Either
variable can go on either axis (``x``, ``y``, ``log_x``, ``log_y``), and
``KymographResult.plot`` sets the colour scale (``log_z``, ``vmin``,
``vmax``).

.. automodule:: hermes_rheo.transforms.kymograph
   :members: Kymograph, KymographResult
   :undoc-members:
   :show-inheritance:

Frequency → Time Relaxation Modulus (iRheoTempo)
------------------------------------------------
Reconstructs time-domain relaxation modulus :math:`G(t)` from
frequency-domain :math:`G'(\omega), G''(\omega)` using the interval-slope
method implemented in iRheoTempo (Ramírez and Tassieri, 2026)
[#ref-RamirezTassieri2026]_.

This transform is intended to compose with existing pipelines. Common
patterns include raw frequency sweeps, and chaining after
``SplitTemperatureSweep`` or ``OWChirpAnalysis``. The resulting
time-domain dataset can be fitted with
``FitModel(domain="relaxation")``.

.. automodule:: hermes_rheo.transforms.frequency_to_time
   :members: FrequencyToTime, FrequencyToTimeResult
   :undoc-members:
   :show-inheritance:

Relaxation Spectrum (pyReSpect)
-------------------------------
Continuous relaxation spectrum :math:`H(s)` and discrete Prony series
:math:`(g_i, \tau_i)` recovered from either a stress-relaxation
:math:`G(t)` experiment (Shanbhag, 2019 [#ref-Shanbhag2019]_; Takeh &
Shanbhag, 2013 [#ref-TakehShanbhag2013]_) or an oscillatory frequency sweep
:math:`G'(\omega), G''(\omega)` (Shanbhag, 2020 [#ref-Shanbhag2020]_) via
the pyReSpect solvers, which are vendored in ``hermes_rheo._vendor``
(MIT licence).

Both branches are exposed through a single ``MeasurementTransform`` with
an explicit ``domain={"time", "frequency"}`` argument.

.. automodule:: hermes_rheo.transforms.relaxation_spectrum
   :members: RelaxationSpectrum, RelaxationSpectrumResult
   :undoc-members:
   :show-inheritance:

Split Temperature Sweep
-----------------------
Splits a multi-temperature isothermal-frequency-sweep block into
individual frequency-sweep measurements suitable for downstream
master-curve construction.

.. automodule:: hermes_rheo.transforms.split_temperature_sweep
   :members: SplitTemperatureSweep
   :undoc-members:
   :show-inheritance:

States to Conditions
--------------------
Convert a data-array (e.g. a per-measurement transient) into a scalar
condition (``average`` / ``first point`` / ``last point``).

.. automodule:: hermes_rheo.transforms.states_to_conditions
   :members: StatesToConditions
   :undoc-members:
   :show-inheritance:

References
----------

.. [#ref-Geri2018] Geri, M., Keshavarz, B., Divoux, T., Clasen, C.,
   Curtis, D. J. and McKinley, G. H., *Time-Resolved Mechanical
   Spectroscopy of Soft Materials via Optimally Windowed Chirps*,
   Phys. Rev. X **8**, 041042 (2018).
   DOI: `10.1103/PhysRevX.8.041042
   <https://doi.org/10.1103/PhysRevX.8.041042>`_ (open access) —
   preprint: `arXiv:1804.03061 <https://arxiv.org/abs/1804.03061>`_.

.. [#ref-Perego2025] Perego, A., Vadillo, D. C., Mills, M. J. L., Das, M. and
   McKinley, G. H., *Evaluation of optimally windowed chirp signals in
   industrial rheological measurements: method development and data
   processing*, Rheologica Acta **64**, 391--406 (2025).
   DOI: `10.1007/s00397-025-01511-0
   <https://doi.org/10.1007/s00397-025-01511-0>`_.
   Application of ``hermes-rheo`` to time-resolved OWChirp analysis
   on a strain-controlled rheometer.

.. [#ref-Lennon2023] Lennon, K. R., McKinley, G. H. and Swan, J. W.,
   *A data-driven method for automated data superposition with
   applications in soft matter science*, Data-Centric Engineering
   **4**, e13 (2023).
   DOI: `10.1017/dce.2023.3
   <https://doi.org/10.1017/dce.2023.3>`_ —
   preprint: `arXiv:2204.09521 <https://arxiv.org/abs/2204.09521>`_.

.. [#ref-Vadillo2026] Vadillo, D. C., Perego, A. *et al.*,
   *TORCh: Thermally Optimised Ramped Chirp for time-temperature
   spectroscopy of polymers*, Macromolecules (2026).
   DOI: `10.1021/acs.macromol.6c00679
   <https://doi.org/10.1021/acs.macromol.6c00679>`_.
   Introduces the TORCh sequence rendered by
   :class:`~hermes_rheo.transforms.kymograph.Kymograph`.

.. [#ref-RH100] Whitcomb, K., *Measurement of Glass Transition Temperatures
   by Dynamic Mechanical Analysis and Rheology*, TA Instruments application
   note RH100. `<https://www.tainstruments.com/>`_.

.. [#ref-Lee2019] Lee, J. C.-W., Weigandt, K. M., Kelley, E. G. and Rogers,
   S. A., *Structure-Property Relationships via Recovery Rheology in
   Viscoelastic Materials*, Phys. Rev. Lett. **122**, 248003 (2019).
   DOI: `10.1103/PhysRevLett.122.248003
   <https://doi.org/10.1103/PhysRevLett.122.248003>`_.

.. [#ref-MoursWinter1994] Mours, M. and Winter, H. H., *Time-resolved
   rheometry*, Rheologica Acta **33**(5), 385--397 (1994).
   DOI: `10.1007/BF00366581 <https://doi.org/10.1007/BF00366581>`_.

.. [#ref-Jamali2022] Jamali, S. and McKinley, G. H., *The Mnemosyne number
   and the rheology of remembrance*, J. Rheol. **66**\ (5), 1027–1039
   (2022).
   DOI: `10.1122/8.0000432 <https://doi.org/10.1122/8.0000432>`_.

.. [#ref-Bonfanti2020] Bonfanti, A., Kaplan, J. L., Charras, G. and Kabla, A., *Fractional viscoelastic models for power-law materials*, Soft Matter **16**(26), 6002–6020 (2020). DOI: 10.1039/d0sm00354a.

.. [#ref-Song2023] Song, J., Holten-Andersen, N. and McKinley, G. H., *Non-Maxwellian viscoelastic stress relaxations in soft matter*, Soft Matter **19**(41), 7885–7906 (2023). DOI: 10.1039/d3sm00736g.

.. [#ref-MirandaValdez2025] Miranda-Valdez, I. Y., Niinisto, H., Makinen, T. P., Lejon, E., Koivisto, J. and Alava, M. J., *pyRheo: an open-source Python package for complex rheology*, Digital Discovery **4**(4), 1075–1082 (2025). DOI: 10.1039/d5dd00021a.
   Provides the fractional and classical viscoelastic models fitted by
   :class:`~hermes_rheo.transforms.fit_model.FitModel`.

.. [#ref-RamirezTassieri2026] Ramírez, J. and Tassieri, M.,
   *i-Rheo-Tempo: A model-free, quadrature-free reconstruction of the
   shear relaxation modulus from complex viscosity*, J. Rheol. **70**,
   1089-1104 (2026). DOI: `10.1122/8.0001232
   <https://doi.org/10.1122/8.0001232>`_. Reference implementation:
   University of Glasgow Research Data, `10.5525/gla.researchdata.2230
   <https://doi.org/10.5525/gla.researchdata.2230>`_.

.. [#ref-Shanbhag2019] Shanbhag, S., *pyReSpect: A Computer Program to
   Extract Discrete and Continuous Spectra from Stress Relaxation
   Experiments*, Macromol. Theory Simul. **28**, 1900005 (2019).
   DOI: `10.1002/mats.201900005
   <https://doi.org/10.1002/mats.201900005>`_.
   Provides the time-domain solver invoked by
   :class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`
   with ``domain="time"``.

.. [#ref-Shanbhag2020] Shanbhag, S., *Relaxation spectra using nonlinear
   Tikhonov regularization with a Bayesian criterion*, Rheologica Acta
   **59**, 509 (2020).
   DOI: `10.1007/s00397-020-01212-w
   <https://doi.org/10.1007/s00397-020-01212-w>`_.
   Provides the frequency-domain solver invoked by
   :class:`~hermes_rheo.transforms.relaxation_spectrum.RelaxationSpectrum`
   with ``domain="frequency"``.

.. [#ref-TakehShanbhag2013] Takeh, A. and Shanbhag, S., *A computer program
   to extract the continuous and discrete relaxation spectra from
   dynamic viscoelastic measurements*, Applied Rheology **23**, 24628
   (2013).
   DOI: `10.3933/applrheol-23-24628
   <https://doi.org/10.3933/applrheol-23-24628>`_.

.. [#ref-WLF1955] Williams, M. L., Landel, R. F. and Ferry, J. D., *The
   temperature dependence of relaxation mechanisms in amorphous polymers
   and other glass-forming liquids*, J. Am. Chem. Soc. **77**\ (14),
   3701–3707 (1955).
   DOI: `10.1021/ja01619a008 <https://doi.org/10.1021/ja01619a008>`_.

.. [#ref-Arrhenius1889] Arrhenius, S., *Über die Reaktionsgeschwindigkeit
   bei der Inversion von Rohrzucker durch Säuren*, Z. Phys. Chem.
   **4**, 226–248 (1889).
   DOI: `10.1515/zpch-1889-0416
   <https://doi.org/10.1515/zpch-1889-0416>`_.
