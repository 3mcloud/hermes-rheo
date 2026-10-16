# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Changed
- `AutomatedMasterCurve` returns a piblin `Measurement` (`MasterCurveMeasurement`):
  a dataset of the shifted data and a dataset of shift factors versus state;
  `details` hold method (mastercurves, Lennon et al. 2023,
  doi:10.1017/dce.2023.3), reference state, shift results and software. The
  `mastercurves` object is available as `.master_curve`; `states`/`hparams`
  are kept. `CompositeMasterCurve` returns a `MasterCurveSet`. New
  `ref_state_idx` argument. `AutomatedMasterCurve._state_to_condition` was
  replaced by the shared `summarize_state` used by `StatesToConditions`.
- `FitModel` now returns a piblin `Measurement` (one-dimensional composite
  dataset with measured and fitted arrays; `details` hold model, parameters,
  goodness of fit, method, references, software). Fit results and `plot()`
  remain available on the transform instance, not the return value.
- `MutationNumber` handles correlated OWCh (TORCh) steps and returns one
  measurement per frequency with time, mutation number, angular frequency
  and temperature.
- Requires Python 3.10–3.13 and `piblin>=0.0.3`.
- `TriosRheoReader` sets per-method default coordinates at read time (for
  example `strain = f(time)` for creep). Pass
  `default_coords="legacy-temperature"` to restore the previous behaviour.
- `FitModel` defaults to `model=None`; pyRheo is optional and imported lazily.
- Tutorial notebooks 1 and 5 renamed without the leading space.

### Added
- `RelaxationSpectrum`: continuous and discrete relaxation spectra from
  `G(t)` or `G'(ω), G''(ω)` with a `domain={"time", "frequency"}` argument.
  The pyReSpect solvers (Shanbhag, MIT) are vendored in `hermes_rheo._vendor`.
- `FrequencyToTime`: relaxation modulus `G(t)` from oscillatory data, following
  the interval-slope reconstruction of i-Rheo-Tempo
  (Ramírez and Tassieri, J. Rheol. 70, 1089 (2026),
  doi:10.1122/8.0001232; code: doi:10.5525/gla.researchdata.2230).
- `OWChirpAnalysis`: the OWChirp / Frequency Chirp FFT moduli extraction,
  split out of `RheoAnalysis`.
- `Kymograph`: two-dimensional maps of any quantity, for example temperature ×
  frequency for TORCh data, with free choice of axes (`x`, `y`, `log_x`,
  `log_y`) and colour range (`vmin`, `vmax`); `to_measurement()` returns a
  piblin `Measurement`.
- `TemperatureRampAnalysis` (`TemperatureRampMeasurement`): Tg from the
  tan δ peak (default), the storage modulus onset and inflection, or both
  (`tg_method`). The tan δ peak detection is robust to several peaks: Tg is
  the most prominent one and secondary relaxations are listed in
  `tan_delta_peaks`. The test frequency is read from the data and reported in
  `details`; `summary()`, `tg_values` and `plot_onset()` report the results.
- `RecoveryRheology`: oscillatory and step-function recovery rheology
  (step strain, creep, start-up shear). Steps are paired by duration, and
  points flagged invalid by RheoCompass are skipped.
- `SplitTemperatureSweep` and the WLF and Arrhenius shift-factor models.
- `FitModel`: classical and fractional viscoelastic models through pyRheo,
  plus a built-in multi-mode Maxwell fit.
- `hermes_rheo.lve_models`: linear viscoelastic models that simulate SAOS,
  creep and stress relaxation and return piblin measurements with full
  metadata (`simulated=True`, model, parameters, grid, references).
  Classical: Maxwell, multi-mode Maxwell, Kelvin–Voigt, Jeffreys.
  Fractional: SpringPot, fractional Maxwell (gel, liquid, general),
  fractional Kelvin–Voigt, fractional Zener. Entry point: `Protocols`.
  Creep and relaxation of the fractional models come from numerical Laplace
  inversion (Talbot) and are cross-checked against pyRheo
  (Bonfanti et al., Phys. Fluids 32, 2020; Garrappa, SIAM J. Numer. Anal. 2015).
- `hermes_rheo.visualization`: unit-aware plotting helpers (van Gurp–Palmen,
  Cole–Cole, master curves, mutation number, Tg from the storage modulus,
  spectra, OWChirp animation) with a shared style.
- `RheoCompassReader` for Anton Paar RheoCompass `.csv` exports.
- `hermes_rheo.read_file`, which accepts `file_reader=` (public piblin does not).
- All readers and transforms are importable from `hermes_rheo`.
- Tutorials 1–13 and the data they use; Sphinx API docs and a MyST tutorial book.
- Test suite, ruff configuration, uv-based development workflow and CI on
  Linux, macOS and Windows.

### Deprecated
- `RheoAnalysis`: use `OWChirpAnalysis`. It remains as a subclass that emits a
  `DeprecationWarning` and will be removed in a future major release.

### Fixed
- `StatesToConditions` and `MutationNumber` keep the active independent
  variable instead of assuming `temperature`.
- `OWChirpAnalysis.parse_value_with_prefix` no longer accepts a literal space
  as a unit prefix.
- `RheoCompassReader` records `source_disk_directory` and `source_file_name`.
- `TriosRheoReader` no longer applies the file metadata and notes twice.

### Migration

```python
# 1.x
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
pipeline = RheoAnalysis() + AutomatedMasterCurve(...)

# 2.0
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
pipeline = OWChirpAnalysis() + AutomatedMasterCurve(...)
```

Reading mixed-method TRIOS files with the old coordinate convention:

```python
TriosRheoReader().data_from_filepath(
    "mixed.txt",
    create_composite_datasets=True,
    default_coords="legacy-temperature",
)
```

Install pyRheo only if you use its models in `FitModel`:

```bash
pip install git+https://github.com/mirandi1/pyRheo.git
```

[2.0.0]: https://github.com/3mcloud/hermes-rheo/compare/v1.3.1...v2.0.0
