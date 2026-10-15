# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Added
- `RelaxationSpectrum`: continuous and discrete relaxation spectra from
  `G(t)` or `G'(ω), G''(ω)` with a `domain={"time", "frequency"}` argument.
  The pyReSpect solvers (Shanbhag, MIT) are vendored in `hermes_rheo._vendor`.
- `FrequencyToTime`: relaxation modulus `G(t)` from oscillatory data, following
  the interval-slope reconstruction of i-Rheo Tempo.
- `OWChirpAnalysis`: the OWChirp / Frequency Chirp FFT moduli extraction,
  split out of `RheoAnalysis`.
- `Kymograph`: temperature × frequency maps for TORCh data and other
  two-axis quantities.
- `SplitTemperatureSweep`, `TemperatureRampAnalysis`, `RecoveryRheology` and
  the WLF and Arrhenius shift-factor models.
- `FitModel`: classical and fractional viscoelastic models through pyRheo,
  plus a built-in multi-mode Maxwell fit.
- `hermes_rheo.synthetic`: `SyntheticRheometry` and Maxwell, multi-mode
  Maxwell, Kelvin–Voigt and Jeffreys models for generating test data.
- `hermes_rheo.visualization`: unit-aware plotting helpers (van Gurp–Palmen,
  Cole–Cole, spectra, OWChirp animation).
- `RheoCompassReader` for Anton Paar RheoCompass `.csv` exports.
- `hermes_rheo.read_file`, which accepts `file_reader=` (public piblin does not).
- All readers and transforms are importable from `hermes_rheo`.
- Tutorials 1–13 and the data they use; Sphinx API docs and a MyST tutorial book.
- Test suite, ruff configuration, uv-based development workflow and CI on
  Linux, macOS and Windows.

### Changed
- Requires Python 3.10–3.13 and `piblin>=0.0.3`.
- `TriosRheoReader` sets per-method default coordinates at read time (for
  example `strain = f(time)` for creep). Pass
  `default_coords="legacy-temperature"` to restore the previous behaviour.
- `FitModel` defaults to `model=None`; pyRheo is optional and imported lazily.
- Tutorial notebooks 1 and 5 renamed without the leading space.

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
