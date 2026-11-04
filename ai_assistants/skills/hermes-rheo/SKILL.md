---
name: hermes-rheo
description: Use hermes-rheo (import name `hermes_rheo`) to analyse rheology data in Python notebooks or scripts (Jupyter, VS Code, Colab, Databricks, ...) — read TA TRIOS .txt / Anton Paar RheoCompass .csv exports, and do OWChirp (optimally windowed chirp) analysis, time-temperature or time-cure superposition master curves, WLF/Arrhenius shift factors, model fitting (Maxwell, fractional, multi-mode), G(t) from G'/G'', relaxation spectra, glass transition (Tg) from temperature ramps, mutation number, kymographs, recovery rheology, van Gurp-Palmen / Cole-Cole plots, and Excel export. Use whenever a user works with rheometer data, viscoelastic moduli, or asks for rheology plots/exports, BEFORE writing any custom parsing, FFT, fitting or shifting code.
---

# hermes-rheo — analysis skill for notebook users

`hermes-rheo` is a Python library for rheology data analysis. It reads
rheometer exports into structured data objects and provides tested,
published analysis methods. Your job as the coding agent is to **compose
these building blocks**, not to re-implement them.

- Import name: `hermes_rheo` (distribution name `hermes-rheo` on PyPI).
- Built on [`piblin`](https://github.com/3mcloud/piblin): every reader
  returns `piblin` collections and every transform consumes/returns them.
- Python 3.10–3.13.
- Docs: <https://3mcloud.github.io/hermes-rheo/> ·
  Tutorials: <https://3mcloud.github.io/hermes-rheo/tutorials/>
  (source notebooks in `tutorial_notebooks/` of
  <https://github.com/3mcloud/hermes-rheo>).

## 1. Rules for the agent (read first)

1. **Search hermes before writing code.** If the user asks to read a
   rheometer file, compute moduli from a chirp, shift curves, fit a model,
   find Tg, etc., use the matching tool from the task table in section 5.
   Writing your own FFT, TTS shifting, WLF fit, Tg detection or TRIOS
   parser when hermes already does it is a bug, not a feature.
2. **Never invent API.** Only use classes, functions, arguments and
   attributes listed in this file. If you need something else, verify it
   first in the user's environment:
   `help(hermes_rheo.SomeClass)`, `inspect.signature(...)`,
   `dataset.data_array_names`, `measurement.details.keys()`.
   If it does not exist, say so and write plain NumPy/pandas/matplotlib
   code instead — do not guess names.
3. **Column names are lowercase strings and must match exactly**
   (`"storage modulus"`, `"angular frequency"`, `"temperature"`,
   `"tan(delta)"`). Print `dataset.data_array_names` before using a name.
4. **Always pass `create_composite_datasets=True`** to `read_file` for
   TRIOS / RheoCompass files (all examples below do).
5. **Read units from the data, never assume them.** Moduli can be Pa, kPa
   or MPa depending on the instrument export; use
   `dataset.data_array_units` or `hermes_rheo.visualization.get_unit`.
6. **Prefer the result objects' own exporters** (`.to_excel(...)`) and the
   plot helpers in `hermes_rheo.visualization` before hand-rolling.
7. Keep the user's raw data untouched: write outputs to a separate folder
   (e.g. `results/` next to the data).

## 2. Install and check

hermes-rheo is on PyPI and is pure Python (NumPy/SciPy/pandas/matplotlib),
so it runs anywhere Python 3.10–3.13 runs: a laptop, JupyterLab, VS Code,
Google Colab, Databricks, cloud VMs or HPC clusters.

```bash
pip install hermes-rheo                 # or: uv pip install hermes-rheo
pip install "hermes-rheo[notebooks]"    # + SciencePlots style and video export
```

In a notebook cell use `%pip install "hermes-rheo[notebooks]"` (installs
into the running kernel), then restart the kernel if the package was
already imported.

```python
import hermes_rheo
print(hermes_rheo.__version__)
```

If the machine has no internet access, ask the user how Python packages
are provided (private mirror, wheel file) — don't guess.

### Optional pyRheo models

`FitModel` with `model_name="Maxwell"`, `"FractionalZener"`, … needs
[pyRheo](https://github.com/mirandi1/pyRheo) (GPL-3.0, not on PyPI):

```bash
pip install git+https://github.com/mirandi1/pyRheo.git
```

Without it only the built-in multi-mode Maxwell
(`FitModel(model="custom", model_name="MultiModeMaxwell")`) works.

### Files, paths and environments

hermes-rheo reads ordinary files with Python file I/O: pass a file path
(`str` or `pathlib.Path`). Ask the user where their data is instead of
inventing a path. Write results to a separate folder next to the data.

Notes for specific environments (use only the one that applies):

- **Local / Jupyter / VS Code:** relative paths such as `data/run1.txt`
  are resolved from the notebook's working directory.
- **Google Colab:** upload the files or mount Drive
  (`from google.colab import drive; drive.mount("/content/drive")`), then
  use `/content/drive/MyDrive/...` paths.
- **Databricks:** after `%pip install ...` run
  `dbutils.library.restartPython()` in its own cell. Use Unity Catalog
  volume paths (`/Volumes/<catalog>/<schema>/<volume>/...`) or workspace
  files; legacy DBFS needs the FUSE form `/dbfs/...`, not `dbfs:/...`.
  hermes works on NumPy/pandas in driver memory — don't load instrument
  files with Spark. Only if the user wants a table:
  `spark.createDataFrame(df).write.saveAsTable("catalog.schema.table")`.
- **Cloud object storage (S3, Azure Blob, GCS):** download the file to a
  local/temporary path first (or use a mounted path); `read_file` does not
  take `s3://` URLs.
- **Headless runs (scripts, jobs, HPC):** set `matplotlib.use("Agg")` and
  save figures with `fig.savefig(...)` instead of showing them.
- Loop over many files with `sorted(Path(folder).glob("*.txt"))`.

## 3. Mental model: how the data is organised

```
read_file(...) ─► ExperimentSet / MeasurementSet        (one file)
                     .measurements  ─► list[Measurement] (one per TRIOS step / RheoCompass interval)
                        .conditions  dict  e.g. {"method": "Step name\tFrequency sweep (150.0 °C)"}
                        .details     dict  e.g. sample_name, instrument_serial_number, run_date, ...
                        .datasets    list  ─► datasets[0] is a OneDimensionalCompositeDataset
                                             .data_array_names   ["temperature", "angular frequency", "storage modulus", ...]
                                             .data_array_units   ["°C", "rad/s", "MPa", ...]
                                             .data_arrays        list of NumPy arrays (same order)
                                             active pair: .x_name/.x_values/.x_unit, .y_name/.y_values/.y_unit
```

- A **composite dataset** holds *all* columns of a step. One pair is
  "active" (used by `.visualize()`, `.x_values`, `.y_values`, `to_csv`).
  Readers pick a sensible active pair per test type (frequency sweep →
  `storage modulus` vs `angular frequency`; creep → `strain` vs `time`).
  Change it with `dataset.switch_coordinates("angular frequency", "loss modulus")`.
- Temperature is usually a **column** (`"temperature"`), not a condition.
  Use `StatesToConditions` (section 5) if you need it as a condition.
- After OWChirp analysis, each measurement has **two** datasets:
  `datasets[0]` = frequency domain (moduli), `datasets[1]` = raw time
  domain (strain, stress, …).

Standard helper to get columns by name (use this, don't index blindly):

```python
def columns(dataset):
    """Return {name: array} and {name: unit} for a composite dataset."""
    return (dict(zip(dataset.data_array_names, dataset.data_arrays)),
            dict(zip(dataset.data_array_names, dataset.data_array_units)))

col, unit = columns(experiment.measurements[0].datasets[0])
col["storage modulus"], unit["storage modulus"]
```

## 4. Reading files

```python
from pathlib import Path
from hermes_rheo import read_file, TriosRheoReader, RheoCompassReader

experiment = read_file(Path("data") / "frequency_sweep_tts.txt",
                       file_reader=TriosRheoReader,
                       create_composite_datasets=True)
print(experiment)                         # overview
for m in experiment.measurements:
    ds = m.datasets[0]
    print(m.conditions["method"], "|", ds.dependent_variable_name, "vs", ds.independent_variable_name)
```

| Instrument / file | Reader (import from `hermes_rheo`) | How to read |
|---|---|---|
| TA Instruments TRIOS `.txt` | `TriosRheoReader` | `read_file(path, file_reader=TriosRheoReader, create_composite_datasets=True)` |
| Anton Paar RheoCompass `.csv` | `RheoCompassReader` | `read_file(path, file_reader=RheoCompassReader, create_composite_datasets=True)` |

Important details:

- **TRIOS export format:** use *Export to LIMS* (TRIOS ≤ 5) or
  *Export to .txt (Full)* (TRIOS ≥ 6). The plain *Export to .txt* drops
  step metadata and will not parse. If parsing fails, ask the user how the
  file was exported.
- **Mixed-method files** (e.g. amplitude sweep + frequency sweep + flow
  ramp in one file): pass `to_experiments=False` → a `MeasurementSet` where
  each step keeps its own natural axes. If you need an `ExperimentSet`
  anyway, pass `default_coords="uniform"` to the TRIOS reader.
- **Keep only some steps** (e.g. one method out of several): read the
  file, then select measurements by `m.conditions["method"]`, e.g.
  `[m for m in exp.measurements if "Frequency sweep" in m.conditions["method"]]`.
- `read_file` without `file_reader` picks `TriosRheoReader` for `.txt` and
  `RheoCompassReader` for `.csv`; passing it explicitly is clearer.
- Other instruments/formats are not supported by built-in readers: load
  them with pandas and use the raw-array routes in sections 6.2 and 6.6.

## 5. What do you want to do → which tool

All analysis classes are **transforms**: build with settings, then call on
data. Chain them with `+`.

```python
result = Transform(settings)(data)                     # one step
result = (TransformA() + TransformB(settings))(data)   # pipeline
```

| Task | Tool | Input | Returns |
|---|---|---|---|
| G′, G″ from OWChirp / Arbitrary Wave / Frequency Chirp steps | `OWChirpAnalysis()` | TRIOS chirp file | `ExperimentSet`; per measurement `datasets[0]` = moduli vs `angular frequency`, `datasets[1]` = raw waves |
| Design an OWChirp signal for the TRIOS Arbitrary Wave macro | `OWChirpGeneration(...)` | design parameters | object with `.print_parameters()`, `.to_txt(folder)`, `.visualize(...)` |
| TTS / time-cure master curve (one quantity) | `AutomatedMasterCurve(state=..., x=..., y=...)` | set of sweeps at different T or times | `MasterCurveMeasurement` |
| Master curves for several quantities at once (G′, G″, tan δ) | `CompositeMasterCurve(n_curves=..., y_quantities=[...])` | same | `MasterCurveSet`; `.get_mastercurve("loss modulus")` |
| WLF / Arrhenius fit of shift factors | `ShiftFactorModel(model="WLF" or "Arrhenius")` (`hermes_rheo.transforms`) | a `MasterCurveMeasurement` | fitted model: `.print_parameters()`, `.predict(T)` |
| Split one long multi-temperature sweep into isotherms | `SplitTemperatureSweep(n_steps=None, temp_step_interval=None, temp_tolerance=1.0)` | sweep(s) | one measurement per temperature |
| Fit a viscoelastic model to G′/G″ (or G(t)) | `FitModel(model_name="Maxwell" / "FractionalZener" / ...)` or `FitModel(model="custom", model_name="MultiModeMaxwell", n_modes=4)` | one `Measurement` (or master curve) | `Measurement` with extra columns `fitted storage modulus`, `fitted loss modulus`; `details["parameters"]`, `details["goodness_of_fit"]` |
| Relaxation modulus G(t) from G′(ω), G″(ω) | `FrequencyToTime()` | `Measurement` with `angular frequency`, `storage modulus`, `loss modulus` | `Measurement` with `time`, `G_from_eta1`, `G_from_eta2` |
| Continuous + discrete relaxation spectrum H(τ) | `RelaxationSpectrum(domain="frequency")` or `RelaxationSpectrum(domain="time", relaxation_name=...)` | moduli or G(t) | call it, then read `transform.result` (`.tau`, `.g`, `.s`, `.H`) |
| Tg from a temperature ramp (tan δ peak / G′ onset) | `TemperatureRampAnalysis(steps_to_merge=["Temperature Ramp"], tg_method="tan_delta" / "storage_modulus" / "both")` | TRIOS ramp file | `TemperatureRampMeasurement`: `.tg`, `.tg_values`, `.summary()`, `.plot_onset()`, `.to_excel(path)` |
| Mutation number (how fast the material changes during a chirp) | `MutationNumber(dependent_variable="complex modulus")` | OWChirp result or TORCh file | `MutationNumberMeasurementSet`: `.visualize(...)`; per measurement `details["mutation_number_max"]` |
| 2-D map of a property vs temperature × frequency | `Kymograph("tan(delta)", x="temperature", y="angular frequency")` | TORCh / multi-T data | `KymographResult`: `.plot()`, `.plot_slice_at_x([...])`, `.slice_at_y(...)`, `.to_excel(path)` |
| Recoverable vs unrecoverable strain (oscillatory, step strain, creep, start-up) | `RecoveryRheology(method="oscillatory", omega=...)` or `RecoveryRheology(method="step")` | RheoCompass recovery file | call, then `.result` → `.plot_lissajous()`, `.plot_step_results()`, `.plot_transient()`, `.to_excel(path)` |
| Put a measured state (e.g. mean temperature) into `conditions` | `StatesToConditions(state="temperature", method="average")` | measurements | same objects with `conditions["temperature"]` |
| Simulate ideal SAOS / creep / relaxation data (test, teach, sanity-check) | `hermes_rheo.lve_models` (`MaxwellModel`, `MultiModeMaxwellModel`, `FractionalZenerModel`, …) + `Protocols.saos/creep/stress_relaxation` | model + x-grid | `Measurement` usable by all transforms |
| van Gurp-Palmen / Cole-Cole plots | `plot_vgp`, `plot_cole_cole` (`hermes_rheo.visualization`) | any set of sweeps | `(fig, ax)` |

`RheoAnalysis` is a **deprecated** alias of `OWChirpAnalysis`; always
write `OWChirpAnalysis` in new code.

## 6. Recipes (tested patterns)

Common header for notebooks:

```python
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hermes_rheo import read_file, TriosRheoReader, RheoCompassReader
from hermes_rheo.visualization import use_science_style, get_unit, axis_label

use_science_style()                 # optional, nicer with the [notebooks] extra
warnings.filterwarnings("ignore")   # optimiser convergence notices from master curves / fits
DATA = Path("data")                 # ask the user where their files are
RESULTS = DATA.parent / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
```

### 6.1 Plot any step quickly / switch what is plotted

```python
exp = read_file(DATA / "frequency_sweep.txt", file_reader=TriosRheoReader,
                create_composite_datasets=True)
ds = exp.measurements[0].datasets[0]
print(ds.data_array_names)

fig, ax = ds.visualize(marker="o", linewidth=1)       # active pair
ds.switch_coordinates("angular frequency", "complex viscosity")
fig, ax = ds.visualize(marker="s")
ax.set_xscale("log"); ax.set_yscale("log")

# overlay several measurements on one axes
fig, ax = plt.subplots()
for m in exp.measurements:
    d = m.datasets[0]
    d.switch_coordinates("angular frequency", "storage modulus")
    d.visualize(axes=ax, marker="o", linestyle="none")
ax.set_xscale("log"); ax.set_yscale("log")
```

### 6.2 OWChirp: moduli from chirp steps + standard figures

```python
from hermes_rheo import OWChirpAnalysis
from hermes_rheo.visualization import OWChirpVisualizer

raw = read_file(DATA / "owchirp.txt", file_reader=TriosRheoReader, create_composite_datasets=True)
chirp = OWChirpAnalysis()(raw)          # options: owchirp_waiting_time="before_signal", filter_criterion="endpoints"

viz = OWChirpVisualizer(chirp)
viz.plot_waves(steps=[0, 5, 10])        # strain/stress waves of selected chirps
viz.plot_fourier(step=0)                # FFT + G', G'' of one chirp
# viz.to_video("results", fps=2)        # movie over all chirps ([notebooks] extra)

moduli = chirp.measurements[0].datasets[0]      # frequency domain
waves = chirp.measurements[0].datasets[1]       # time domain (strain, stress, ...)
```

OWChirpAnalysis detects strain- vs stress-controlled instruments (DHR)
and Arbitrary Wave vs Frequency Chirp steps automatically from the file
metadata. Do **not** re-implement the FFT.

If the user only has **raw arrays** (e.g. time / strain / stress columns in
Excel) and no TRIOS file, still reuse the hermes FFT routine:

```python
import pandas as pd
from hermes_rheo import OWChirpAnalysis

df = pd.read_excel("chirp.xlsx")
strain = df["Shear Strain"].to_numpy()
stress = df["Shear Stress"].to_numpy()
fs = 1.0 / np.median(np.diff(df["Time"].to_numpy()))     # sampling rate, points/s
w1, w2 = 1.0, 100.0                                       # chirp band, rad/s (from the chirp design)
omega, strain_ft, stress_ft, Gp, Gpp, Gstar = OWChirpAnalysis.moduli_owchirp_strain_controlled(
    strain, stress, fs, w1, w2)
```

Ask the user for `w1`, `w2` and the waiting time if they are not in the
file. Baseline-correct first if needed with
`OWChirpAnalysis().filter_signal(t_w, time, signal, method="tw")` (strain)
or `method="chirp"` (stress).

### 6.3 Time-temperature superposition + WLF

```python
from hermes_rheo import AutomatedMasterCurve
from hermes_rheo.transforms import ShiftFactorModel
from hermes_rheo.visualization import plot_master_curve, plot_shift_factors

sweeps = read_file(DATA / "frequency_sweep_tts.txt", file_reader=TriosRheoReader,
                   create_composite_datasets=True)
mc = AutomatedMasterCurve(state="temperature", state_mode="average",
                          x="angular frequency", y="storage modulus")(sweeps)
# OWChirp data: (OWChirpAnalysis() + AutomatedMasterCurve(...))(raw)

T_REF = 20
mc.change_ref(min(mc.states, key=lambda T: abs(T - T_REF)))   # reference must be one of mc.states
fig, axes = plot_master_curve(mc)

wlf = ShiftFactorModel(model="WLF")(mc)          # or model="Arrhenius"
wlf.print_parameters()
fig, ax = plot_shift_factors(wlf)

curve = mc.datasets[0]      # shifted master curve: x = a_T·ω, y = quantity
shifts = mc.datasets[1]     # temperature vs horizontal shift factor
mc.to_excel(RESULTS / "master_curve.xlsx")
```

- Several quantities sharing the same shifts:
  `CompositeMasterCurve(n_curves=3, y_quantities=["storage modulus", "loss modulus", "tan(delta)"], state="temperature", x="angular frequency")(sweeps)`;
  loop `for m in result.measurements: m.change_ref(...)`; get one with
  `result.get_mastercurve("loss modulus")`.
- `vertical_shift=True` only when curves are also offset vertically
  (filled systems, gels); default `False`.
- One file with many temperatures in a single sweep step → put
  `SplitTemperatureSweep()` before the master curve.

### 6.4 Time-cure superposition (curing / gelation)

```python
mc = (OWChirpAnalysis() + AutomatedMasterCurve(
    state="time", state_mode="first point",
    x="angular frequency", y="complex modulus",
    vertical_shift=False, reverse_data=True))(raw)
mc.change_ref(mc.states[-1])
plot_master_curve(mc, cmap="viridis", lowest_on_top=False)
shift_uncertainty = mc.details["horizontal_shift_uncertainties"]
```

### 6.5 Model fitting

```python
from hermes_rheo import FitModel
from hermes_rheo.visualization import plot_model_fit

np.random.seed(0)                                  # pyRheo uses random initial guesses
sweep = exp.measurements[0]                        # one Measurement with G' and G''
fit = FitModel(model_name="FractionalZener")       # pyRheo models: "Maxwell", "FractionalMaxwell", "FractionalKelvinVoigt", ...
fitted = fit(sweep)
print(fitted.details["parameters"], fitted.details["goodness_of_fit"])
fig, ax = plot_model_fit(fit)                      # pass the FitModel transform, not `fitted`

prony = FitModel(model="custom", model_name="MultiModeMaxwell", n_modes=6)(sweep)   # no pyRheo needed
```

- `fit_to="both"` (default), `"storage"` or `"loss"` chooses which
  modulus is fitted.
- Fit a master curve: `(... + CompositeMasterCurve(...) + FitModel(...))(data)`.
- Fit G(t): `FitModel(domain="relaxation", model_name="FractionalMaxwell")`
  on the output of `FrequencyToTime` (default `relaxation_name="G_from_eta1"`),
  or on measured relaxation data with `relaxation_name="relaxation modulus"`.

### 6.6 G(t) and relaxation spectrum

```python
from hermes_rheo import FrequencyToTime, RelaxationSpectrum
from hermes_rheo.visualization import plot_relaxation_spectrum

gt = FrequencyToTime()(sweep)              # columns: time, G_from_eta1, G_from_eta2
t, g1, g2 = gt.datasets[0].data_arrays

rs = RelaxationSpectrum(domain="frequency")    # needs angular frequency, storage modulus, loss modulus
rs(sweep)
spec = rs.result                               # spec.tau, spec.g (discrete modes); spec.s, spec.H (continuous)
fig, ax = plot_relaxation_spectrum(spec)
```

Both need G′ and G″ on the **same** frequency grid in one dataset. A
`CompositeMasterCurve` gives them on different grids — interpolate onto one
grid and build a `Measurement` as in tutorials 12 and 13:

```python
from piblin.data import Measurement, OneDimensionalCompositeDataset

master = Measurement(datasets=[OneDimensionalCompositeDataset(
    data_arrays=[w, gp, gpp],
    data_array_names=["angular frequency", "storage modulus", "loss modulus"],
    data_array_units=["rad/s", "Pa", "Pa"],
    default_independent_name="angular frequency",
    default_dependent_name="storage modulus")])
```

### 6.7 Glass transition from a temperature ramp

```python
from hermes_rheo import TemperatureRampAnalysis
from hermes_rheo.visualization import plot_tg_onset

ramp_file = read_file(DATA / "temp_ramp.txt", file_reader=TriosRheoReader,
                      create_composite_datasets=True)
ramp = TemperatureRampAnalysis(
    steps_to_merge=["Temperature Ramp"],               # step name(s) in the TRIOS file
    use_temps=[-50, 0, 25, 100],                       # optional: report properties at these T
    properties=["storage modulus", "loss modulus", "tan(delta)"],
    tg_method="both",                                  # "tan_delta" (default) | "storage_modulus" | "both"
)(ramp_file)
print(ramp.summary())
print(ramp.tg, ramp.tg_values, ramp.details["tg_unit"])
fig, axes = plot_tg_onset(ramp)                        # needs tg_method "storage_modulus" or "both"
ramp.to_excel(RESULTS / "tg_results.xlsx")
```

### 6.8 Mutation number, kymographs, VGP / Cole-Cole

```python
from hermes_rheo import MutationNumber, Kymograph
from hermes_rheo.visualization import plot_vgp, plot_cole_cole, plot_mutation_number

mu = (OWChirpAnalysis() + MutationNumber(dependent_variable="complex modulus"))(raw)
fig, axes = plot_mutation_number(mu, show_all_frequencies=True)

torch = read_file(DATA / "torch.txt", file_reader=TriosRheoReader, create_composite_datasets=True)
k = Kymograph("tan(delta)", x="temperature", y="angular frequency")(torch)
fig, ax = k.plot(colormap="turbo")
k.plot_slice_at_x([0, 25, 80])
k.to_excel(RESULTS / "kymograph.xlsx")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_vgp(sweeps, ax=axes[0])
plot_cole_cole(sweeps, ax=axes[1])
```

### 6.9 Recovery rheology (RheoCompass)

```python
from hermes_rheo import RecoveryRheology

osc = read_file(DATA / "recovery_oscillatory.csv", file_reader=RheoCompassReader,
                create_composite_datasets=True)
rec = RecoveryRheology(method="oscillatory", omega=0.316)(osc).result
rec.plot_lissajous()
rec.to_excel(RESULTS / "recovery.xlsx")

creep = read_file(DATA / "creep_recovery.csv", file_reader=RheoCompassReader,
                  create_composite_datasets=True)
step = RecoveryRheology(method="step")(creep).result
step.plot_step_results()
step.plot_transient(logx=False)
```

Batch amplitude series from a folder:
`from hermes_rheo.transforms.recovery_rheology import batch_oscillatory_recovery, plot_decomposed_moduli`.

### 6.10 Design an OWChirp signal

```python
from hermes_rheo import OWChirpGeneration

sig = OWChirpGeneration(output_file_name="owchirp_parameters.txt", waiting_time=1.0,
                        signal_duration=14.0, signal_amplitude=8.5e-5,
                        initial_frequency=0.3, final_frequency=60.0,   # rad/s
                        tapering_parameter=0.1, initial_phase_shift=0.0)
sig.print_parameters()
sig.to_txt(RESULTS)                       # zone equations for the TRIOS Arbitrary Wave macro
sig.visualize(display_r0=True, filepath=RESULTS)
```

## 7. Exporting results

| What | How |
|---|---|
| Master curve | `mc.to_excel("file.xlsx")` |
| Tg / temperature ramp | `ramp.to_excel("file.xlsx")` |
| Kymograph | `k.to_excel("file.xlsx")` |
| Recovery | `rec.to_excel("file.xlsx")` |
| Any dataset, **all columns** | `dataset_to_frame(ds)` below, then `.to_excel` / `.to_csv` |
| Fit parameters | `pd.Series(fitted.details["parameters"])` |
| Figures | `fig.savefig("results/name.png", dpi=300, bbox_inches="tight")`, or `save="path.png"` on `plot_*` helpers |

```python
import pandas as pd

def dataset_to_frame(ds):
    """All columns of a composite dataset, with units in the headers."""
    return pd.DataFrame({f"{n} [{u}]": a for n, u, a in
                         zip(ds.data_array_names, ds.data_array_units, ds.data_arrays)})

with pd.ExcelWriter(RESULTS / "all_steps.xlsx") as xl:
    for i, m in enumerate(exp.measurements):
        dataset_to_frame(m.datasets[0]).to_excel(xl, sheet_name=f"step_{i + 1}", index=False)
```

**Pitfall:** `dataset.to_csv(path)` and `dataset.to_pandas()` export only
the currently **active x/y pair** (and `to_csv` writes no header). Use
`dataset_to_frame` above for complete tables.

## 8. Plotting helpers (`hermes_rheo.visualization`)

All return `(fig, ax)` / `(fig, axes)`; most accept `ax=` so they can be
combined with plain matplotlib.

| Function | Input |
|---|---|
| `plot_master_curve(mc, cmap="coolwarm", show_gp=True, save=None)` | `AutomatedMasterCurve` result or one entry of a `CompositeMasterCurve` result |
| `plot_shift_factors(shift_model, ax=None, ...)` | fitted `ShiftFactorModel` |
| `plot_model_fit(fit_model, show_residuals=False, ...)` | the **applied `FitModel` transform** |
| `plot_mutation_number(mu, show_all_frequencies=False, threshold=0.15, ...)` | `MutationNumber` result |
| `plot_tg_onset(ramp, xlim=None, ...)` | `TemperatureRampAnalysis` result (`tg_method` "storage_modulus"/"both") |
| `plot_relaxation_spectrum(result, ax=None, ...)`, `plot_spectrum_diagnostics(result)` | `RelaxationSpectrum(...).result` |
| `plot_vgp(data, ax=None)`, `plot_cole_cole(data, ax=None)` | `ExperimentSet` / `MeasurementSet` / `Measurement` |
| `OWChirpVisualizer(chirp).plot_waves / plot_fourier / save_frames / to_video` | `OWChirpAnalysis` result |
| `use_science_style()`, `reset_style()` | global matplotlib style |
| `get_unit(ds, "storage modulus")`, `axis_label(ds, "temperature", text="T")` | unit-aware axis labels |
| `dataset.visualize(axes=ax, **matplotlib_kwargs)` | any composite dataset (active pair) |

## 9. Common pitfalls

- **`KeyError` / "not found" for a column** → print `ds.data_array_names`;
  names are lowercase and exact (`"tan(delta)"`, not `"tan delta"`).
  RheoCompass exports may call it `"loss factor"`.
- **Units differ between files** (Pa vs MPa) → convert with
  `ds.change_dependent_variable_data_unit(factor, "Pa")` or scale arrays
  using `data_array_units`; never hard-code.
- **`InconsistentMeasurementsError`** reading a mixed-method TRIOS file →
  `to_experiments=False` or `default_coords="uniform"`.
- **`change_ref` value** must be one of `mc.states` (pick the nearest).
- **Fit results vary run to run** → `np.random.seed(...)` before
  `FitModel`; increase `num_initial_guesses`.
- **Long prints from optimisers** → wrap the call in
  `contextlib.redirect_stdout(io.StringIO())`.
- **Index-based selection** (`measurements[4::6]`) depends on file order;
  prefer selecting by the `temperature` column or `conditions`.
- **Chirp moduli look wrong at the band edges** → check sampling rate,
  tapering (`r ≥ 0.05`) and the waiting-time window
  (`OWChirpAnalysis(owchirp_waiting_time=...)`) before changing code.

## 10. When hermes does not have it

Some things are genuinely not provided (e.g. spectrograms of the raw chirp,
custom statistics across samples, bespoke report layouts). Then:

1. Still use hermes to **load** the data and pull arrays by name
   (`columns(ds)` / `dataset_to_frame(ds)`).
2. Write small, plain NumPy / SciPy / pandas / matplotlib code on top.
3. Tell the user which part is custom so they can review it.

Discover what exists in the installed version:

```python
import hermes_rheo, hermes_rheo.transforms, hermes_rheo.visualization, hermes_rheo.lve_models
print(hermes_rheo.__all__)
print(hermes_rheo.transforms.__all__)
print(hermes_rheo.visualization.__all__)
help(hermes_rheo.AutomatedMasterCurve)
```

## 11. References for the methods

- OWChirp: Geri et al., *Phys. Rev. X* 8, 041042 (2018); Perego et al.,
  *Rheol. Acta* (2025).
- Automated master curves: Lennon, McKinley & Swan, *Data-Centric Eng.* 4,
  e13 (2023).
- Mutation number: Mours & Winter, *Rheol. Acta* 33, 385 (1994).
- G(t) from G*(ω): Ramírez & Tassieri, i-Rheo-Tempo, *J. Rheol.* 70, 1089
  (2026).
- Relaxation spectra (pyReSpect): Takeh & Shanbhag, *Appl. Rheol.* 23,
  24628 (2013); Shanbhag, *Macromol. Theory Simul.* 28, 1900005 (2019).
- Each class docstring lists its exact references (`help(...)`).
