# Project / team instructions — rheology analysis with hermes-rheo

Always-on instructions for an AI coding assistant used by a team or
project. Edit the parts in `<angle brackets>`, delete what does not apply,
and load it where your tool expects project-wide instructions (see
`README.md` in this folder).

## Who uses this project

- Scientists and engineers analysing rheometer data (TA Instruments TRIOS,
  Anton Paar RheoCompass) in Python notebooks or scripts.
- Many users are not software developers. Generated code must be short,
  readable, commented where the physics is not obvious, and runnable
  top-to-bottom.

## Approved libraries

- Rheology analysis: **hermes-rheo** (import name `hermes_rheo`, PyPI
  `hermes-rheo`, docs <https://3mcloud.github.io/hermes-rheo/>). Its data
  model comes from **piblin**.
- General: NumPy, SciPy, pandas, matplotlib, openpyxl.
- Do not add other rheology or curve-shifting packages unless the user asks.

## Use hermes-rheo, do not reinvent it

- Before writing any code for rheology data, check whether hermes-rheo
  already provides it. Use the `hermes-rheo` skill (`SKILL.md`) for the
  API, recipes and the task-to-tool table.
- hermes-rheo already covers: reading TRIOS `.txt` and RheoCompass `.csv`
  files, OWChirp / arbitrary-wave FFT analysis, OWChirp signal design,
  time-temperature and time-cure superposition (master curves),
  WLF/Arrhenius shift-factor fits, viscoelastic model fitting, G(t) from
  G′/G″, relaxation spectra, Tg from temperature ramps, mutation number,
  kymographs, recovery rheology, van Gurp-Palmen and Cole-Cole plots, and
  Excel export.
- Never write your own TRIOS/RheoCompass parser, chirp FFT, master-curve
  shifting, WLF fit or Tg detection.
- Never invent classes, functions, arguments or column names. If unsure,
  check with `help(...)`, `inspect.signature(...)`,
  `print(hermes_rheo.__all__)` or `dataset.data_array_names`, and say so.
- If hermes-rheo does not provide something, load the data with hermes-rheo
  anyway, write a small plain NumPy/pandas/matplotlib addition, and tell
  the user which part is custom.

## Installing

- `pip install "hermes-rheo[notebooks]"` (in a notebook: `%pip install ...`,
  then restart the kernel; on Databricks: `dbutils.library.restartPython()`).
- Optional pyRheo models for `FitModel`:
  `pip install git+https://github.com/mirandi1/pyRheo.git`
- <If your team pins a version or uses a private package index, state it
  here, e.g. `pip install hermes-rheo==2.0.0`.>

## Where data and results live

- Raw instrument exports: `<path or storage location of raw files>`
- Results (Excel, CSV, figures): `<path of results folder>`
- Pass file paths (`pathlib.Path`) to `hermes_rheo.read_file`. Files in
  cloud object storage must be downloaded or mounted to a local path first.
- Never modify or delete raw data. Write outputs to the results folder only.

## Conventions for generated code

- Read files with `read_file(path, file_reader=TriosRheoReader or
  RheoCompassReader, create_composite_datasets=True)`.
- Refer to data columns by their exact lowercase names (`"storage modulus"`,
  `"angular frequency"`, `"tan(delta)"`), never by position.
- Take units from the data (`dataset.data_array_units`); never assume Pa.
- Label every plot axis with quantity and unit. Use log axes for moduli
  versus frequency or time.
- Use the hermes-rheo exporters (`.to_excel(...)`) and plot helpers in
  `hermes_rheo.visualization` before writing custom ones.
- Set `np.random.seed(...)` before model fits so results are reproducible.
- Keep secrets, tokens and personal data out of code and notebooks.

## Guardrails

- Don't change instrument data values (no smoothing, outlier removal or
  unit conversion) unless the user asks, and state it clearly when you do.
- When a result looks physically implausible (negative moduli, Tg outside
  the measured range, shift factors that are not monotonic), point it out
  instead of hiding it.
- If a TRIOS file fails to parse, ask how it was exported: it must be
  *Export to LIMS* (TRIOS 5 and older) or *Export to .txt (Full)* (TRIOS 6 and newer).
