# My instructions

Example personal instructions for a hermes-rheo user, for any AI coding
assistant. Keep what fits you, delete the rest, and fill in the
`<angle brackets>`.

## About me and my work

- I am a <rheologist / materials scientist / formulation chemist> working
  on <polymers, adhesives, gels, ...>.
- I analyse rheometer data from <TA Instruments DHR / ARES-G2, Anton Paar
  MCR> with the **hermes-rheo** Python library in <Jupyter / VS Code /
  Databricks / Colab> notebooks.
- I am <comfortable with / new to> Python. Explain code
  <briefly / step by step>.

## How I want you to work

- Always use hermes-rheo for reading rheometer files and for the analysis
  (master curves, OWChirp, Tg, model fits, spectra, kymographs, ...).
  Use the `hermes-rheo` skill when I work with rheology data.
- Don't reinvent functions hermes-rheo already provides, and never guess
  API names. If you are not sure something exists, check it with
  `help(...)` first.
- Give me complete cells or scripts I can run top to bottom. Put imports
  and paths at the top.
- Ask me before you assume something that changes the result: reference
  temperature, which steps or temperatures to use, units, model choice.
- When you finish an analysis, give a short plain-language summary of the
  result (e.g. Tg value, WLF constants, fit quality).

## My data

- Raw files: `<folder with my instrument exports>`
- Save results to: `<my results folder>`
- My usual reference temperature for master curves is <25> °C.

## Plots and exports

- Use `hermes_rheo.visualization.use_science_style()` for figures.
- <Use the `viridis` colour map for temperature series.>
- Axis labels with units, e.g. `G′ [Pa]`, `ω [rad/s]`; log axes for moduli.
- Save figures as PNG at 300 dpi next to the Excel results.
- Export tables to Excel with one sheet per sample or step, units in the
  column headers.

## Response style

- Be concise. Lead with the answer or the code, then a short explanation.
- Use SI units and the usual rheology symbols (G′, G″, tan δ, ω, a_T).
