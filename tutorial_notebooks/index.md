# hermes-rheo — Tutorials

Interactive tutorials for the [`hermes-rheo`](https://github.com/3mcloud/hermes-rheo)
package.  Each tutorial is a runnable Jupyter notebook that
demonstrates one reader or transform in isolation.

:::{note}
The tutorials use [SciencePlots](https://github.com/garrettj403/SciencePlots)
for publication-quality figures.  To run and visualise the notebooks in
your own environment, install the notebook extra:

```bash
pip install "hermes-rheo[notebooks]"
```
:::

## What's covered

| # | Tutorial                                                                                                                      | Topic                                                             |
| - | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1 | [Reading and visualising common rheological datasets](<./1_read%20and%20visualize%20common%20rheological%20datasets.ipynb>) | TA TRIOS + Anton Paar RheoCompass readers                         |
| 2 | [Designing an OWChirp waveform](<./2_owchirp%20wave%20design.ipynb>)                                                           | `OWChirpGeneration` — waveform export for TRIOS Arbitrary Wave |
| 3 | [Reading and visualising OWChirp experiments](<./3_read%20and%20visualize%20data%20from%20OWChirp%20experiments.ipynb>)        | `OWChirpAnalysis` — FFT-based moduli extraction                |
| 4 | [Time–Temperature Superposition (TTS)](<./4_time%20temperature%20superposition.ipynb>)                                        | `AutomatedMasterCurve`                                          |
| 5 | [Time–Cure Superposition (TCS)](<./5_time%20curing%20superposition.ipynb>)                                                 | `CompositeMasterCurve` — cure-time shifting                    |
| 6 | [Mutation Number](<./6_mutation%20number%20calculation.ipynb>)                                                                 | `MutationNumber` — time-resolved mutation index                |
| 7 | [Van Gurp–Palmen and Cole–Cole plots](./7_visualization.ipynb)                                                               | Visualisation helpers                                             |
| 8 | [Temperature Ramp Analysis](<./8_temperature%20ramp%20analysis.ipynb>)                                                         | `TemperatureRampAnalysis` — Tg + interpolated properties       |
| 9 | [Kymographs from TORCh experiments](<./9_kymographs%20tutorial.ipynb>)                                                         | `Kymograph` — 2-D T × ω → modulus                           |

:::{tip}
Every tutorial follows the same pipeline pattern:

```python
pipeline = SomeTransform(**kwargs)
result   = pipeline(experiment)
```

even when a single transform is applied — this keeps the mental model
uniform with multi-step workflows.
:::

:::{seealso}
- API reference: [hermes-rheo documentation](https://3mcloud.github.io/hermes-rheo/).
- Source: [github.com/3mcloud/hermes-rheo](https://github.com/3mcloud/hermes-rheo).
:::
