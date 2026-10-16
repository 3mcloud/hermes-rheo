# Vendored packages

Unmodified copies of third-party packages that are not published on PyPI
or that require a newer Python than hermes-rheo supports. Each directory
keeps the upstream licence. Do not edit these files; update them by copying
the package directory from a new upstream commit and updating the table.

| Package          | Upstream                                     | Commit    | Licence |
|------------------|----------------------------------------------|-----------|---------|
| `pyrespect_time` | https://github.com/shane5ul/pyReSpect-time   | `754335a` | MIT     |
| `pyrespect_freq` | https://github.com/shane5ul/pyReSpect-freq   | `5bcfa60` | MIT     |

Author of both packages: Sachin Shanbhag (Florida State University).

## References

If you use `RelaxationSpectrum`, please cite the papers for the solver you use.

`pyrespect_time` (relaxation spectra from G(t), `domain="time"`):

- Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and
  Continuous Spectra from Stress Relaxation Experiments", *Macromol. Theory
  Simul.* **28**, 1900005 (2019).
  [doi:10.1002/mats.201900005](https://doi.org/10.1002/mats.201900005)

`pyrespect_freq` (relaxation spectra from G'(w), G''(w), `domain="frequency"`):

- Shanbhag, S., "Relaxation spectra using nonlinear Tikhonov regularization
  with a Bayesian criterion", *Rheol. Acta* **59**, 509 (2020).
  [doi:10.1007/s00397-020-01212-w](https://doi.org/10.1007/s00397-020-01212-w)
- Shanbhag, S., *Macromol. Theory Simul.* **28**, 1900005 (2019), as above.
- Takeh, A. and Shanbhag, S., "A computer program to extract the continuous
  and discrete relaxation spectra from dynamic viscoelastic measurements",
  *Appl. Rheol.* **23**, 24628 (2013).
  [doi:10.3933/applrheol-23-24628](https://doi.org/10.3933/applrheol-23-24628)
