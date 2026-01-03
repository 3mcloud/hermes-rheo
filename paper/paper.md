---
title: 'hermes-rheo: An open-source Python package for rheological data analysis'
tags:
  - Rheology
  - Time-resolved mechanical spectroscopy 
  - Data Science
  - Python
  - OWChirp
  - Machine Learning
authors:
  - name: Alessandro Perego
    orcid: 0000-0002-0570-3210
    affiliation: 1
  - name: Matthew J.L. Mills
    orcid: 0000-0002-7208-7542
    affiliation: 1
  - name: Damien C. Vadillo
    orcid: 0000-0003-0555-745X
    affiliation: 1
affiliations:
  - name: Corporate Research Analytical Laboratory, 3M, 3M Corporate Headquarters, St. Paul, MN, 55144, USA
    index: 1
date: 11 February 2025
bibliography: paper.bib
---
# Summary
Scientific research produces significant volumes of structured and hi-fidelity data which require expert-guided processing 
prior to the generation of insight through visualization and modeling. Data scientists with relevant physical science 
domain knowledge are key to making the connection between subject matter experts and emerging technologies with 
potential to improve their workflows. However, in many cases there are gaps in applications between generalized 
'big data' approaches that seek to identify and establish qualitative 'trends' and the 
specific quantitative needs of measurement science. The _piblin_ [@mills2024piblin] Python package developed at [3M](https://www.3m.com/3M/en_US/company-us/about-3m/research-development/) 
aims to address these needs by providing a fundamental conceptual framework for reading, 
visualizing, processing, and writing analytical data, along with a concrete, accessible implementation. 

To specifically address the needs of the rheology community, we have developed the _hermes-rheo_ [@perego2024hermesrheo] 
Python package to 
complement and extend the piblin library. _hermes-rheo_ offers a set of specialized transforms tailored for advanced rheological 
analysis within the piblin framework. These transforms can enhance data analysis 
workflows in rheological datasets, bridging the gap between general data-rich methodologies and the specialized research and 
development requirements of measurement science. 


# Statement of Need

The analysis of rheological datasets presents several technical challenges that hinder efficient data processing and the 
integration of novel analytical methodologies. Current workflows are predominantly dependent on proprietary software, 
which imposes significant limitations in customizing analysis pipelines and implementing emerging techniques such as 
Optimally Windowed Chirp [@owchirp; @peregoowch2025], 
Gaborheometry [@gabor], and Recovery Rheology [@recovery]. Moreover, the frequent development of new rheological models requires adaptable 
tools capable of accommodating evolving analytical frameworks, a flexibility often lacking in existing solutions. 
The increasing volume of experimental data further exacerbates the complexity of managing and processing large datasets 
efficiently. Additionally, the integration of multi-instrument and multi-technique data formats remains a critical bottleneck, 
complicating data interoperability and standardization across different measurement platforms.

The _hermes-rheo_ Python package addresses these limitations by providing an open-source, extensible framework designed to 
facilitate the analysis, visualization, and processing of rheological data with enhanced flexibility, scalability, 
and reproducibility.

# Installation

_hermes-rheo_ is available on the Python Package Index (PyPI). It can be installed or updated using the following commands:

```console
pip install hermes-rheo
pip install hermes-rheo --upgrade
```

# Documentation

The package documentation is publicly available at [hermes-rheo.readthedocs.io](https://hermes-rheo.readthedocs.io).

# Features 

The `hermes-rheo` package provides a series of powerful transforms designed to handle rheological data analysis, 

#### Key transforms include:
```console
   from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
   from hermes_rheo.transforms.automated_mastercurve import AutomatedMastercurve
   from hermes_rheo.transforms.mutation_number import MutationNumber
   from hermes_rheo.transforms.owchirp_generation import OWChirpGeneration
```

* The `RheoAnalysis` transform is the core tool in _hermes-rheo_, offering a fast and efficient analysis of
viscoelastic properties from datasets collected in both the frequency and time domains.
It supports standard rheological tests, including frequency and temperature sweeps, creep, 
stress-relaxation, and flow and temperature ramps. Additionally, it can analyze time-resolved 
mechanical spectroscopy measurements obtained via Optimally Windowed Chirp, accommodating 
data from both stress- and strain-controlled rheometers.

* The `AutomatedMasterCurve` transform automatically generates master curve datasets (e.g., time-temperature superposition) 
through a data-driven machine learning algorithm developed by [@lennontts]. The method employs Gaussian process regression 
and maximum a posteriori estimation to automatically superimpose datasets.

* The `MutationNumber` transform returns the mutation number, $M_u$, using the following definition:[@mutation]

$$M_u = \frac{T}{\lambda_{\mu}}$$

where $\lambda_{\mu}(t)$ is:

$$\lambda_{\mu}(t) = \left( \frac{d \ln g}{dt} \right)^{-1} \approx \frac{t_2 - t_1}{\ln \left( \frac{g_{t_2}}{g_{t_1}} \right)}$$

where g can be any viscoelastic property of interest (e.g. G*, G', G'').

* The `OWChirpGeneration` transform helps users design Optimally Windowed Chirp signals for use in their rheometers and is currently optimized for experiments in TA TRIOS via its arbitrary waveform functionality [@peregoowch2025].

The [_hermes-rheo_](https://github.com/3mcloud/hermes-rheo/) project is under continuous development, and new transforms are regularly introduced to expand its functionality. If you have ideas or suggestions for additional transforms, please open a new issue on the [GitHub Issues page](https://github.com/3mcloud/hermes-rheo/issues). Contributions and feedback from the community help shape the future of **hermes-rheo** and ensure its capabilities stay aligned with user needs.

# Tutorials

The software repository contain a [`tutorial_notebooks`](https://github.com/3mcloud/hermes-rheo/tree/main/tutorial_notebooks) folder 
with a comprehensive set of Jupyter notebooks demonstrating the diverse capabilities of *hermes-rheo*. 


# Available File Readers

The [`file_readers`](https://github.com/3mcloud/hermes-rheo/tree/main/src/hermes_rheo/file_readers) directory contains an example file reader for rheological data collected using [`TA TRIOS software`](https://www.tainstruments.com/trios-software).

This reader was specifically designed to handle `.txt` files generated via the "Export to LIMS" command in TRIOS.  
With the release of TRIOS V5, a new export format, `.json`, has been introduced ([see new features here](https://www.tainstruments.com/wp-content/uploads/NewFeaturesTRIOS.pdf)).  
A reader for this format is currently in development. Additionally, a reader for Anton Paar CSV files is actively being developed and is expected to be released soon.

For other data formats or instruments, users can develop custom readers while leveraging the `hermes-rheo` package's data transforms for analysis. Requests for additional file formats can be raised as an issue in the [hermes-rheo GitHub repository](https://github.com/3mcloud/hermes-rheo/issues).

# Acknowledgments

The authors gratefully acknowledge the collaboration with Gareth H. McKinley and Mohua Das from the Department of Mechanical Engineering at the Massachusetts Institute of Technology. Their insightful discussions significantly contributed to the development of the Optimally Windowed Chirp analysis within the hermes-rheo software.

# References

