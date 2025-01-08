---
title: 'Hermes-Rheo: A Python Package for Advanced Rheological Data Analysis'
tags:
  - Rheology
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
    orcid: 
    affiliation: 1
affiliations:
  - name: Corporate Research Laboratory, 3M, United States of America
    index: 1
date: 10 December 2024
bibliography: paper.bib
---

# Summary

Rheological measurements are foundational in understanding material properties, yet traditional techniques are often time-consuming and limited in frequency resolution. To address these limitations, the Optimally Windowed Chirp (OWChirp) methodology, introduced in Geri et al.'s 2018 study, revolutionized data collection by replacing discrete frequency sweeps with a continuously varying input signal. This approach significantly reduces data collection time while enabling high-density and high-quality data acquisition, making it particularly valuable for applications such as transient material characterization, generating material data cards, high-frequency testing (e.g., vibration damping), and quality control.

Despite its potential, the OWChirp method has been constrained by the lack of scalable and automated tools. The only existing software, MIT OWCh V1.01, is limited to MATLAB©, requiring a licensed environment, manual data cleaning, and single-signal processing. To overcome these challenges, we developed \textit{hermes-rheo}, a Python package designed for advanced rheological analysis, with a specific focus on OWChirp data processing.

# Statement of Need

The rapid evolution of data science has opened new possibilities for rheological data analysis, but a significant gap exists between general-purpose tools and the specific needs of rheology. \textit{Hermes-rheo} bridges this gap by providing an open-source, scalable framework built on the Python library \textit{piblin}, which itself derives its name from the Welsh word for "pipeline." \textit{Hermes-rheo}, named after the ancient Greek god Hermes known for speed and efficient delivery, enables researchers to efficiently analyze complex rheological datasets, from OWChirp data to traditional frequency sweeps.

Existing tools like MIT OWCh V1.01 and chrpy (from Sheridan et al.'s BOTTS methodology) lack scalability and require significant manual input, making high-throughput analysis impractical. Additionally, the fragmentation of experimental metadata, data outputs, and sample formulations has hindered the adoption of machine learning (ML) and AI in rheological research. \textit{Hermes-rheo} addresses these limitations by integrating test metadata with results, enabling the creation of FAIR datasets essential for ML-driven material development.

# Installation

\textit{Hermes-rheo} is available on the Python Package Index (PyPI). It can be installed or updated using the following commands:

```console
pip install hermes-rheo
pip install hermes-rheo --upgrade
```
The package documentation is available at [hermes-rheo.readthedocs.io](https://hermes-rheo.readthedocs.io).

# File Reader

The [`file_readers`](https://github.com/3mcloud/hermes-rheo/tree/main/src/hermes_rheo/file_readers) directory contains an example file reader for rheological data collected using [`TA TRIOS software`](https://www.tainstruments.com/trios-software).

This reader was specifically designed to handle `.txt` files generated via the "Export to LIMS" command in TRIOS. With the release of TRIOS V5, a new export format, `.json`, has been introduced ([see new features here](https://www.tainstruments.com/wp-content/uploads/NewFeaturesTRIOS.pdf)). A reader for this format is currently in development.

For other data formats or instruments, users can develop custom readers while leveraging the `hermes-rheo` package's data transforms for analysis. For assistance in creating a custom reader for your specific data format, please contact aperego[at]mmm.com.


# Features

The `hermes-rheo` package provides several powerful transforms designed to handle advanced rheological data analysis, particularly for the OWChirp methodology and other complex techniques. Key transforms include:

1. **Rheological Analysis**: 
   This transform, implemented in the `RheoAnalysis` class, specializes in analyzing rheological data, particularly from OWChirp experiments. It includes methods for bias correction, filtering, and computing viscoelastic properties. Key functionalities include:
   - Preparing chirp data by adjusting units and switching coordinates.
   - Applying various bias correction methods to ensure data symmetry and accuracy.
   - Calculating viscoelastic moduli using Fourier transform techniques on strain and stress signals.

2. **Automated Master Curve Construction**:
   The `AutomatedMasterCurve` class enables automated construction of rheological master curves using Gaussian process regression for statistical modeling and optimal superposition of datasets. This transform simplifies the creation of master curves from multiple datasets, with features like:
   - Flexible state and property selection (e.g., time, temperature, angular frequency, storage modulus).
   - Support for vertical shifts and data reversal.
   - Integration of open-source methodologies from the MIT Swan and McKinley labs.

3. **Optimal Window Chirp (OWChirp) Generation**:
   The `OWChirpGeneration` class provides tools for generating and visualizing OWChirp signals. These signals, essential for high-frequency rheological tests, are designed with precise control over parameters such as:
   - Strain amplitude, initial and final frequencies, signal duration, and tapering.
   - Waiting time and phase shift for signal refinement.

4. **Mutation Number Calculation**:
   The `MutationNumber` transform calculates and visualizes the mutation number, which characterizes the time evolution of rheological properties during processes like gelation or curing. Features include:
   - Customizable state variables (e.g., time, temperature) and dependent variables (e.g., complex modulus).
   - Integration with datasets for automated state conditioning and mutation number extraction.

These transforms are designed to enhance reproducibility, scalability, and efficiency in rheological data analysis, making `hermes-rheo` a versatile tool for both research and industrial applications.
