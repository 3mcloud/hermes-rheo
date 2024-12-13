---
title: 'hermes-rheo -- piblin Transforms Tailored for Advanced Rheological Analysis'
tags:
  - Rheology
  - Data Science
  - Python
authors:
  - name: Alessandro Perego
    orcid: 0000-0002-0570-3210
    affiliation: 1
  - name: Matthew J.L Mills
    orcid: 
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
Currently, rheology data collection is mainly limited to discrete frequencies, resulting in time-consuming measurements 
and a lack of relevant frequency selection.  
The solution is the use of Optimally Windowed Chirp (OW-Chirp) methodology. This concept, initially presented 
in Geri et al.’s 2018 study,1 further develops rheology data collection by 
replacing discrete frequency data with a continuously increasing frequency input signal, known as CHIRP 
(same type of signal used in radars).  The method significantly reduces data collection time, allowing for high-density, high-quality data acquisition in a matter of seconds.  Such futures are extremely valuable in applications such as: rheological characterization of transient materials, material data cards generations, high frequency rheological tests (e.g., drop tests, vibration damping), and product development and quality control. 
However, the only existing software for analyzing CHIRP signals for rheological applications, MIT OWCh V1.01, 
has several limitations, including being written in the proprietary software MATLAB©, lacking deployment ease and 
high-throughput capacity, and requiring manual data cleaning and organization.  To overcome this limitation, 
a new python software named hermes was developed in conjunction to the analytical technique.  
Built on open-source tools and designed to complement and extend the cralds library, 
hermes offers a set of transforms tailored for advanced r&d rheological analysis within the cralds2  'Dataset' framework.  

Initially developed as a Python package to facilitate the analysis of rheological data collected using the OWChirp method. 
The software has now become a universal rheology tool containing specialized transforms for advanced rheological analysis
within the piblin framework.
# Statement of Need
Scientific research produces significant volumes of structured and hi-fidelity data which require expert-guided processing 
prior to the generation of insight through visualization and modeling. The chemical community is embracing modern 
data-handling concepts like cloud processing and advanced machine learning to extract maximum scientific and economic 
value from data-rich sources. A bridge from existing approaches for collection, storage, and use of analytical data to 
this new paradigm is needed. Data scientists with relevant physical science domain knowledge are key to making the 
connection between subject-matter experts and emerging technologies with potential to improve their workflows. 
However, in many cases there are gaps in applications between generalized ‘big data’ approaches that seek to identify 
and establish qualitative “trends” and the specific quantitative needs of measurement science.  The piblin Python package 
aims to address these needs by providing a fundamental conceptual framework for reading, visualizing, processing, and 
writing analytical data, along with a concrete, accessible implementation.
To specifically address the needs of the rheology community, we have developed the hermes Python package to complement 
and extend the piblin library.  Hermes offers a set of specialized transforms tailored for advanced rheological analysis 
within the piblin framework.  The paper we will demonstrate how these transforms can enhance data analysis workflows in 
rheological datasets, bridging the gap between general data-rich methodologies and the specialized research and 
development requirements of measurement science.