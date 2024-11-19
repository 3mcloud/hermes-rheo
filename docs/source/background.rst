Background
==========

The rapid evolution of data science has opened new possibilities for advanced analysis in measurement science,
particularly within the field of rheology. Despite these advancements, a gap remains between generalized data processing
techniques and the specific requirements of rheological data analysis. This often leaves researchers searching for specialized
tools tailored to their unique data challenges.

To address this need, we introduce the `hermes` Python package. Building on the abstraction provided by the
`piblin <https://github.com/3mcloud/piblin>`_ library, `hermes` offers a comprehensive suite of specialized transforms and
analytical tools explicitly designed for rheological data. The package enhances research workflows by enabling precise reading,
visualization, processing, and export of rheological datasets, all within a flexible framework.

Key features of `hermes` include the ability to analyze novel rheological methods such as the
`Optimal Windowed Chirp (OWChirp) <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.041042>`_ technique and the
`automated, data-driven creation of rheological mastercurves <https://www.cambridge.org/core/journals/data-centric-engineering/article/datadriven-method-for-automated-data-superposition-with-applications-in-soft-matter-science/DA44C868EE1128DD79798653A1376594?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark>`_. Additionally,
`hermes` provides an ideal infrastructure for implementing and exploring new analytical rheological techniques.
These capabilities position `hermes` as a powerful tool for advancing research in rheology, bridging the gap
between data science and rheological measurement needs.


Citing
------

Publication of this work is forthcoming. For now, if you use this software, please cite it using:
    .. image:: https://zenodo.org/badge/851879885.svg
       :target: https://doi.org/10.5281/zenodo.14182224
       :alt: DOI

Installation
------------

- `hermes` is in the Python Package Index! You can now quickly install it using pip.

.. code-block:: console

   $ pip install hermes-rheo

You can also use pip to download updates:

.. code-block:: console

   $ pip install hermes-rheo --upgrade

Contributing
------------

Questions, comments, or suggestions can be raised as issues on `GitHub <https://github.com/3mcloud/hermes-rheo>`_
or emailed directly to aperego[at]mmm.com.