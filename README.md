[pypi-image]: https://badge.fury.io/py/hermes-rheo.svg
[pypi-url]: https://pypi.org/project/hermes-rheo/
[pypi-download]: https://static.pepy.tech/badge/hermes-rheo?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads
[docs-image]: https://img.shields.io/badge/docs-latest-blue

# hermes-rheo

[![PyPI Version][pypi-image]][pypi-url] [![pypi download][pypi-download]][pypi-url] 

Python package designed to complement and extend the [piblin](https://github.com/3mcloud/piblin) library, 
offering a set of transforms tailored for analysis of rheological data.
Publication of this work is forthcoming. For now, if you use this software, please cite it using the metadata in the 
[citation](https://github.com/3mcloud/hermes-rheo/blob/main/CITATION.cff) file.

## Documentation

Documentation to learn more about the package and how to use its API will be available soon. In the meantime, tutorial 
notebooks can be found in the tutorial_notebooks folder

## Installation

`hemres-rheo` is in the PyPI! You can easily install it using `pip`:

```
pip install hermes-rheo
```

and likewise update it:

```
pip install hermes-rheo --upgrade
```

## Usage

### Importing the package

Once the package has been installed, you can simply import its modules:

```python
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis
```

## Examples

The [tutorial_notebooks folder](https://github.com/3mcloud/hermes-rheo/tree/main/tutorial_notebooks), 
contains multiple examples that showcase the software’s functionality in detail.

## Style and Supporting Tools

This repository follows the naming conventions in
[PEP-8](https://www.python.org/dev/peps/pep-0008/#package-and-module-names), 
docstring conventions in
[PEP-257](https://www.python.org/dev/peps/pep-0257/)
and versioning conventions in
[PEP-440](https://www.python.org/dev/peps/pep-0440/#public-version-identifiers).
The [pytest](https://docs.pytest.org/en/latest/) library is used for testing,
using the 
[pytest-html](https://pypi.org/project/pytest-html/1.6/)
and 
[pytest-cov](https://pypi.org/project/pytest-cov/) 
plugins.
Documentation is produced with
[Sphinx](http://www.sphinx-doc.org/en/master/)
and all docstrings are written in the 
[numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).
Distributions are produced with 
[setuptools](https://setuptools.readthedocs.io/en/latest/) and 
[conda-build](https://docs.conda.io/projects/conda-build/en/latest/).
The use of the
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
format for commit messages is suggested but not enforced.

## Contibuting

Inquiries and suggestions can be directed to aperego[at]mmm.com. 
## License

[MIT](https://choosealicense.com/licenses/mit/)

