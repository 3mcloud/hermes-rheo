# Sphinx configuration for hermes-rheo.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

project = "hermes-rheo"
copyright = "2024-2026, Alessandro Perego - 3M"
author = "Alessandro Perego, Matthew J.L. Mills, Damien C. Vadillo"
try:
    release = version("hermes-rheo")
except PackageNotFoundError:
    release = ""
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = ["pyRheo", "cv2"]
numpydoc_show_class_member = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

html_theme = "sphinx_rtd_theme"
html_static_path = []
