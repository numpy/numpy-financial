# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from datetime import datetime

import numpy_financial

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'numpy-financial'
copyright = f'2005-{datetime.now().year}, NumPy Developers'
author = 'numpy-financial developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.mathjax',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

version = numpy_financial.__version__

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_copy_source = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/numpy_financial_logov.svg"
html_favicon = "_static/numpy_financial_favicon.png"

# -- Version switcher configuration ------------------------------------------

# Determine the version string for the switcher.
# Must match a "version" field in versions.json for the dropdown to show it selected.
#
# In production (CI), DOCS_VERSION is set explicitly:
#   - "dev" for main branch builds (deployed to /dev/)
#   - "X.Y.Z" for release tag builds (deployed to /version/X.Y.Z/)
#
# For local development, default to showing stable version to test the typical UX.
if os.environ.get("DOCS_VERSION"):
    # CI sets this explicitly based on branch/tag
    switcher_version = os.environ["DOCS_VERSION"]
else:
    # Local development: show stable version (1.1.0) as default for UX testing
    # This matches what users see when visiting the main docs site
    switcher_version = "1.1.0"

# versions.json is generated dynamically on gh-pages by the CI workflow.
# Always use the absolute URL (local development requires internet access).
json_url = "https://numpy.org/numpy-financial/_static/versions.json"

html_theme_options = {
    "github_url": "https://github.com/numpy/numpy-financial",
    # Navbar layout: theme switcher, version switcher, then GitHub icon
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": json_url,
        "version_match": switcher_version,
    },
    "show_version_warning_banner": True,
}
