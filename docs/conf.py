# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# This site renders the tutorial notebooks with nbsphinx. The build does NOT execute
# notebooks (``nbsphinx_execute = "never"``) and does NOT import jaxpm, so ReadTheDocs
# only needs sphinx + nbsphinx + theme (see docs/requirements.txt) plus pandoc (apt).

project = "JaxPM"
author = "JaxPM developers"
copyright = "2026, JaxPM developers"

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
]

# Notebooks are rendered from their committed outputs; never executed at build time.
nbsphinx_execute = "never"

root_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "sphinx_rtd_theme"
