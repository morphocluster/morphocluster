# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

from morphocluster import __version__  # noqa

project = "MorphoCluster"
copyright = "2018-2023, Simon-Martin Schroeder"
author = "Simon-Martin Schroeder"
version = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# napoleon is for parsing of Google-style docstrings:
# https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
extensions = [
    # "sphinx.ext.autodoc",
    # "sphinx.ext.coverage",
    # "sphinx.ext.napoleon",
    # "sphinxcontrib.programoutput",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.intersphinx",
    # "sphinx_autodoc_typehints",
    # "sphinx.ext.coverage",
    # "sphinx.ext.autosummary",
]

# napoleon_use_param = False
# napoleon_use_keyword = False
# napoleon_use_rtype = False
# typehints_document_rtype = False

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     # TODO: Always link to the latest stable
#     "pims": ("https://soft-matter.github.io/pims/v0.4.1/", None),
#     "skimage": ("https://scikit-image.org/docs/stable/", None),
#     "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# rst_prolog = """
# .. |stream| raw:: html

#    <span class="label">Stream</span>
# """

# html_css_files = ["custom.css"]
