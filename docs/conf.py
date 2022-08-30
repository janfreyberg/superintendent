#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# superintendent documentation build configuration file, created by
# sphinx-quickstart on Sun Feb 18 13:13:36 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os

# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "jupyter_sphinx",
    "sphinx_automodapi.automodapi",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "myst_parser",
]

numpydoc_show_class_members = False
# autosummary_generate = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}
# source_suffix = '.rst'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "superintendent"
copyright = "2018, Jan Freyberg"
author = "Jan Freyberg"

# get version from python package:
here = os.path.dirname(__file__)
repo = os.path.join(here, "..")
_version_py = os.path.join(repo, "src", "superintendent", "_version.py")
version_ns = {}
with open(_version_py) as f:
    exec(f.read(), version_ns)

# The short X.Y version.
version = "%i.%i" % version_ns["version_info"][:2]
# The full version, including alpha/beta/rc tags.
release = version_ns["__version__"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_css_files = ["css/hide-double-widgets.css"]

html_favicon = "favicon.ico"
html_logo = "img/logo.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
        "globaltoc.html",
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "superintendentdoc"


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "superintendent", "superintendent Documentation", [author], 1)
]


# -- Options for nbsphinx -------------------------------------------------

nbsphinx_prolog = """
.. note::
    This page will display what superintendent widgets look like, but not
    respond to user input (as it's not connected to a backend).
"""

# if this is not set to empty string, widgets get displayed twice, see:
# https://nbsphinx.readthedocs.io/en/0.5.0/usage.html#nbsphinx_widgets_path
# https://github.com/spatialaudio/nbsphinx/issues/378
# nbsphinx_widgets_path = ""


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/": None}


def setup(app):
    app.add_css_file("css/custom.css")
    # app.add_directive('autoautosummary', AutoAutoSummary)
