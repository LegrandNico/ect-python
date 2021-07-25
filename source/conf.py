# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import time
import systole
import sphinx_bootstrap_theme

# -- Project information -----------------------------------------------------

project = "systole"
copyright = u"2020-{}, Nicolas Legrand".format(time.strftime("%Y"))
author = "Nicolas Legrand"
release = systole.__version__


image_scrapers = ("matplotlib",)

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "backreferences_dir": "api",
    "image_scrapers": image_scrapers,
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "jupyter_sphinx",
    "sphinx_panels",
    "nbsphinx",
    "sphinx_gallery.load_style",
]

panels_add_bootstrap_css = False

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Include the example source for plots in API docs
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {
    'icon_links': [
        dict(name='GitHub',
             url='https://github.com/embodied-computation-group/systole',
             icon='fab fa-github-square'),
        dict(name='Twitter',
             url='https://twitter.com/visceral_mind',
             icon='fab fa-twitter-square'),
    ],
    "logo_link": "https://systole-docs.github.io/"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_logo = "images/logo.png"

# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}
