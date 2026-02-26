"""Sphinx configuration for quantkit documentation."""

project = "QuantKit"
copyright = "2026, QuantKit contributors"
# author = "Ayoub Ghriss"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]

# MyST settings
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_mock_imports = ["torch", "triton"]

# Theme
html_theme = "sphinx_rtd_theme"

# Source suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_static_path = ["_static"]
html_logo = "_static/quantkitlogo.png"
html_css_files = ["custom.css"]
