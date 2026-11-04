"""Sphinx configuration: tutorial gallery, background pages and API reference.

The tutorials are built by sphinx-gallery, the background pages are Markdown read by myst-parser.

Build locally with ``python -m sphinx -b html docs docs/_build/html`` (needs the ``docs`` extra).
Environment variables:

* ``DIRCONN_QUICK=1`` runs the tutorials in their quick mode (shorter simulations), for a fast
  build;
* ``DIRCONN_DOCS_SAMPLE`` decides whether tutorials 7, 8, 13 and 14 are run. They need the MNE
  ``sample`` dataset (7 and 8 also MNE's 3D plotting). By default they are run when the dataset is
  already on this
  machine (it is never downloaded by the build) and shown with code and text but without output
  otherwise; ``1`` forces them to run and ``0`` to be skipped.
"""

import os

import dirconn

project = "dirconn"
author = "Hanwei Liu"
copyright = "Hanwei Liu, BSD-3-Clause"
release = dirconn.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
# the *_findings.md files and benchmark_conclusions.md are the hand-written parts that the
# report scripts put in front of the generated pages, so they are not pages of their own
exclude_patterns = [
    "_build",
    "generated/*.ipynb",
    "*_findings.md",
    "benchmark_conclusions.md",
]
myst_heading_anchors = 3

html_theme = "pydata_sphinx_theme"
html_title = f"dirconn {release}"
html_theme_options = {
    "github_url": "https://github.com/HanweiLiu-Viola/dirconn",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}


def _sample_dataset_is_here():
    """Whether the sample dataset is on this machine; looking never downloads anything."""
    try:
        dirconn.datasets.require_sample()
    except FileNotFoundError:
        return False
    return True


_setting = os.environ.get("DIRCONN_DOCS_SAMPLE")
RUN_SAMPLE_TUTORIALS = _setting == "1" or (_setting is None and _sample_dataset_is_here())

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_tutorials"],
    # every tutorial is run except 7, 8, 13 and 14 without the sample dataset (see above)
    # (the pattern accepts both path separators: Windows paths use a backslash)
    "filename_pattern": r"[/\\]plot_" if RUN_SAMPLE_TUTORIALS else r"[/\\]plot_(?!0[78]_|1[34]_)",
    "within_subsection_order": "FileNameSortKey",  # by tutorial number, not by code length
    # view_*.py scripts open interactive windows; they are not tutorials and are never run here
    "ignore_pattern": r"view_\d+_",
    "remove_config_comments": True,
    "show_memory": False,
    "download_all_examples": False,
    "min_reported_time": 5,
}
