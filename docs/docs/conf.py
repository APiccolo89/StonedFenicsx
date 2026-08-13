import os
import sys
from pathlib import Path

# repo root (docs/..)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "StonedFEniCSx"
html_title = project
html_short_title = project

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.githubpages",
]

autosummary_generate = True

autodoc_mock_imports = ["dolfinx", "gmsh", "petsc4py", "mpi4py", "ufl", "basix", "ffcx"]

autodoc2_packages = [
    "../../stonedfenicsx",
]

bibtex_bibfiles = ["bibliography.bib"]
numfig = True


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "piccolo_theme"
html_static_path = ["_static"]
html_css_files = ['custom.css']