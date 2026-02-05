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
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",   # <-- add this
]

autosummary_generate = True

autodoc_mock_imports = ["dolfinx", "gmsh", "petsc4py", "mpi4py", "ufl", "basix", "ffcx"]

autodoc_typehints = "description"   # or "none"
autodoc_typehints_format = "short"

bibtex_bibfiles = ["bibliography.bib"]
numfig = True


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "piccolo_theme"
html_static_path = ["_static"]
