"""
Centralized imports for StonedFenicsx package
Organized by category to avoid redundancy
"""

# Standard library
import sys
import os
import time as timing
from functools import wraps
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from numpy import ndarray

# Third-party: Numerical computing
import numpy as np
from numpy import load
from numpy.typing import NDArray

# Third-party: Scientific computing
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg.dsolve as linsolve
from scipy.interpolate import griddata
from scipy.optimize import bisect

# Third-party: MPI and PETSc
from mpi4py import MPI
from petsc4py import PETSc

# Third-party: Mesh and FEM
import gmsh
import meshio
import ufl
from ufl import exp, conditional, eq, as_ufl, Constant,sqrt,inner
import basix
import basix.ufl
import dolfinx
from dolfinx import fem, io, nls, log, plot
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary, create_submesh
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, form
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem, assemble_matrix_block, assemble_vector_block
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.cpp.mesh import to_type

# Third-party: JIT compilation
from numba.experimental import jitclass
from numba import int64, float64, int32, types,jit, prange, njit

