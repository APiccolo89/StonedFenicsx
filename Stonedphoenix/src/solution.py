import numpy as np
import gmsh 

import ufl
import meshio
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile, gmshio
import gmsh 
from ufl import exp, conditional, eq, as_ufl
import scal as sc_f 
import basix.ufl



class Solution():
    def __init__(self):
        self.P_L : dolfinx.fem.function.Function 
        self.T_O : dolfinx.fem.function.Function 
        self.T_N : dolfinx.fem.function.Function 
        self.Vel : dolfinx.fem.function.Function 
        self.P   : dolfinx.fem.function.Function 

# Extract solution array 
