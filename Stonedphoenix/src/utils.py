import sys, os

from mpi4py                          import MPI
from petsc4py                        import PETSc

import ufl
import time                          as timing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import wraps
# Util performance 
#---------------------------------------------------------------------------------------------------------
def timing_function(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        time_A = timing.time()
        result = fun(*args, **kwargs)
        time_B = timing.time()
        dt = time_B - time_A
        global_dt = comm.allreduce(dt, op=MPI.MAX)
        if comm.rank == 0:
            print(f"{fun.__name__} took {dt:.2f} sec")
        return result
    return wrapper
#---------------------------------------------------------------------------------------------------------
def print_ph(string):
    
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0: 
        print(string)

#---------------------------------------------------------------------------------------------------------