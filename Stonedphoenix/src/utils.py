import sys, os

from mpi4py                          import MPI
from petsc4py                        import PETSc

import ufl
import time                          as timing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import wraps
import cmcrameri as cmc
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

#---------------------------------------------------------------------------
def get_discrete_colormap(n_colors, base_cmap='cmc.lipari'):
    import numpy as np
    """
    Create a discrete colormap with `n_colors` from a given base colormap.
    
    Parameters:
        n_colors (int): Number of discrete colors.
        base_cmap (str or Colormap): Name of the matplotlib colormap to base on.
    
    Returns:
        matplotlib.colors.ListedColormap: A discrete version of the colormap.
        copied from internet
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_colors))
    return mcolors.ListedColormap(color_list, name=f'{base_cmap}_{n_colors}')
#---------------------------------------------------------------------------