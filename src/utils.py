import sys, os

from mpi4py                          import MPI
from petsc4py                        import PETSc

import ufl
import time                          as timing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import wraps
import cmcrameri as cmc
import numpy as np 
from dataclasses import dataclass
from dolfinx import fem


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
            if dt > 60.0:
                m, s = divmod(dt, 60)
                print(f".  {fun.__name__} took {m:.2f} min and {s:.2f} sec")
            if dt > 3600.0:
                m, s = divmod(dt, 60)
                h, m = divmod(m, 60)
                print(f".  {fun.__name__} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
            else:
                print(f".  {fun.__name__} took {dt:.2f} sec")
        return result
    return wrapper

def time_the_time(dt):
    comm = MPI.COMM_WORLD 
    global_dt = comm.allreduce(dt,op=MPI.MAX)
    
    return global_dt 

    
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

def interpolate_from_sub_to_main(u_dest, u_start, cells,parent2child=0):
    """
    Interpolate the solution from the subdomain to the main domain.
    
    Parameters:
        u_slab (Function): The solution in the subdomain.
        u_global (Function): The solution in the main domain.
        M (Mesh): The mesh of the main domain.
        V (FunctionSpace): The function space of the main domain.
    """
    from dolfinx                         import mesh
    import numpy as np
    
    if parent2child == 0: 
        a = np.arange(len(cells))
        b = cells
    else: 
        a = cells
        b = np.arange(len(cells))
    
    
    u_dest.interpolate(u_start,cells0=a,cells1=b)

    return u_dest

def gather_vector(v): 
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Suppose v is fem.Function(V) (not the UFL test function!)
    v.x.scatter_forward()  # update ghosts from owners

    # Get number of owned dofs (exclude ghosts)
    imap = v.function_space.dofmap.index_map
    n_owned = imap.size_local

    # Slice only owned part
    lv = np.asarray(v.x.array[:n_owned], dtype=np.float64)  # ensure dtype/contiguous

    # Gather element counts on root
    sizes = comm.gather(lv.size, root=0)

    if rank == 0:
        counts = np.asarray(sizes, dtype='i')
        displs = np.insert(np.cumsum(counts[:-1]), 0, 0).astype('i')
        gv = np.empty(int(counts.sum()), dtype=np.float64)
    else:
        counts = None
        displs = None
        gv = None

    # Gather variable-length arrays; only root provides recv buffers/metadata
    comm.Gatherv(lv, (gv, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:

        return lv 


def gather_coordinates(V):
    """
    Gather DOF coordinates for a dolfinx FunctionSpace V to rank 0.
    Returns an (ndofs_global, gdim) array on rank 0, else None.
    """
    comm  = V.mesh.comm
    gdim  = V.mesh.geometry.dim

    # Local coordinates (shape: n_local_total x gdim). Slice to owned DOFs only.
    coords_local = V.tabulate_dof_coordinates()
    n_owned = V.dofmap.index_map.size_local
    coords_owned = coords_local[:n_owned, :]

    # Flatten to 1D buffer for Gatherv
    sendbuf = np.ascontiguousarray(coords_owned.ravel(), dtype=np.float64)

    # Gather row counts, then convert to element counts by * gdim
    rows_local = coords_owned.shape[0]
    rows_counts = comm.gather(rows_local, root=0)

    if comm.rank == 0:
        elem_counts = np.asarray(rows_counts, dtype='i') * gdim           # elements, not bytes
        elem_displs = np.insert(np.cumsum(elem_counts[:-1]), 0, 0).astype('i')
        recvbuf = np.empty(int(elem_counts.sum()), dtype=np.float64)
    else:
        elem_counts = None
        elem_displs = None
        recvbuf = None

    # Gather – use MPI.DOUBLE (not a NumPy dtype)
    comm.Gatherv(sendbuf, (recvbuf, elem_counts, elem_displs, MPI.DOUBLE), root=0)

    if comm.rank == 0:
        return recvbuf.reshape(-1, gdim)
    else:
        return None


#----------------------------------------------------------------------------          
def compute_strain_rate(u):
    
    e = ufl.sym(ufl.grad(u))
    
    return e  
#---------------------------------------------------------------------------

def compute_eII(e):
    from ufl import inner, sqrt
    e_II  = sqrt(0.5*inner(e, e))    
    return e_II
#---------------------------------------------------------------------------

@dataclass(slots=True)
class Phase:
    """
    Phase: container for rheological and thermal material parameters.

    ------------------
    Rheology (viscosity)
    ------------------
    name_diffusion : str
        Diffusion creep flow law name.
        Options include (non-exhaustive):
          - 'Constant'                : constant viscosity
          - 'Hirth_Dry_Olivine_diff'  : Hirth & Kohlstedt (2003), dry olivine
          - 'Van_Keken_diff'          : Van Keken et al. (2008) style diffusion
          - 'Hirth_Wet_Olivine_diff'  : Hirth & Kohlstedt (2003), wet olivine

    Edif : float
        Activation energy for diffusion creep [J/mol].
    Vdif : float
        Activation volume for diffusion creep [m³/mol].
    Bdif : float
        Pre-exponential factor for diffusion creep [1/Pa/s].

    name_dislocation : str
        Dislocation creep flow law name.
        Options include:
          - 'Constant'
          - 'Hirth_Dry_Olivine_disl'
          - 'Van_Keken_disl'
          - 'Hirth_Wet_Olivine_disl'

    n : float
        Stress exponent. **NB**: if you change this, Bdis must be updated consistently.
    Edis : float
        Activation energy for dislocation creep [J/mol].
    Vdis : float
        Activation volume for dislocation creep [m³/mol].
    Bdis : float
        Pre-exponential factor for dislocation creep [1/Pa^n/s].

    eta : float
        Constant viscosity [Pa·s] (used if rheology is 'Constant').

    ------------------
    Thermal properties
    ------------------
    Cp : float
        constant heat capacity [J/kg/K].
    k : float
        constant thermal conductivity [W/m/K].
    rho0 : float
        Reference / constant density [kg/m³].

    name_capacity : str
        Heat capacity law.
        Options:
          - 'Constant'
          - 'Berman_Forsterite'
          - 'Berman_Fayalite'
          - 'Berman_Aranovich_Forsterite'
          - 'Berman_Aranovich_Fayalite'
          - 'Berman_Fo_Fa_01'
          - 'Bermann_Aranovich_Fo_Fa_0_1'
          - 'Oceanic_Crust'
          - 'ContinentalCrust' (not implemented / to be removed).

    name_density : str
        Density law.
        Options:
          - 'Constant' : ρ = ρ0.
          - 'PT'       : ρ(P,T) with constant bulk modulus K₀ ≈ 130e9 Pa and
                         thermal expansivity consistent with `name_alpha`.

    name_alpha : str
        Thermal expansivity law (α).
        Options:
          - 'Constant'      : α = 3e-5 K⁻¹.
          - 'Mantle'        : olivine / mantle α (e.g., Groose & Afonso 2013;
                              Richardson et al. 2020).
          - 'Oceanic_Crust' : basaltic crustal α.

    name_conductivity : str
        Thermal conductivity law (k).
        Options:
          - 'Constant'
          - 'Mantle'
          - 'Oceanic_Crust'

    ------------------
    Internal heating
    ------------------
    radio : float
        Radiogenic heat production [W/m³] (or [Pa/s] if used as source in σ units).
    radio_flag : float
        Activation flag for radiogenic heating / radiative conductivity
        (0.0 = off, 1.0 = on, or a more general scaling factor).

    Notes
    -----
    This class is intended as a flexible container for building a PhaseDataBase.
    For your current kinematic slab work it may be somewhat overkill, but it
    should be reusable for other problems.
    """
    name_phase:str = "Undefined Phase"
    # Viscosity / rheology
    name_diffusion: str = "Constant"
    Edif: float = -1e23
    Vdif: float = -1e23
    Bdif: float = -1e23

    name_dislocation: str = "Constant"
    n: float = -1e23
    Edis: float = -1e23
    Vdis: float = -1e23
    Bdis: float = -1e23

    eta: float = 1e20  # constant viscosity

    # Thermal properties
    Cp: float = 1250.0
    k: float = 3.0
    rho0: float = 3300.0
    radio_flag: float = 0.0

    name_capacity: str = "Constant"
    name_conductivity: str = "Constant"
    name_alpha: str = "Constant"
    name_density: str = "Constant"
    alpha0      : float = 3e-5  # constant thermal expansivity
    Hr : float = 0.0
    # Internal heating
    radio: float = 0.0

class Ph_input():
    def __init__(self):
        pass
    
def evaluate_material_property(expr, V):
    F = fem.Function(V)
    F.interpolate(fem.Expression(expr, V.element.interpolation_points()))
    return F 
