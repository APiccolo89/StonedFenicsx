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



def logistic_function_decoupling():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",       # serif = Computer Modern
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    })
    
    
    
    z = np.linspace(0,300e3,500)
    lit = 50e3
    dec = 100e3 
    creep = 40e3 
    
    # jump function 
    # Linear 
    jfl         = np.zeros(len(z))
    jfl[z<lit]  = 1.0 
    jfl[z>=lit] = 1+(z[z>=lit]-lit)/(lit-dec)
    jfl[z>dec]  = 0.0
    
    # Tanh 
    jtanh      = np.zeros(len(z))
    m          = (lit+dec)/2
    ls         = (dec-m)
    jtanh      = 1-0.5 * ((1)+(1)*np.tanh((z-m)/(ls/4)))
    
    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(jfl,-z/1e3,c='k',label='linear_function_jump')
    ax0.set_ylabel('Depth [km]')    
    ax0.set_xlabel('friction efficiency [n.d.]')
    ax0.plot(jtanh,-z/1e3,c='firebrick',label='tan_function_jump')
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend()
    plt.savefig("../../jump_functions.png", dpi=300, bbox_inches='tight')
    plt.close()
    # Friction function 
    # linear 
    frl           = np.zeros(len(z))
    frl[z<creep]  = 1.0 
    frl[z>=creep] = 1+(z[z>=creep]-creep)/(creep-dec)
    frl[z>dec]    = 0 
    
    # Tanh 
    frtan           = np.zeros(len(z))
    m          = (creep+dec)/2
    ls         = 2*(dec-m)
    frtan      = 1-0.5 * ((1)+(1)*np.tanh((z-m)/(ls/4)))
    
    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(frl,-z/1e3,c='k',label='linear friction efficiency')
    ax0.set_ylabel('Depth [km]')    
    ax0.set_xlabel('friction efficiency [n.d.]')
    ax0.plot(frtan,-z/1e3,c='firebrick',label='tan friction efficiency')
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend()
    plt.savefig("../../efficiency_functions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # case 1 
    eff1 = jfl*frl 
    # case 2 
    eff2 = jfl*frtan
    # case 3 
    eff3 = jtanh*frl 
    # case 4
    eff4 = jtanh*frtan 
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)      
    ax0.plot(eff1,-z/1e3,c='k',label='linearxlinear',linestyle='-.')
    ax0.plot(eff2,-z/1e3,c='r',label='linearxnonlinear',linestyle='-.')
    ax1.plot(eff3,-z/1e3,c='forestgreen',label='nonlinearxlinear')
    ax1.plot(eff4,-z/1e3,c='firebrick',label='nonlinearxnonlinear')
    ax1.set_ylim([-100,0])
    ax0.set_ylim([-100,0])
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax0.set_ylabel(r'Depth [km]')
    ax1.set_xlabel(r'effective_efficiency [n.d.]')
    ax0.set_xlabel(r'effective_efficiency [n.d.]')
    ax0.legend()
    ax1.legend()
    plt.savefig("../../effective_function.png", dpi=300, bbox_inches='tight')
    plt.close()
    # W/m3 frictional heating
    pl = 3300.0*9.81*z/1e3
    v  = 5.0/1e2/(365.25*24*60*60)
    c1 = eff1*pl*v 
    c2 = eff2*pl*v
    c3 = eff3*pl*v
    c4 = eff4*pl*v
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)      
    ax0.plot(c1,-z/1e3,c='k',label='linearxlinear',linestyle='-.')
    ax0.plot(c2,-z/1e3,c='r',label='linearxnonlinear',linestyle='-.')
    ax1.plot(c3,-z/1e3,c='forestgreen',label='nonlinearxlinear')
    ax1.plot(c4,-z/1e3,c='firebrick',label='nonlinearxnonlinear')
    ax1.set_ylim([-100,0])
    ax0.set_ylim([-100,0])
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax0.set_ylabel(r'Depth [km]')
    ax1.set_xlabel(r'$\psi$ [$\frac{W}{m^3}$]')
    ax0.set_xlabel(r'$\psi$ [$\frac{W}{m^3}$]')
    ax0.legend()
    ax1.legend()
    plt.savefig("../../Power.png", dpi=300, bbox_inches='tight')
    plt.close()

    dc_u = 100e3
    lit = 50e3
    lit = lit/dc_u
    z = z/dc_u
    dc = 1.0
    zm = (dc + lit)/2
    k = 15.0
    C = 1.0 / (1 + np.exp(-k *(z-zm)))
    C2 = np.zeros(len(z))
    C2[z < lit] = 0.0
    C2[z >= lit] = 1.0/(dc-lit)*(z[z >= lit]-lit)
    C2[C2 > 1.0] = 1.0
    plt.plot(z-lit,C)
    plt.plot(z-lit,C2)
    plt.xlabel('Depth [m]')
    plt.ylabel('Decoupling factor')
    plt.title('Logistic function for decoupling')
    plt.grid()
    plt.show()
    plt.close()
    
    dv = ((1.0 - C)*1e-9)*((z*dc)*9.81*3300)*0.06
    dv2 = ((1.0 - C2)*1e-9)*((z*dc)*9.81*3300)*0.06

    plt.plot((z)*dc_u,dv)
    plt.plot((z)*dc_u,dv2)

    plt.xlabel('Depth [m]')
    plt.ylabel('Decoupling viscous force [Pa]')
    plt.title('Decoupling viscous force')
    plt.grid()
    
    #hobson 
    u = 5.0 
    l = 0.1 
    el = np.linspace(0,300,300)
    c = 100 
    L = 10 
    vel = 0.5 * ((1+0.01)+(1-0.01)*np.tanh((el-c)/(L/4)))
    
    zt = 40e3 
    z0 = 100e3 
    zm = (zt+z0)/2
    zd = (z0-zm)
    z = np.linspace(0,100e3,1000)
    fu = np.zeros(len(z))    
    fu = 1- 0.5 * ((1+0)+(1)*np.tanh((z-zm)/(zd/2)))
    fu2 = np.zeros(len(z))
    fu2[z<zt]  = 1 
    fu2[z>=zt] = 1-(zt-z[z>=zt])/(zt-z0)
    vel = 0.5 * ((1+0.01)+(1-0.01)*np.tanh((z-z0)/(10e3/4)))
    return 0

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

@dataclass
class Phase:
    """
    Edif          : float -> energy activation of diffusion creep 
    Vdif          : flaot -> volume activation of diffusion creep 
    Edis          : float -> energy activation of dislocation creep 
    Vdis          : float -> volume activation of dislocation creep
    n             : float -> stress exponent ** NB: if you change this value, -> the Bdis must be changed accordingly
    Bdif          : float -> Pre-exponential factor of diffusion [1/Pa/s]
    Bdis          : float -> Pre-exponential factor of dislocation [1/Pa^n/s] 
    Cp            : float -> Constant heat capacity     [J/kg/K]
    k             : float -> constant heat conductivity [K/W/m]
    rho0          : float -> reference density & constant density [kg/m3]
    eta           : float -> constant viscosity 
    name_diffusion: str -> Viscosity diffusion law 
    name_capacity : str -> Heat capacity law 
    name_density  : str -> Density law -> Constant,PT 
    name_alpha    : str -> Thermal expansivity law {if constant alpha is ALWAYS equal to 3e-5}
    radio         : float -> Radiogenic heat production [W/m3] or [Pa/s]
    radio_flag    : float -> activation flag for the radiative conductivity 
    
    name_capacity_option: 
    
    
    
    
    
    
    
    name_density: 
    Constat      : Density is not depending on P,T and is equal to the input reference density 
    PT           : Density depends on pressure and temperature. The thermal expansivity is the same that is defined by name_alpha, and Kb is constant and equal to 130e9 Pa 
    
    name_alpha: 
    Constant      : thermal expansivity is constant 
    Mantle        : thermal expansivity data are the one for Fosterite as Groose and Afonso 2013, Richardson et al 2020 
    Oceanic_Crust : thermal expansivity data are assuming a basalt mode composition, as Groose and Afonso 2013 et al 2020 
    
    name_conductivity: 
    
    Constant      : thermal conductivity is constant 
    Mantle        : thermal conductivity data are the same for Fosterite as Groose and Afonso 2013, Richardoson et al 2020 and Korenaga 2016 
    Oceanic_Crust : thermal conductivity data are the same are computed assuming a basalt mode composition, as Groose and Afonso 2013, Richardson et al 2020
    
    -----
    Viscosity. 
    Constant -> constant viscosity
    -----
    
    name_diffusion: 
    NB: ' Diffusion creep imply that the internal loop for Stokes is deactivated, because the viscosity is not depending in any solution of the stokes equation' 
    
    1. 'Hirth_Dry_Olivine_diff' : Classical Hirth and Kolhstedt (2003) diffusion creep for anyhydrous mantle. B_dis is computed as such to consider
    a fixed grain size of 1 micro meter. 
    2. 'Van_Keken_diff'         : The original source is Karato, but the name is reflecting the primary source where I find used. The data was given in other
    units and I converted them to be similar to the Hirth formulation.  
    3. 'Hirth_Wet_Olivine_diff' : Hydrous diffusion creep from Hirth and Kohlstedt (2003) for wet olivine. B_dis is computed assuming a fixed grain size (similar to LaMEM Kaus et al 2016) 
    
    name_dislocation
    1. 'Hirth_Dry_Olivine_disl' : Hirth and Kohlstedt dry dislocation creep for olivine
    2. 'Van_Keken_disl'         : Dislocation creep flow law from Van Keken et al 2008  
    3. 'Hirth_Wet_Olivine_disl' : Hirth and Kohlstedt wet dislocation creep for olivine 
    
    
    ---
    Class for handling the input. I took the main arguments of the function that generate
    the PhaseDataBase. The idea is allowing costumisation of the phases. For the work that 
    I am doing for the kynematic of slab, is an overkill. But the hope is to generate a few 
    utilities that can be used elsewhere for other problem. In principle, 
    """ 
    
    name_diffusion:str     = 'Constant',
    Edif:float             = -1e23, 
    Vdif:float             = -1e23,
    Bdif:float             = -1e23, 
    name_dislocation:float = 'Constant',
    n:float                = -1e23,
    Edis:float             = -1e23,
    Vdis:float             = -1e23, 
    Bdis:float             = -1e23, 
    Cp:float               = 1250,
    k:float                = 3.0,
    rho0:float             = 3300,
    eta:float              = 1e20,
    name_capacity:str      = 'Constant',
    name_conductivity:str  = 'Constant',
    name_alpha:str         = 'Constant',
    name_density:str       = 'Constant',
    radio:float            = 0.0,
    radio_flag:float       = 0


if __name__ == '__main__':
    logistic_function_decoupling()