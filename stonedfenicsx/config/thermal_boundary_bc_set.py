# calculate the thermal structure of the ocean according to Richards et al., 2018 JGR: Solid Earth 
# we use a time- and space-centered Crank-Nicholson finite-difference scheme with a predictor-corrector step (Press et al., 1992)

# import all the constants and defined model setup parameters 
"""
This module is adapted from FieldStone (Van Zelst et al., 2023). It has been refactored to match the current code structure
and extended to handle temperature-dependent material properties via a fixed-point (Picard) iteration.

Context (from reviewer feedback in Van Zelst et al., revision):
    “Eqns 13–16 define the solution procedure for a linear problem (i.e., when ρ, Cp and k are not functions of T).
     You stated earlier you incorporate the nonlinear parameters into this 1D model and use them as boundary conditions.
     Please correct the description of the method used to obtain the 1D temperature profile for the non-linear case.” 
'' https://egusphere.copernicus.org/preprints/2022/egusphere-2022-768/egusphere-2022-768-AR1.pdf'' 


Non-linear solution procedure:
    When ρ(T,P), Cp(T) and/or k(T,P) depend on temperature (and possibly pressure), the 1D temperature profile is obtained
    through a fixed-point iteration at each time step. In practice:
        1) initialize material properties from the previous time step (or an initial guess);
        2) solve the Crank–Nicolson discretization for T using the current material properties;
        3) update material properties from the new temperature estimate;
        4) repeat steps (2)–(3) until convergence.

Time stepping and stability:
    With temperature-dependent coefficients, the Crank–Nicolson scheme is no longer unconditionally stable for arbitrarily
    large time steps. Stability is ensured by using sufficiently small time steps. For the parameter ranges targeted here,
    the fixed-point iteration typically converges in 2–3 iterations per time step, and becomes robust after the first step.

Convergence criterion:
    Convergence is monitored using the relative change in temperature between successive iterations at the same time step,
    which is a commonly used stopping criterion in kinematic thermal models. The best alternative is computing the effective 
    energy conservation residuum, but the amount of work required, is not exactly paying off in accuracy. 

"""

# modules
from stonedfenicsx.config.phase_db import PhaseDataBase,compute_thermal_properties,density,heat_capacity
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.numerical_control import CtrlTemperatureBC,NumericalControls,IOControls
from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.create_mesh.create_mesh import dict_surf
import h5py
import numpy as np
from numpy.typing import NDArray
from numba import njit
import mpi4py 
import psutil as pst
from pathlib import Path
from stonedfenicsx.utils import timing_function
from scipy.special import erf as erf_sc

_NAME_H5_FILE_TMP = 'temporary_file.h5'


def save_data_set(f:h5py.File,buf:any,name:str)->None:
    """Write a dataset to an open HDF5 file, overwriting if it already exists.

    Args:
        f (h5py.File): Open HDF5 file handle (must be writable).
        buf (any): Data to write — any type accepted by h5py.create_dataset.
        name (str): Full HDF5 path for the dataset (e.g. 'left_bc/temp_save').
    """

    if name in f:
        del f[name]

    f.create_dataset(name,data=buf)

def check_race_condition(ioctrl:IOControls)->bool:
    """Check whether the temporary HDF5 cache file is held open by another process.

    Iterates over all running processes via psutil. Returns True if any process
    has the cache file open, False otherwise. Processes that have died or are
    inaccessible between iteration and inspection are silently skipped.

    Args:
        ioctrl (IOControls): I/O control object providing path_cached_information.

    Returns:
        bool: True if the file is currently open by another process, False if safe to write.
    """

    path_cached = ioctrl.path_cached_information

    path_h5 = path_cached/_NAME_H5_FILE_TMP
    
    race = False
    
    for proc in pst.process_iter(['pid', 'name']):
        try:
            for f in proc.open_files():
                if Path(f.path).resolve() == Path(path_h5):
                    race = True
        except (pst.NoSuchProcess,
                pst.AccessDenied,
                pst.ZombieProcess):
            pass

    return race
# --- 
@njit
def _compute_lithostatic_pressure(
                                 nz: int,
                                 ph: NDArray[np.int32],          # (nz,) phase id per level (or per cell)
                                 g: float,                       # m/s^2, vertical component
                                 dz: float,                      # m
                                 temp: NDArray[np.float64],         # (nz,) K
                                 pdb: "PhaseDataBase",
                                )  -> NDArray[np.float64]:
    """
    Compute a 1D lithostatic pressure profile with pressure-dependent properties.

    The lithostatic pressure is obtained by vertically integrating the weight of the
    overburden:

        dP/dz = rho(P, T, phase) * g_z

    Because density (and possibly other properties) may depend on pressure, the
    profile is computed with a fixed-point (Picard) nonlinear iteration: starting
    from an initial pressure guess, evaluate rho(P, T), integrate to update P(z),
    and repeat until convergence. The relative tollerance is hardcoded to be 1e-6

    Parameters
    ----------
    nz : int
        Number of grid points in the vertical direction.
    ph : NDArray[np.int32]
        Phase identifier array along the 1D column (length ``nz``). Used to select
        material laws/properties from ``pdb``.
    g : float
        Vertical component of gravitational acceleration (m/s^2). Use sign
        consistently with your z-axis convention.
    T : NDArray[np.float64]
        Temperature profile along the column (K), length ``nz``.
    pdb : object
        Material database/dataset providing density and other thermodynamic/elastic
        properties as functions of phase, temperature and pressure.

    NB: all parameters are already in dimensionless (scaled) units when called from the solver.
    
    Returns
    -------
    lit_p : NDArray[np.float64]
        Lithostatic pressure profile (Pa), length ``nz``.

    Notes
    -----
    - The boundary condition is typically ``P(z_surface) = 0`` (or atmospheric),
      and pressure increases downward.
    - Convergence is usually checked with a norm on successive pressure iterates,
      e.g. ``max(|P_new - P_old|) / max(P_new, eps)``.
    - If ``rho`` depends strongly on pressure, under-relaxation may be required.

    """

    #compute lithostatic pressure:
    lit_p_o =  np.zeros((nz))
    lit_p   = np.zeros((nz))
    res = 1.0
    while res>1e-6:
        for i in range(0,nz):
            
            rho = density(pdb,temp[i],lit_p_o[i],ph[i])
            rhog = rho*g
            if i == 0:
                lit_p[i] = 0.0
            else:
                lit_p[i] = lit_p[i-1]+rhog*dz
        
        res =  np.linalg.norm(lit_p-lit_p_o,2)/np.linalg.norm(lit_p+lit_p_o,2)
        lit_p_o[:]=lit_p
    return lit_p

#-----------------------------------------------------------------------------------------
@njit
def compute_cp_k_rho(ph   : NDArray[np.int32]
                      ,pdb : PhaseDataBase
                      ,temp : NDArray[np.float64]
                      ,pres  : NDArray[np.float64])->tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    
    """Compute the thermal material properties 
    Function that compute all the three material properties that are required for solving the energy equation. 
    Input: 
        ph : phase vector 
        pdb : Material phase structure
        temp  : temperature vector
        pres   : pressure vector 
    Returns:
        cp,rho,k => vector containing density, heat capacity, and conductivities that are computed using pressure and temperature vector
    """
    cp = np.zeros(len(temp),dtype=np.float64)
    k = np.zeros(len(temp),dtype=np.float64)
    rho = np.zeros(len(temp),dtype=np.float64)
    
    for jj,t_i in enumerate(temp):
        cp[jj], rho[jj], k[jj] = compute_thermal_properties(pdb,t_i,pres[jj],ph[jj])

    return cp,k,rho

#-----------------------------------------------------------------------------------------
@njit
def build_coefficient_matrix(pdb:PhaseDataBase,
                             ph:NDArray[np.int32],
                             temp_old:NDArray[np.float64],
                             temp_guess:NDArray[np.float64],
                             temp_pr:NDArray[np.float64],
                             step:int,
                             lit_p:NDArray[np.float64],
                             temp_min:np.float64,
                             temp_max:np.float64,
                             nz:int,
                             dt:np.float64,
                             dz_m:np.float64)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Assemble the Crank-Nicolson finite-difference system for one predictor or corrector step.

    Implements the predictor-corrector scheme of Press et al. (1992) extended to
    temperature-dependent material properties. Material properties (ρ, Cp, k) are
    evaluated at the guess temperature (predictor, step=0) or as the arithmetic mean
    of guess and predicted temperatures (corrector, step=1).

    The boundary conditions are Dirichlet: T=temp_min at node 0, T=temp_max at node nz-1.
    Interior nodes follow the standard CN discretization including a mass-matrix
    correction term (b_vct) that accounts for the change in ρCp between time levels,
    and a radiogenic heat source term.

    Args:
        pdb (PhaseDataBase): Phase material database (jitclass).
        ph (NDArray[np.int32]): Phase index per node, shape (nz,). 0-based.
        temp_old (NDArray[np.float64]): Temperature at the previous time level, shape (nz,).
        temp_guess (NDArray[np.float64]): Current Picard iterate (best guess for T^{n+1}), shape (nz,).
        temp_pr (NDArray[np.float64]): Predictor temperature (output of step=0 solve), shape (nz,).
                                       Only used when step=1.
        step (int): 0 for predictor pass, 1 for corrector pass.
        lit_p (NDArray[np.float64]): Lithostatic pressure profile, shape (nz,).
        temp_min (float): Dirichlet temperature at the top boundary (node 0).
        temp_max (float): Dirichlet temperature at the bottom boundary (node nz-1).
        nz (int): Number of grid nodes.
        dt (float): Time step (dimensionless).
        dz_m (float): Uniform node spacing (dimensionless).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            mass_matrix — tridiagonal system matrix, shape (nz, nz).
            d_vct       — right-hand side vector, shape (nz,).
    """

    mass_matrix = np.zeros((nz, nz),dtype=np.float64)     # pre-allocate mass_matrix array


    a_vct  = np.zeros((nz),dtype=np.float64)
    d_vct  = np.zeros((nz),dtype=np.float64)     # pre-allocate D column vector
    q_vct  = np.zeros((nz),dtype=np.float64)
    b_vct  = np.zeros((nz),dtype=np.float64)
    
    if step == 0:

        # predictor step

        # m = n
        # Compute the current material property with the guess temperature
        cp_m,k_m,rho_m=compute_cp_k_rho(ph=ph
                                        ,pdb=pdb
                                        ,temp=temp_guess
                                        ,pres=lit_p)

    elif step == 1:

        # corrector step

        # m = n+1/2
        # Compute the temperature with the guess and with the predicted temperature 
        cp_m0,k_m0,rho_m0=compute_cp_k_rho(ph=ph
                                           ,pdb=pdb
                                           ,temp = temp_guess
                                           ,pres = lit_p)

        cp_m1,k_m1,rho_m1=compute_cp_k_rho(ph=ph
                                           ,pdb=pdb
                                           ,temp = temp_pr
                                           ,pres = lit_p)
        cp_m = (cp_m1+cp_m0)/2

        k_m              = (k_m0+k_m1)/2

        rho_m        = (rho_m0+rho_m1)/2
    
    k_m_m = (k_m[1:]+k_m[:-1])/2
    
    a_vct[:] = dt / (rho_m[:] * cp_m[:] * (2.0 * dz_m))
    
    for i in range(0,nz):


        if (i == 0):

            # boundary condition at the top 

            mass_matrix[i,i] = 1.

            d_vct[i]   = temp_min

        elif (i == nz-1):

            # boundary condition at the bottom

            mass_matrix[i,i] = 1.

            d_vct[i]   = temp_max

        else:

            mass_matrix[i,i+1] = -a_vct[i] * ( k_m_m[i]  / dz_m)
            mass_matrix[i,i] = 1. + a_vct[i] * (  k_m_m[i] + ( k_m_m[i-1]))/dz_m
            mass_matrix[i,i-1] = -a_vct[i] * ( k_m_m[i-1]  / dz_m)

            q_vct[i] = (1/dz_m) * (k_m_m[i]*temp_old[i+1] - (k_m_m[i-1]+k_m_m[i])*temp_old[i]+k_m_m[i-1]*temp_old[i-1])     
            rho_a = density(pdb, temp_old[i], lit_p[i], ph[i])
            cp_a  = heat_capacity(pdb, temp_old[i], ph[i])      
            if step == 0:
                rho_b = density(pdb,temp_guess[i],lit_p[i],ph[i])
                cp_b = heat_capacity(pdb,temp_guess[i],ph[i])

                # b_vct - predictor step 
                b_vct[i] = -temp_old[i] * ( rho_a * cp_a - rho_b * cp_b) / (rho_b * cp_b)       
            elif step == 1:
                rho_b = density(pdb,temp_pr[i],lit_p[i],ph[i])
                cp_b = heat_capacity(pdb,temp_pr[i],ph[i])      
                # b_vct - corrector step        
                b_vct[i] = - ((temp_pr[i] + temp_old[i]) * ( rho_b*cp_b - rho_a*cp_a ) / ( rho_b*cp_b + rho_a*cp_a))        
            d_vct[i] = temp_old[i] + a_vct[i] * q_vct[i]+ b_vct[i] + \
                dt * pdb.radiogenic_heat[ph[i]]/rho_m[i]/cp_m[i]
                    
    return mass_matrix,d_vct
# --- 
def fill_phase_properties(g_input:GeomInput,
                          z:NDArray[np.float64],
                          left_right:bool)->NDArray[np.int32]:
    """Assign a phase index to each node of a 1D vertical column.

    For the left (subducting) boundary the column contains oceanic crust
    above ocr and subducting-plate mantle below. For the right (overriding)
    boundary it contains upper crust, optionally lower crust, lithospheric
    mantle, and asthenospheric wedge, in order of increasing depth.

    The returned indices are 0-based (i.e. dict_surf values minus 1) to match
    the PhaseDataBase array indexing convention.

    Args:
        g_input (GeomInput): Geometric input parameters (depths in dimensionless units).
        z (NDArray[np.float64]): Depth coordinate vector, shape (nz,), increasing downward.
        left_right (bool): True → left (subducting) boundary; False → right (overriding) boundary.

    Returns:
        NDArray[np.int32]: Phase index per node, shape (nz,), 0-based.
    """
    ph = np.zeros([len(z)],dtype=np.int32)
    if left_right:
        ph[z<g_input.ocr] = dict_surf['oceanic_crust']
        ph[z>=g_input.ocr] = dict_surf['sub_plate']
    else:
        if g_input.lc != 0.0:
            ph[z<g_input.cr*(1-g_input.lc)] = dict_surf['upper_crust']
            ph[(z>=g_input.cr*(1-g_input.lc))
               & (z<g_input.cr)] = dict_surf['lower_crust']
        elif g_input.lc == 0.0:
            ph[z<g_input.cr] = dict_surf['upper_crust']
        ph[(z>=g_input.cr) &  (z<g_input.lit_mt+g_input.cr)] = dict_surf['overriding_lm']
        ph[(z>=g_input.lit_mt+g_input.cr)] = dict_surf['wedge']
  
    return ph-1

def compute_half_space_cooling_model_analytical(ctrl_tbc:CtrlTemperatureBC,
                                                z:NDArray[np.float64])->CtrlTemperatureBC:
    """Compute the left boundary temperature using the half-space cooling analytical solution.

    Used for the Van Keken benchmark suite where constant material properties allow
    the exact erf solution:

        T(z, t) = T_top + (T_max - T_top) * erf( z / (2 * sqrt(κ * t)) )

    where κ = k / (ρ Cp) is the thermal diffusivity computed from the fixed benchmark
    values stored in ctrl_tbc (not from the phase database).

    Writes the result directly into ctrl_tbc.temperature_1d and ctrl_tbc.z.

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control object. Must have scalar k, rho, cp,
                                      slab_age, temp_top, temp_max set (benchmark values).
        z (NDArray[np.float64]): Depth coordinate vector, shape (nz,), positive downward.

    Returns:
        CtrlTemperatureBC: Updated ctrl_tbc with temperature_1d and z filled.
    """
    cp    = ctrl_tbc.cp
    k     = ctrl_tbc.k
    rho   = ctrl_tbc.rho
    kappa = k/rho/cp
    t     = ctrl_tbc.slab_age
    temperature_bc = ctrl_tbc.temp_top+(ctrl_tbc.temp_max-ctrl_tbc.temp_top) * erf_sc(z /2 /np.sqrt(t * kappa))
    ctrl_tbc.z[:] = -z[:]
    ctrl_tbc.temperature_1d = temperature_bc
    return ctrl_tbc

def initialise_geometry_1d(ctrl_tbc:CtrlTemperatureBC
                           ,g_input:GeomInput
                           ,left_right:bool)->tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Build the 1D depth grid and phase array for a boundary column.

    For the left boundary the grid spans [0, (nz-1)*dz] with uniform spacing
    ctrl_tbc.dz, so that np.diff(z).mean() == ctrl_tbc.dz exactly.
    For the right boundary the grid spans [0, g_input.lab_d] (lithosphere-
    asthenosphere boundary depth) with nz nodes.

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control (provides nz, dz).
        g_input (GeomInput): Geometric input (provides lab_d for right BC).
        left_right (bool): True → left (subducting) boundary; False → right (overriding).

    Returns:
        tuple[NDArray[np.int32], NDArray[np.float64]]:
            ph — 0-based phase index per node, shape (nz,).
            z  — depth coordinate vector, shape (nz,), positive downward.
    """

    if left_right:
        z = np.linspace(0,(ctrl_tbc.nz-1)*ctrl_tbc.dz,ctrl_tbc.nz)
    else:
        z = np.linspace(0,g_input.lab_d,ctrl_tbc.nz)
    
    ph = fill_phase_properties(g_input=g_input
                               ,z=z
                               ,left_right=left_right)
    
    
    return ph, z
# --- 
def solve_temperature_1d_bc(ctrl_tbc:CtrlTemperatureBC
                            ,g_input:GeomInput
                            ,pdb:PhaseDataBase
                            ,z:NDArray[np.float64]
                            ,ph:NDArray[np.int32]
                            ,g:float
                            ,left_right:bool):

    """Advance the 1D thermal diffusion equation in time using Crank-Nicolson with Picard iteration.

    At each time step a predictor-corrector cycle is run (build_coefficient_matrix
    with step=0 then step=1), followed by a fixed-point (Picard) iteration that
    updates material properties and the lithostatic pressure until the relative
    change in temperature falls below 1e-6 or 10 iterations are reached.

    The initial temperature is either isothermal at temp_max (oceanic / left BC)
    or a linear geotherm from temp_top to temp_max over lab_d (continental right BC).

    The loop runs while t[time-1] < end_time AND time < nt, where
    end_time = ctrl_tbc.end_time (left) or ctrl_tbc.right_age * 1.01 (right).

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control (nz, nt, dt, dz, temp_max,
                                      temp_top, end_time, right_age, right_boundary).
        g_input (GeomInput): Geometric input (lab_d for right BC initial condition).
        pdb (PhaseDataBase): Phase material database (jitclass).
        z (NDArray[np.float64]): Depth coordinate vector, shape (nz,).
        ph (NDArray[np.int32]): Phase index per node, shape (nz,). 0-based.
        g (float): Gravitational acceleration (dimensionless scaled).
        left_right (bool): True → left (subducting) boundary; False → right (overriding).

    Raises:
        ValueError: If a NaN is detected in the Picard residual.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            t           — time vector, shape (nt,). Entries beyond the last completed
                          step remain zero.
            temperature — temperature field, shape (nt, nz). Row 0 is the initial
                          condition; rows beyond the last step are zero.
    """


    temp_max = ctrl_tbc.temp_max
    temp_min = ctrl_tbc.temp_top
    nz = ctrl_tbc.nz
    nt = ctrl_tbc.nt
    dt = ctrl_tbc.dt
    dz = ctrl_tbc.dz
    if left_right or ctrl_tbc.right_boundary == 'Oceanic':
        temp_old = np.ones((len(z)),dtype=np.float64) * temp_max
    else:
        # Initial guess linear
        gr = (temp_max-temp_min)/(g_input.lab_d)
        temp_old = temp_min + gr * z

    lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_old,pdb)

    temperature       = np.zeros([nt,nz],dtype=np.float64)
    temp_pr = np.zeros([nz],dtype=np.float64)
    t = np.zeros([ctrl_tbc.nt],dtype=np.float64)
    temperature[0,:] = temp_old
    
    if left_right: 
        end_time = ctrl_tbc.end_time
    else:
        end_time = ctrl_tbc.right_age * (1 + 0.01)

    time = 1
    
    while t[time-1] < end_time and time < nt:

        t[time] = t[time-1] + dt
        temp_old_tl = temperature[time-1,:] # temperature at the previous time step
        temp_guess  = temp_old_tl.copy()
        
        it = 0
        res = 1.0
        # Fixed point iteration
        while res > 1e-6 and it < 10:
            for step in range(2):
                    mass_matrix,d_vct = build_coefficient_matrix(pdb=pdb
                                                                 ,ph=ph
                                                                 ,temp_old = temp_old_tl
                                                                 ,temp_guess=temp_guess
                                                                 ,temp_pr=temp_pr
                                                                 ,step=step
                                                                 ,lit_p=lit_p
                                                                 ,temp_min=temp_min
                                                                 ,temp_max=temp_max
                                                                 ,dt = ctrl_tbc.dt
                                                                 ,dz_m = ctrl_tbc.dz
                                                                 ,nz = ctrl_tbc.nz)
                    temp_new = np.linalg.solve(mass_matrix, d_vct)

                    if step == 0:
                        temp_pr = temp_new

            res = np.linalg.norm(temp_new-temp_guess,2)/np.linalg.norm(temp_new+temp_guess,2)
            

            if np.isnan(res):
                raise ValueError("NaN detected in the residual")
            
            
            lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_new,pdb)

            temp_guess = temp_new * 0.8 + temp_guess * (0.2)
            
            it += 1
        lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_new,pdb)

        temperature[time,:] = temp_new
        time = time + 1

    return t, temperature

# ---
@timing_function
def compute_thermal_boundary(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,sc:Scal
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput
                                 ,save_data:bool
                                 ,left_right:bool)->CtrlTemperatureBC:
    """Compute the 1D thermal boundary condition and optionally cache the result to HDF5.

    Orchestrates the full boundary computation:
      1. Build the depth grid and phase array (initialise_geometry_1d).
      2. If van_keken benchmark mode and left BC: use the analytical erf solution.
      3. Otherwise: run the Crank-Nicolson + Picard time-stepping solver.
      4. Extract the temperature profile at the target age (slab_age or right_age).
      5. If save_data=True and MPI rank==0 and no race condition: write the full
         time-temperature history and material property arrays to the HDF5 cache.

    The HDF5 groups are 'left_bc' and 'right_bc'. Within each group:
      - temp_save, time_2d          : 2D fields for post-processing (physical units)
      - phases                       : unique phase ids present in the column
      - phase_properties_<i>/...     : per-phase thermal coefficient arrays
      - data_2_load/temperature_1d   : profile at target age (left BC)
      - data_2_load/temp_1d_right    : profile at target age (right BC)
      - data_2_load/temperature      : full time-temperature array (scaled)
      - data_2_load/time_v           : time vector (scaled)
      - data_2_load/z                : depth vector (scaled)

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control (already scaled).
        ctrl (NumericalControls): Numerical controls (provides g).
        ioctrl (IOControls): I/O controls (provides cache path).
        sc (Scal): Scaling object (used for unit conversion when saving).
        pdb (PhaseDataBase): Phase material database (jitclass, already scaled).
        g_input (GeomInput): Geometric input (already scaled).
        save_data (bool): Whether to write the result to the HDF5 cache.
        left_right (bool): True → left (subducting) boundary; False → right (overriding).

    Returns:
        tuple[CtrlTemperatureBC, GeomInput]:
            ctrl_tbc — updated with temperature_1d / temp_1d_right and z / z_right.
            g_input  — unchanged (returned for call-site symmetry).
    """
    # Spell out the structure
    g = ctrl.g

    # Initialise the geometry and the phases 
    ph, z  = initialise_geometry_1d(ctrl_tbc=ctrl_tbc
                           ,g_input=g_input
                           ,left_right=left_right)

    if g_input.van_keken and left_right: 
        ctrl_tbc = compute_half_space_cooling_model_analytical(ctrl_tbc,z)
        return ctrl_tbc,g_input

    time_v, temperature = solve_temperature_1d_bc(ctrl_tbc=ctrl_tbc
                            ,pdb=pdb
                            ,g_input=g_input
                            ,g=g
                            ,z=z
                            ,ph=ph
                            ,left_right=left_right)

    # Current age index
    if left_right:
        current_age_index = np.where(time_v >= ctrl_tbc.slab_age)[0][0]
        ctrl_tbc.temperature_1d = temperature[current_age_index,:]
        ctrl_tbc.z[:] = - z
        if ctrl_tbc.constant == 0:
            ctrl_tbc.temperature_2d_field[:,:] = temperature
        else: 
            ctrl_tbc.temperature_2d_field = None
    else:
        current_age_index = np.where(time_v >= ctrl_tbc.right_age)[0][0]
        ctrl_tbc.temp_1d_right[:] = temperature[current_age_index,:]
        ctrl_tbc.z_right = - z
    
    rank = mpi4py.MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        race_condition = check_race_condition(ioctrl)
        if race_condition: 
            print('    The file is opened for an other process, skip the save.')
        if save_data and not race_condition:
            ttime,zz = np.meshgrid(time_v,z)
            ttime = ttime*sc.time/365.25/60/60/24/1e6
            ttime = ttime[:,1::]
            zz    = zz[:,1::]
            zz    = zz*sc.length/1e3
            temp_save = temperature.T[:,1::]*sc.temp - 273.15
            id_phase = np.unique(ph)
            path_cache = ioctrl.path_cached_information
            path_h5_file = path_cache/_NAME_H5_FILE_TMP
            with h5py.File(path_h5_file,'a') as f:
                if left_right:
                    grp = 'left_bc'
                else:
                    grp = 'right_bc'
                save_data_set(f,temp_save,f'{grp}/temp_save')
                save_data_set(f,ttime,f'{grp}/time_2d')
                save_data_set(f,id_phase,f'{grp}/phases')
                for _,i in enumerate(id_phase):
                    array_cp = [pdb.c0[i],
                                pdb.c1[i],
                                pdb.c2[i],
                                pdb.c3[i],
                                pdb.c4[i],
                                pdb.c5[i]]
                    array_k = [pdb.k0[i],
                               pdb.k_a[i],
                               pdb.k_b[i],
                               pdb.k_c[i],
                               pdb.k_d[i],
                               pdb.k_e[i],
                               pdb.k_f[i],
                               pdb.radiative_conductivity[i]]
                    array_rho = [pdb.rho0[i],
                                 pdb.alpha0[i],
                                 pdb.alpha1[i],
                                 pdb.alpha2[i],
                                 pdb.kb[i]]
                    hr = pdb.radiogenic_heat[i]
                    name = f'{grp}/phase_properties_{i}'
                    save_data_set(f,hr,f'{name}/hr')
                    save_data_set(f,array_rho,f'{name}/rho_prop')
                    save_data_set(f,array_cp,f'{name}/array_cp')
                    save_data_set(f,array_k,f'{name}/array_cond')
                name = f'{grp}/data_2_load'
                if left_right:
                    save_data_set(f,ctrl_tbc.temperature_1d,name=f'{name}/temperature_1d')
                    save_data_set(f,ctrl_tbc.slab_age,name=f'{name}/slab_age')
                    save_data_set(f,ctrl_tbc.z,name=f'{name}/z')


                else: 
                    save_data_set(f,ctrl_tbc.temp_1d_right,name=f'{name}/temp_1d_right')
                    save_data_set(f,ctrl_tbc.right_age,name=f'{name}/right_age')
                    save_data_set(f,ctrl_tbc.z_right,name=f'{name}/z')

                    
                save_data_set(f,temperature,name=f'{name}/temperature')
                save_data_set(f,time_v,name=f'{name}/time_v')
                        
                print('             temporary data base is saved...')

    return ctrl_tbc,g_input

def check_material_property(f,pdb,ph_id,main_grp)->bool:
    """Verify that the material properties in the HDF5 cache match the current phase database.

    Compares the stored thermal coefficient arrays (conductivity, heat capacity,
    density parameters, radiogenic heat) against the values in pdb for each phase
    in ph_id. Raises ValueError on any mismatch so the caller knows the cache must
    be recomputed before reuse.

    Args:
        f (h5py.File): Open HDF5 file handle (read mode).
        pdb (PhaseDataBase): Current phase material database (jitclass, already scaled).
        ph_id (list[int]): List of phase ids (1-based dict_surf values) present in the column.
        main_grp (str): HDF5 group name — 'left_bc' or 'right_bc'.

    Raises:
        ValueError: If any stored coefficient array differs from the current pdb
                    by more than 1e-12 in absolute value. Signals that the cached
                    thermal history was computed with different material properties.

    Returns:
        bool: True if all arrays match (always True when no exception is raised).
    """
    check_array_f = lambda x,y :  np.any(np.abs(x - y) > 1e-12)
    
    flag = True
    warnings = 0
    for i in ph_id: 
        # Check conductivity: 
        i = i - 1
        array = f[f'{main_grp}/phase_properties_{i}/array_cond']
        array_k = [pdb.k0[i],
                   pdb.k_a[i],
                   pdb.k_b[i],
                   pdb.k_c[i],
                   pdb.k_d[i],
                   pdb.k_e[i],
                   pdb.k_f[i],
                   pdb.radiative_conductivity[i]]
        if check_array_f(array[:], array_k[:]):
            raise ValueError(f'Conductivity mismatch for phase {i}: cached thermal history must be recomputed.')
        array = f[f'{main_grp}/phase_properties_{i}/array_cp']
        array_cp = [pdb.c0[i],
                    pdb.c1[i],
                    pdb.c2[i],
                    pdb.c3[i],
                    pdb.c4[i],
                    pdb.c5[i]]
        if check_array_f(array[:], array_cp[:]):
            raise ValueError(f'Heat capacity mismatch for phase {i}: cached thermal history must be recomputed.')
        array = f[f'{main_grp}/phase_properties_{i}/rho_prop']
        array_rho = [pdb.rho0[i],
                     pdb.alpha0[i],
                     pdb.alpha1[i],
                     pdb.alpha2[i],
                     pdb.kb[i]]
        if check_array_f(array[:], array_rho[:]):
            raise ValueError(f'Density mismatch for phase {i}: cached thermal history must be recomputed.')
        diff = f[f'{main_grp}/phase_properties_{i}/hr'] - pdb.radiogenic_heat[i]
        if diff != 0.0:
            raise ValueError(f'Radiogenic heat mismatch for phase {i}: cached thermal history must be recomputed.')

    return True


# ---
def read_temporary_file(ctrl_tbc:CtrlTemperatureBC
                        ,ctrl:NumericalControls
                        ,ioctrl:IOControls
                        ,g_input:GeomInput
                        ,pdb:PhaseDataBase
                        ,sc:Scal
                        ,left_right)->tuple[CtrlTemperatureBC,GeomInput]:
    """Load a previously computed boundary temperature profile from the HDF5 cache.

    If the cache file does not exist, falls back to computing it from scratch via
    compute_thermal_boundary (with save_data=True so the result is cached for next time).

    If the file exists, first calls check_material_property to ensure the stored
    coefficients match the current pdb. On success, reads the full temperature array,
    finds the time index closest to the target age (slab_age or right_age), and
    loads that profile into ctrl_tbc.

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control (already scaled).
        ctrl (NumericalControls): Numerical controls (passed through to compute_thermal_boundary
                                  if the cache is missing).
        ioctrl (IOControls): I/O controls (provides path_cached_information).
        g_input (GeomInput): Geometric input (already scaled).
        pdb (PhaseDataBase): Phase material database (jitclass, already scaled).
        sc (Scal): Scaling object (passed through if recomputation is needed).
        left_right (bool): True → left (subducting) boundary; False → right (overriding).

    Returns:
        tuple[CtrlTemperatureBC, GeomInput]:
            ctrl_tbc — updated with temperature_1d / temp_1d_right and z / z_right.
            g_input  — unchanged (returned for call-site symmetry).
    """
    
    path_cache = ioctrl.path_cached_information
    path_h5_file = path_cache/_NAME_H5_FILE_TMP
    redo = False

    if not path_h5_file.exists():
        print('Temporary file has not been created yet, running the cooling model')
        redo = True
    else:
        print('Temporary file exists')
        if left_right:
            ph_id = [dict_surf['sub_plate'],dict_surf['oceanic_crust']]
        else: 
            ph_id = [dict_surf['upper_crust'],dict_surf['lower_crust'],dict_surf['overriding_lm'],dict_surf['wedge']]
        
        with h5py.File(path_h5_file,'r') as f:
            
            if left_right:
                main_grp = 'left_bc'
                age = ctrl_tbc.slab_age
            else:
                main_grp = 'right_bc'
                age = ctrl_tbc.right_age
            
            if main_grp  not in f:
                print(f'{main_grp} is not yet in the temporary file')
                redo = True
            else:
        

                time_v = f[f'{main_grp}/data_2_load/time_v'][:]
                temperature = f[f'{main_grp}/data_2_load/temperature'][:]
                z = f[f'{main_grp}/data_2_load/z'][:]

                flag = check_material_property(f,pdb,ph_id,main_grp)
                current_age_index = np.where(time_v >= age)[0][0]

                if flag:
                    if not left_right: 
                        ctrl_tbc.temp_1d_right[:] = temperature[current_age_index,:]
                        ctrl_tbc.z_right = z
                        g_input.lab_d = np.abs(np.min(z))
                    else:
                        ctrl_tbc.temperature_1d[:] = temperature[current_age_index,:]
                        ctrl_tbc.z = z
        
    if redo:
        ctrl_tbc, g_input = compute_thermal_boundary(ctrl_tbc=ctrl_tbc
                                                 ,ctrl=ctrl
                                                 ,ioctrl=ioctrl
                                                 ,sc=sc
                                                 ,pdb=pdb
                                                 ,g_input=g_input
                                                 ,save_data=True
                                                 ,left_right=left_right)
 
    return ctrl_tbc,g_input



# --- 
@timing_function
def configure_thermal_bc(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,sc:Scal
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput
                                 ,left_right:bool)->CtrlTemperatureBC:
    """Dispatch thermal boundary computation: recompute from scratch or load from cache.

    If ctrl_tbc.recalculate is set, runs the full Crank-Nicolson solver and saves
    the result to the HDF5 cache. Otherwise attempts to load from the existing cache
    via read_temporary_file (which itself falls back to recomputation if the file is
    absent or the stored material properties do not match).

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control (already scaled).
        ctrl (NumericalControls): Numerical controls.
        ioctrl (IOControls): I/O controls (cache path).
        sc (Scal): Scaling object.
        pdb (PhaseDataBase): Phase material database (jitclass, already scaled).
        g_input (GeomInput): Geometric input (already scaled).
        left_right (bool): True → left (subducting) boundary; False → right (overriding).

    Returns:
        tuple[CtrlTemperatureBC, GeomInput]:
            ctrl_tbc — updated with the boundary temperature profile and depth vector.
            g_input  — unchanged (returned for call-site symmetry).
    """
    
    if ctrl_tbc.recalculate:
        ctrl_tbc, g_input = compute_thermal_boundary(ctrl_tbc=ctrl_tbc
                                                     ,ctrl=ctrl
                                                     ,ioctrl=ioctrl
                                                     ,sc=sc
                                                     ,pdb=pdb
                                                     ,g_input=g_input
                                                     ,save_data=True
                                                     ,left_right=left_right)
    else:
        ctrl_tbc,g_input = read_temporary_file(ctrl_tbc=ctrl_tbc
                                               ,ctrl=ctrl
                                               ,pdb=pdb
                                               ,g_input=g_input
                                               ,ioctrl=ioctrl
                                               ,sc=sc
                                               ,left_right=left_right)
    return ctrl_tbc,g_input

# --- # 
def configure_boundary_condition(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,sc:Scal
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput)->CtrlTemperatureBC:
    
    """Configure both the left and right thermal boundary conditions.

    Public entry point for the thermal boundary setup. Calls configure_thermal_bc
    twice — first for the left (subducting slab) boundary, then for the right
    (overriding plate) boundary — each of which either recomputes or loads from cache.

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal BC control (already scaled).
        ctrl (NumericalControls): Numerical controls.
        ioctrl (IOControls): I/O controls (cache path).
        sc (Scal): Scaling object.
        pdb (PhaseDataBase): Phase material database (jitclass, already scaled).
        g_input (GeomInput): Geometric input (already scaled).

    Returns:
        tuple[CtrlTemperatureBC, GeomInput]:
            ctrl_tbc — updated with temperature_1d (left) and temp_1d_right (right).
            g_input  — unchanged (returned for call-site symmetry).
    """
    # Configure left boundary condition
    ctrl_tbc,g_input = configure_thermal_bc(ctrl_tbc = ctrl_tbc
                         ,ctrl = ctrl
                         ,ioctrl=ioctrl
                         ,sc=sc
                         ,pdb=pdb
                         ,g_input=g_input
                         ,left_right=True)
    
    # Configure right boundary condition
    ctrl_tbc,g_input = configure_thermal_bc(ctrl_tbc = ctrl_tbc
                         ,ctrl = ctrl
                         ,ioctrl=ioctrl
                         ,sc=sc
                         ,pdb=pdb
                         ,g_input=g_input
                         ,left_right=False)
    
    return ctrl_tbc,g_input
# --- # 

def test_configure_boundary():
    from stonedfenicsx.config.simulation_config import configure_simulation
    from stonedfenicsx.config.input_parser import parse_input
    from stonedfenicsx.config.phase_db import Phase,PhInput

    # Find the main folder of the package
    pkg_root = Path(__file__)
    # Select the appropriate path for the input file
    input_file = Path(pkg_root.parents[2], "input.yaml")
    # parse the input file
    input_data, ph_in = parse_input(input_file)
    # Set the path of the tests
    path_save = pkg_root.parents[2] / "Results"
    test_name = "Mock_test"
    input_data.ctrl_io.test_name = test_name
    input_data.ctrl_io.path_save = path_save
    input_data.ctrl_tbc.slab_age = 100.0
    #
    ph_in.oceanic_crust.name_alpha = "Oceanic_crust"
    ph_in.oceanic_crust.name_capacity = "Oceanic_crust"
    ph_in.oceanic_crust.radiative_conductivity = 1
    ph_in.oceanic_crust.rho0 = 3300
    ph_in.oceanic_crust.name_conductivity = "Crust_Richards_2018"
    ph_in.oceanic_crust.name_density = "PT"
    ph_in.oceanic_crust.radiogenic_heat = 0.25e-6

    ph_in.subducting_plate_mantle.name_capacity = "Mantle_Bernard_Ar_199x_FO_FA"
    ph_in.subducting_plate_mantle.name_conductivity = "Mantle_Richards_2018"
    ph_in.subducting_plate_mantle.name_alpha = "Mantle"
    ph_in.subducting_plate_mantle.rho0 = 3300
    ph_in.subducting_plate_mantle.name_density = "PT"

    ph_in.wedge_mantle.name_capacity = "Mantle_Bernard_Ar_199x_FO_FA"
    ph_in.wedge_mantle.name_conductivity = "Mantle_Richards_2018"
    ph_in.wedge_mantle.name_alpha = "Mantle"
    ph_in.wedge_mantle.rho0 = 3300
    ph_in.wedge_mantle.name_density = "PT"
    ph_in.wedge_mantle.name_dislocation = "VK_Dislocation_creep"
    ph_in.wedge_mantle.name_diffusion = "VK_Diffusion_creep"
    
    ph_in.overriding_lower_crust.radiogenic_heat = 0.5e-6
    ph_in.overriding_upper_crust.radiogenic_heat = 1.0e-6

    
    ctrl_sim, _ , _, sc  = configure_simulation(ph_in, input_data)
    
    import matplotlib.pyplot as plt 
    
    fig = plt.figure()
    ax = fig.gca() 
    temp_plate = ctrl_sim.ctrl_tbc.temperature_1d.copy()*sc.temp-273.15 
    ax.plot(ctrl_sim.ctrl_tbc.temperature_1d*sc.temp-273.15, ctrl_sim.ctrl_tbc.z*sc.length/1e3,c='firebrick')   
    ax.plot(ctrl_sim.ctrl_tbc.temp_1d_right*sc.temp-273.15, ctrl_sim.ctrl_tbc.z_right*sc.length/1e3,c='forestgreen')  
    ctrl_tbc = compute_half_space_cooling_model_analytical(ctrl_sim.ctrl_tbc,np.abs(ctrl_sim.ctrl_tbc.z))
    ax.plot(ctrl_tbc.temperature_1d*sc.temp-273.15, ctrl_tbc.z*sc.length/1e3,c='cadetblue')   

    plt.show()
    fig = plt.figure()
    ax = fig.gca() 
    ax.plot(ctrl_tbc.temperature_1d*sc.temp-273.15-temp_plate, ctrl_tbc.z*sc.length/1e3,c='cadetblue')   
    
    
    
    return 0


def main():
    
    test_configure_boundary()
    
    return 0

if __name__ == '__main__':
    main()