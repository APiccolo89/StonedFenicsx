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
import pathlib 
from pathlib import Path
from stonedfenicsx.utils import timing_function
from scipy.special import erf as erf_sc

_NAME_H5_FILE_TMP = 'temporary_file.h5'


def save_data_set(f:h5py.File,buf:any,name:str)->None:
    """save the cached file

    Args:
        f (h5py.File): 
        buf (any): _description_
        name (str): _description_
    """

    if name in f:
        del f[name]

    f.create_dataset(name,data=buf)

def check_race_condition(ioctrl:IOControls)->bool:
    """Check if the file is opened by an other process 
    if not -> go and open

    Args:
        ioctrl (IOControls): _description_

    Returns:
        bool: _description_
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

    NB: the parameters are usually sced within the numerical routines
    
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
def compute_cp_k_rho(ph   : NDArray[np.float64]
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
    cp = np.zeros([len(temp)],dtype=np.float64)
    k = np.zeros([len(temp)],dtype=np.float64)
    rho = np.zeros([len(temp)],dtype=np.float64)
    
    for jj in enumerate(temp):
        i = jj[0]
        cp[i], rho[i], k[i] = compute_thermal_properties(pdb,jj[1],pres[i],ph[i])

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
    """Assembly the system of equation

    Args:
        a_vct (NDArray[np.float64]): _description_
        b_vct (NDArray[np.float64]): _description_
        d_vct (NDArray[np.float64]): _description_
        q_vct (NDArray[np.float64]): _description_
        mass_matrix (NDArray[np.float64]): _description_
        pdb (PhaseDataBase): _description_
        ph (NDArray[np.int32]): _description_
        temp_old (NDArray[np.float64]): _description_
        temp_guess (NDArray[np.float64]): _description_
        temp_pr (NDArray[np.float64]): _description_
        k_m (NDArray[np.float64]): _description_
        density_m (NDArray[np.float64]): _description_
        heat_capacity_m (NDArray[np.float64]): _description_
        step (int): _description_
        lit_p (NDArray[np.float64]): _description_
        temp_min (np.float64): _description_
        temp_max (np.float64): _description_
        nz (int): _description_
        dt (np.float64): _description_
        dz (np.float64): _description_

    Returns:
        tuple[NDArray[np.float64],NDArray[np.float64]]: _description_
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


            if i > 0 and i < nz-1:

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
    """_summary_

    Args:
        g_input (GeomInput): _description_
        z (NDArray[np.float64]): _description_
        left_right (bool): _description_
        ph (NDArray[np.int32]): _description_

    Returns:
        NDArray[np.int32]: _description_
    """
    ph = np.zeros([len(z)],dtype=np.int32)
    if left_right:
        ph[z<g_input.ocr] = dict_surf['oceanic_crust']
        ph[z>=g_input.ocr] = dict_surf['sub_plate']
    elif not left_right:
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
    """_summary_

    Args:
        ctrl_tbc (CtrlTemperatureBC): _description_

    Returns:
        _type_: _description_
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
                           ,left_right:bool)->tuple[NDArray[np.float64]
                                                    ,NDArray[np.int32]
                                                    ,float]:
    """_summary_

    Args:
        ctrl_tbc (CtrlTemperatureBC): _description_
        g_input (GeomInput): _description_
        left_right (bool): _description_

    Returns:
        tuple[NDArray[np.float64] ,NDArray[np.int32] ,float]: _description_
    """

    if left_right:
        z = np.linspace(0,(ctrl_tbc.nz*ctrl_tbc.dz),ctrl_tbc.nz)
    else:
        z = np.linspace(0,g_input.lab_d,ctrl_tbc.nz)
    
    dz = np.nanmean(np.diff(z))

    ph = fill_phase_properties(g_input=g_input
                               ,z=z
                               ,left_right=left_right)
    
    
    return ph, z, dz
# --- 
def solve_temperature_1d_bc(ctrl_tbc:CtrlTemperatureBC
                            ,g_input:GeomInput
                            ,pdb:PhaseDataBase
                            ,z:NDArray[np.float64]
                            ,ph:NDArray[np.int32]
                            ,dz:float
                            ,g:float
                            ,left_right:bool):

    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """


    temp_max = ctrl_tbc.temp_max
    temp_min = ctrl_tbc.temp_top
    nz = ctrl_tbc.nz
    nt = ctrl_tbc.nt
    dt = ctrl_tbc.dt
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
    while t[time-1] < end_time:

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
                    temp_new = np.transpose(np.linalg.solve(mass_matrix, d_vct))    # solve Mx = d_vct for x

                    if step == 0:
                        temp_pr = temp_new

            res = np.linalg.norm(temp_new-temp_guess,2)/np.linalg.norm(temp_new+temp_guess,2)
            lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_new,pdb)

            if np.isnan(res):
                raise ValueError("NaN detected in the residual")
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
    """_summary_

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal boundary condition
        ctrl (NumericalControls): Numerical Control
        ioctrl (IOControls): Input/Output control
        pdb (PhaseDataBase): Phase database
        g_input (GeomInput): input geometry
        save_data (bool): save h5py file with all the data 
        left_right (bool): True:left , False: Right 

    Returns:
        CtrlTemperatureBC: Updated boundary control 
    """
    # Spell out the structure
    dz = ctrl_tbc.dz
    g = ctrl.g
    nt = ctrl_tbc.nt
    # Initialise the geometry and the phases 
    t = np.zeros((nt))

    ph, z ,dz = initialise_geometry_1d(ctrl_tbc=ctrl_tbc
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
                            ,dz=dz
                            ,left_right=left_right)

    # Current age index
    if left_right:
        current_age_index = np.where(time_v >= ctrl_tbc.slab_age)[0][0]
        ctrl_tbc.temperature_1d = temperature[current_age_index]
        ctrl_tbc.z[:] = - z
        if ctrl_tbc.constant == 0:
            ctrl_tbc.temperature_2d_field[:,:] = temperature
        else: 
            ctrl_tbc.temperature_2d_field = None
    else:
        current_age_index = np.where(time_v >= ctrl_tbc.right_age)[0][0]
        ctrl_tbc.temp_1d_right[:] = temperature[current_age_index]
        ctrl_tbc.z_right = - z
    
    rank = mpi4py.MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        race_condition = check_race_condition(ioctrl)
        if race_condition: 
            print('    The file is opened for an other process, skip the save.')
        if save_data and not race_condition and left_right:
            ttime,zz = np.meshgrid(t[0],z)
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
                for i in enumerate(id_phase):
                    array_cp = [pdb.c0[i[1]],
                                pdb.c1[i[1]],
                                pdb.c2[i[1]],
                                pdb.c3[i[1]],
                                pdb.c4[i[1]],
                                pdb.c5[i[1]]]
                    array_k = [pdb.k0[i[1]],
                               pdb.k_a[i[1]],
                               pdb.k_b[i[1]],
                               pdb.k_c[i[1]],
                               pdb.k_d[i[1]],
                               pdb.k_e[i[1]],
                               pdb.k_f[i[1]],
                               pdb.radiative_conductivity[i[1]]]
                    array_rho = [pdb.rho0[i[1]],
                                 pdb.alpha0[i[1]],
                                 pdb.alpha1[i[1]],
                                 pdb.alpha2[i[1]],
                                 pdb.kb[i[1]]]
                    hr = pdb.radiogenic_heat[i[1]]
                    name = f'{grp}/phase_properties_{i[1]}'
                    save_data_set(f,hr,f'{name}/hr')
                    save_data_set(f,array_rho,f'{name}/rho_prop')
                    save_data_set(f,array_cp,f'{name}/array_cp')
                    save_data_set(f,array_k,f'{name}/array_cond')
                    name = f'{grp}/data_2_load'
                    if left_right:
                        save_data_set(f,ctrl_tbc.temperature_1d,name=f'{name}/temperature_1d')
                        save_data_set(f,ctrl_tbc.slab_age,name=f'{name}/slab_age')

                    else: 
                        save_data_set(f,ctrl_tbc.temp_1d_right,name=f'{name}/temp_1d_right')
                        save_data_set(f,ctrl_tbc.right_age,name=f'{name}/right_age')
                    
                    save_data_set(f,temperature,name=f'{name}/temperature')
                    save_data_set(f,time_v,name=f'{name}/time_v')
                    save_data_set(f,ctrl_tbc.z,name=f'{name}/z')


                        
                    print('             temporary data base is saved...')

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
    """_summary_

    Args:
        ctrl_tbc (CtrlTemperatureBC): _description_
        ctrl (NumericalControls): _description_
        ioctrl (IOControls): _description_
        sc (Scal): _description_
        pdb (PhaseDataBase): _description_
        g_input (GeomInput): _description_
        left_right (bool): _description_

    Returns:
        CtrlTemperatureBC: _description_
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
    elif not ctrl_tbc.recalculate: 
        # Read the data base.# PLACE HOLDER
        pass
    
    
    return ctrl_tbc,g_input

# --- # 
def configure_boundary_condition(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,sc:Scal
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput)->CtrlTemperatureBC:
    
    """_summary_

    Returns:
        _type_: _description_
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

    ph_in.oceanic_crust.name_alpha = "Oceanic_crust"
    ph_in.oceanic_crust.name_capacity = "Oceanic_crust"
    ph_in.oceanic_crust.radiative_conductivity = 1
    ph_in.oceanic_crust.rho0 = 2800
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

    
    configure_simulation(ph_in, input_data)
    
    
    return 0


def main():
    
    test_configure_boundary()
    
    return 0

if __name__ == '__main__':
    main()