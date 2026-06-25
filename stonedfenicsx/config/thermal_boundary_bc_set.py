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
from stonedfenicsx.material_property.phase_db import PhaseDataBase
from stonedfenicsx.material_property.compute_material_property import compute_thermal_properties
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.numerical_control import CtrlTemperatureBC,NumericalControls,IOControls
from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.create_mesh.create_mesh import dict_surf
import h5py 
import numpy as np 
from numba import njit


power       = np.power
exp         = np.exp
transpose   = np.transpose 
array       = np.array
full        = np.full
all         = np.all
diagonal    = np.diagonal
solve_banded= la.solve_banded


@njit
def _compute_lithostatic_pressure(
                                 nz: int,
                                 ph: NDArray[np.int32],          # (nz,) phase id per level (or per cell)
                                 g: float,                       # m/s^2, vertical component
                                 dz: float,                      # m
                                 T: NDArray[np.float64],         # (nz,) K
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
    ph : ndarray
        Phase identifier array along the 1D column (length ``nz``). Used to select
        material laws/properties from ``pdb``.
    g : float
        Vertical component of gravitational acceleration (m/s^2). Use sign
        consistently with your z-axis convention.
    T : ndarray
        Temperature profile along the column (K), length ``nz``.
    pdb : object
        Material database/dataset providing density and other thermodynamic/elastic
        properties as functions of phase, temperature and pressure.

    NB: the parameters are usually scaled within the numerical routines
    
    Returns
    -------
    lit_p : ndarray
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
    lit_p_o =  zeros((nz))
    lit_p   = zeros((nz))
    res = 1.0
    while res>1e-6:
        for i in range(0,nz):
            rho = density(pdb,T[i],lit_p_o[i],ph[i])
            rhog = rho*g
            if i == 0:
                lit_p[i] = 0.0
            else: 
                lit_p[i] = lit_p[i-1]+rhog*dz 
        
        res =  np.linalg.norm(lit_p-lit_p_o,2)/np.linalg.norm(lit_p+lit_p_o,2)
        lit_p_o[:]=lit_p 
    return lit_p
    
        
# we will solve the system Mx = d_vct with
# x = the temperature vector (our unknowns)
# mass_matrix = our matrix of coefficients in front of the temperature (known)
# d_vct = the right hand side vector (known)

# first, we need to build mass_matrix and d_vct
# mass_matrix and d_vct are different in the predictor and corrector step 
#-----------------------------------------------------------------------------------------
@njit
def _compute_Cp_k_rho(ph   : NDArray[np.float64]
                      ,pdb : PhaseDataBase
                      ,Cp  : NDArray[np.float64]
                      ,k   : NDArray[np.float64]
                      ,rho : NDArray[np.float64]
                      ,T   : NDArray[np.float64]
                      ,p   : NDArray[np.float64]
                      ,flagNL : int 
                      ,CValues: NDArray[np.float64]
                      ,ind_z: int)->tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    
    """Compute the thermal material properties 
    Function that compute all the three material properties that are required for solving the energy equation. 
    Input: 
        ph : phase vector 
        pdb : Material phase structure
        Cp  : heat capacity vector 
        k   : heat conductivity vector 
        rho : rho vector
        T   : temperature vector
        p   : pressure vector 
        ind_z: must be removed i don't remember what does 
    Returns:
        Cp,k,rho => vector containing density, heat capacity, and conductivities that are computed using pressure and temperature vector
    """
    
    
    it = len(T)
    
    for i in range(it):
        cp[i], rho[i], k[i] = compute_thermal_properties(pdb,T[i],p[i],ph[i])

    return cp,k,rho
#-----------------------------------------------------------------------------------------
@njit
def build_coefficient_matrix(a_vct:np.ndarray,
                             b_vct:np.ndarray,
                             d_vct:np.ndarray,
                             q_vct:np.ndarray,
                             mass_matrix:np.ndarray,
                             pdb:PhaseDataBase,
                             ph:np.ndarray,
                             temp_old:np.ndarray,
                             temp_guess:np.ndarray,
                             temp_pr:np.ndarray,
                             k_m:np.ndarray,
                             density_m:np.ndarray,
                             heat_capacity_m:np.ndarray,
                             step:int,
                             lit_p:np.ndarray,
                             temp_min:np.float64,
                             temp_max:np.float64,
                             nz:int,
                             dt:np.float64,
                             dz:np.float64)->tuple[np.ndarray,np.ndarray]:
    """Assembly the system of equation

    Args:
        a_vct (np.ndarray): _description_
        b_vct (np.ndarray): _description_
        d_vct (np.ndarray): _description_
        q_vct (np.ndarray): _description_
        mass_matrix (np.ndarray): _description_
        pdb (PhaseDataBase): _description_
        ph (np.ndarray): _description_
        temp_old (np.ndarray): _description_
        temp_guess (np.ndarray): _description_
        temp_pr (np.ndarray): _description_
        k_m (np.ndarray): _description_
        density_m (np.ndarray): _description_
        heat_capacity_m (np.ndarray): _description_
        step (int): _description_
        lit_p (np.ndarray): _description_
        temp_min (np.float64): _description_
        temp_max (np.float64): _description_
        nz (int): _description_
        dt (np.float64): _description_
        dz (np.float64): _description_

    Returns:
        tuple[np.ndarray,np.ndarray]: _description_
    """


    if step == 0:

        # predictor step

        # m = n

        heat_capacity_m,k_m,density_m=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,temp_guess,lit_p,NLflag,CVal,ind_z)

        dz_m               = full((nz), dz) # current assumption: incompressible

    elif step == 1: 

        # corrector step 

        # m = n+1/2

        Cp_m0,k_m0,rho_m0=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,temp_guess,lit_p,NLflag,CVal,ind_z)

        Cp_m1,k_m1,rho_m1=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,temp_guess,lit_p,NLflag,CVal,ind_z)

        heat_capacity_m = (Cp_m1+Cp_m0)/2

        k_m              = (k_m0+k_m1)/2

        density_m        = (rho_m0+rho_m1)/2

        dz_m           = (full((nz), dz) + full((nz), dz) ) / 2.  # current assumption: incompressible


    for i in range(0,nz):

        if i == 0: 

            start_loop = i

            end_loop   = i+1

        elif i == nz-1:

            start_loop = i-1

            end_loop   = i  

        else: 

            start_loop = i-1

            end_loop   = i+1  


        for j in range(start_loop,end_loop+1): 

        

            if j == 0:

                # calculate a_vct  

                a_vct[j] = (dt / ( density_m[j] * heat_capacity_m[j] * ( dz_m[j] + dz_m[j] ) ))

            elif j > 0:

                # calculate a_vct 

                a_vct[j] = (dt / ( density_m[j] * heat_capacity_m[j] * ( dz_m[j] + dz_m[j-1] ) ))  


            # ========== BUILD mass_matrix ========== - coefficient matrix 

            # ========== boundary conditions ==========

            if (i == 0 and j == 0): 

                # boundary condition at the top 

                mass_matrix[i,j] = 1. 

                d_vct[j]   = Ttop 

            elif (i == nz-1 and j == nz-1):

                # boundary condition at the bottom 

                mass_matrix[i,j] = 1. 

                d_vct[j]   = Tmax 

            else:

                # ========== BUILD mass_matrix ========== - coefficient matrix 

                if i - j == 1 and j < nz - 2:

                    # T_j+1 

                    if j > 0:

                        mass_matrix[i,j] = -a_vct[j] * ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j]

                    elif j == 0:

                        mass_matrix[i,j] = -a_vct[j] * ( (k_m[j] + k_m[j] ) / 2. ) / dz_m[j]

                elif i == j:

                    # diagonal: T_j

                    mass_matrix[i,j] = 1. + a_vct[j] * ( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] + ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1])

                elif i - j == -1 and j > 1 and j < nz:

                    # T_j-1

                    if j < nz - 1:

                        mass_matrix[i,j] = -a_vct[j] * ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1]

                    elif j == nz - 1:

                        mass_matrix[i, j] = -a_vct[j] * ((k_m[j] + k_m[j]) / 2.) / dz_m[j]


                # ========== BUILD d_vct ========== - right hand side vector 

                # d_vct consists of multiple components

                # we say d_vct = T + a_vct * q_vct + b_vct

                if j > 0 and j < nz-1:

                    q_vct[j] = ( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] )*temp_old[j+1] - (( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] ) + ( ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1] )) * temp_old[j] + ( ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1] )*temp_old[j-1]


                    # b_vct - correction that represents the second term on the right-hand side on the equation 
                    if NLflag == 0:
                        rho_A = density(pdb, temp_old[j], lit_p[j], ph[j])
                        Cp_A  = heat_capacity(pdb, temp_old[j], ph[j])
                    else: 
                        rho_A = lhs.rho
                        Cp_A = lhs.Cp
           
                    if step == 0:
                        if NLflag == 0:
                            rho_B = density(pdb,temp_old[j],lit_p[j],ph[j])
                            Cp_B = heat_capacity(pdb,temp_old[j],ph[j])
                        else: 
                            rho_B = lhs.rho 
                            Cp_B = lhs.Cp 
                        # b_vct - predictor step 
                        b_vct[j] = -temp_old[j] * ( rho_A * Cp_A - rho_B * Cp_B) / (rho_A * Cp_A)

                    elif step == 1: 
                        if NLflag == 0:
                            rho_B = density(pdb,temp_old[j],lit_p[j],ph[j])
                            Cp_B = heat_capacity(pdb,temp_old[j],ph[j])
                        else: 
                            rho_B = lhs.rho 
                            Cp_B = lhs.Cp 

                        # b_vct - corrector step 

                        b_vct[j] = - ((temp_pr[j] + temp_old[j]) * ( rho_B*Cp_B - rho_A*Cp_A ) / ( rho_B*Cp_A + rho_B*Cp_A))


                    d_vct[j] = temp_old[j] + a_vct[j] * q_vct[j] + b_vct[j]
    return mass_matrix,d_vct
# --- 
def fill_phase_properties(g_input:GeomInput,z:np.ndarray,left_right:bool,ph:np.ndarray)->np.ndarray:
    
    if left_right:
        ph[z<g_input.ocr] = dict_surf['oceanic_crust']
        ph[z>=g_input.ocr] = dict_surf['sub_plate']
    elif not left_right:
        if g_input.lc != 0.0:
            ph[z<g_input.cr*(1-g_input.lc)] = dict_surf['upper_crust']
            ph[(z>=g_input.cr*(1-g_input.lc))
               and (z<g_input.cr)] = dict_surf['lower_crust']
        elif g_input.lc == 0.0:
            ph[z<g_input.cr] = dict_surf['upper_crust']
        ph[(z>=g_input.cr) and  (z<g_input.lit_mt+g_input.cr)] = dict_surf['overriding_lm']
        ph[(z>=g_input.lit_mt+g_input.cr)] = dict_surf['wedge']
    
    return ph-1

#@njit 
def compute_half_space_cooling_model(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
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
    dt = ctrl_tbc.dt
    end_time = ctrl_tbc.end_time
    nz = ctrl_tbc.nz
    g = ctrl.g
    temp_max = ctrl_tbc.temp_max
    temp_min = ctrl_tbc.temp_top
    van_keken = g_input.van_keken
    theta_in = g_input.theta_in_slab
    
    nt = int(end_time / dt + 1)

    t = zeros((1,nt))

    z = np.linspace(0,(nz*dz),nz)

    if van_keken:
        from scipy import special
        print('         Computing the temperature field using'
              'the analytical half-space cooling model...')

        cp    = ctrl_tbc.cp
        k     = ctrl_tbc.k
        rho   = ctrl_tbc.rho
        kappa = k/rho/cp
        t     = ctrl_tbc.slab_age
        temperature_bc = temp_min+(temp_max-temp_min) * special.erf(z /2 /np.sqrt(t * kappa))
        ctrl_tbc.z[:] = -z[:]
        ctrl_tbc.temperature_1d = temperature_bc
    else:

        if theta_in != 0 and left_right:
            z    = np.linspace(0,(nz*dz)/np.cos(theta_in),nz)
            dz = np.diff(z)
            ctrl_tbc.dz = dz
            print('         The z vector of the left boundary condition has been'
                  'corrected with the initial slab angle.')

        ph   = np.zeros([nz],dtype = np.int32)

        ph = fill_phase_properties(g_input,z,left_right,ph)
    
        temp_old = np.ones((len(z)),dtype=np.float64) * temp_max
    
        lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_old,pdb)
    
        temperature       = np.zeros([nt,nz],dtype=np.float64)
        pressure          = np.zeros([nt,nz],dtype=np.float64)
        conductivity      = np.zeros([nt,nz],dtype=np.float64)
        capacity          = np.zeros([nt,nz],dtype=np.float64)
        density           = np.zeros([nt,nz],dtype=np.float64)

        temperature[0,:] = temp_old
        pressure[0,:] = lit_p


        k_m = np.zeros((nz),dtype=np.float64)
        heat_capacity_m = np.zeros((nz),dtype=np.float64)
        density_m = np.zeros((nz),dtype=np.float64)
        k_tmp = np.zeros((nz),dtype=np.float64)
        cp_tmp = np.zeros((nz),dtype=np.float64)
        rho_tmp = np.zeros((nz),dtype=np.float64)

        for time in range(1,nt):

            t[0,time] = t[0,time] + time*dt
            temp_old_tl = temperature[time-1,:] # temperature at the previous time step
            temp_guess  = temp_old_tl

            mass_matrix = np.zeros((nz, nz),dtype=np.float64)     # pre-allocate mass_matrix array


            a_vct  = np.zeros((nz),dtype=np.float64)
            d_vct  = np.zeros((nz),dtype=np.float64)     # pre-allocate D column vector
            q_vct  = np.zeros((nz),dtype=np.float64)
            b_vct  = np.zeros((nz),dtype=np.float64)

            temp_pr = np.zeros(nz,dtype=np.float64)

            it = 0
            res = 1.0
            # Fixed point iteration
            while res > 1e-6 and it < 10:
                for step in range(2):
                        mass_matrix,d_vct = build_coefficient_matrix(a_vct=a_vct
                                                                     ,b_vct=b_vct
                                                                     ,d_vct=d_vct
                                                                     ,q_vct=q_vct
                                                                     ,mass_matrix=mass_matrix
                                                                     ,pdb=pdb
                                                                     ,ph=ph
                                                                     ,temp_old = temp_old
                                                                     ,temp_guess=temp_guess
                                                                     ,temp_pr=temp_pr
                                                                     ,k_m=k_m
                                                                     ,density_m = density_m
                                                                     ,heat_capacity_m=heat_capacity_m
                                                                     ,step=step
                                                                     ,lit_p=lit_p
                                                                     ,temp_min=temp_min
                                                                     ,temp_max=temp_max
                                                                     ,dt = ctrl_tbc.dt
                                                                     ,dz = ctrl_tbc.dz
                                                                     ,nz = ctrl_tbc.nz)
                        temp_new = transpose(np.linalg.solve(mass_matrix, d_vct))    # solve Mx = d_vct for x

                        if step == 0:
                            temp_pr = temp_new

                res = np.linalg.norm(temp_new-temp_guess,2)/np.linalg.norm(temp_new+temp_guess,2)
                if np.isnan(res):
                    raise ValueError("NaN detected in the residual")
                temp_guess = temp_new * 0.8 + temp_guess * (0.2)
                lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_old,pdb)
                it += 1


            lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,temp_new,pdb)
            cp_tmp,k_tmp,rho_tmp = _compute_Cp_k_rho(ph,pdb,cp_tmp,k_tmp,rho_tmp,temp_new,lit_p,0)
            temperature[time,:] = temp_new
            pressure[time,:]    = lit_p
            capacity[time,:]    = cp_tmp
            conductivity[time,:] = k_tmp 
            density[time,:]      = rho_tmp



        # Current age index
        current_age_index = np.where(t[0] >= ctrl_tbc.slab_age)[0][0]
        ctrl_tbc.temperature_1d = temperature[current_age_index]
        ctrl_tbc.z[:] = - z
        ctrl_tbc.temperature_2d_field[:,:] = temperature


        if data_base: 
            

        TTime,ZZ = np.meshgrid(t[0],z)
        TTime = TTime*scal.T/365.25/60/60/24/1e6
        TTime = TTime[:,1::]
        ZZ    = ZZ[:,1::] 
        ZZ    = ZZ*scal.L/1e3
        Tem   = temperature.T[:,1::]*scal.Temp - 273.15
        Cp    = capacity.T[:,1::]*scal.Cp 
        k     = conductivity.T[:,1::]*scal.k 
        rho   = density.T[:,1::]*scal.rho 
        LP    = pressure.T[:,1::]*scal.stress/1e9



    return ctrl_tbc,g_input




def compute_initial_LHS(ctrl,lhs,scal,pdb,theta_in):
    

    
    if lhs.van_keken == 0 or lhs.non_linearities == 0 :
        lhs,t, temperature = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb,theta_in)
        from scipy.interpolate import RegularGridInterpolator
 
        t_re = np.linspace(lhs.c_age_var[0],lhs.c_age_var[1], num = lhs.LHS_var.shape[1])
        T,Z  = np.meshgrid(t_re,lhs.z,indexing='ij')
        TT,ZZ = np.meshgrid(t,lhs.z,indexing='ij')
        interp_func = RegularGridInterpolator((t[0], lhs.z), temperature)
        points_coarse = np.column_stack((T.ravel(), Z.ravel()))
        lhs.LHS_var = interp_func(points_coarse).reshape(len(t_re), len(lhs.z))
        lhs.t_res_vec = t_re 
    else:
        lhs,_,_ = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb,theta_in)
        

    return lhs


def update_age_lhs(ctrl,lhs,scal,pdb,theta_in):
    

    
    if lhs.van_keken == 0 or lhs.non_linearities == 0 :
        from scipy.interpolate import RegularGridInterpolator
 
        t_re = np.linspace(lhs.c_age_var[0],lhs.c_age_var[1], num = lhs.LHS_var.shape[1])
        T,Z  = np.meshgrid(t_re,lhs.z,indexing='ij')
        TT,ZZ = np.meshgrid(t,lhs.z,indexing='ij')
        interp_func = RegularGridInterpolator((t[0], lhs.z), temperature)
        points_coarse = np.column_stack((T.ravel(), Z.ravel()))
        lhs.LHS_var = interp_func(points_coarse).reshape(len(t_re), len(lhs.z))
        lhs.t_res_vec = t_re 
    else:
        lhs,_,_ = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb,theta_in)
        

    return lhs

def configure_right_boundary_condition(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput)->CtrlTemperatureBC:
    
    if ctrl_tbc.right_boundary == 'Continental':
        # Compute the continental boundary condition
        print('')
    elif ctrl_tbc.right_boundary == 'Oceanic': 
        if ctrl_tbc.right_age is None: 
            raise ValueError('Age of the right boundary condition has not been set.')
        # -> change the g_input 
        if g_input.lab_d != g_input.slab_tk:
            raise Warning('The lithosphere depth of the right boundary condition is not the same of the slab thickness. '
                          'The Geom_Input has been modified')

            g_input.lab_d = g_input.slab_tk
        
        if ctrl_tbc.recalculate: 
            # Compute half space cooling model 
            print('')
        else: 
            # read database -> if same data -> import from h5py -> and set up the boundary condition
            print('')
    
    
    
    
    return ctrl_tbc 
# --- # 

def configure_left_boundary_condition(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput)->CtrlTemperatureBC:
    
    if ctrl_tbc.recalculate:
        # Compute lhs 
        print('')
    elif not ctrl_tbc.recalculate: 
        # read cache database -> are the data of the involved the phase the same? 
        # if not -> compute and save 
        print('')
    
    
    return ctrl_tbc

# --- # 
def configure_boundary_condition(ctrl_tbc:CtrlTemperatureBC
                                 ,ctrl:NumericalControls
                                 ,ioctrl:IOControls
                                 ,pdb:PhaseDataBase
                                 ,g_input:GeomInput)->CtrlTemperatureBC:
    
    # Configure left boundary condition
    
    
    # Configure right boundary condition 
    
    
    
    
    
    
    
    return ctrl_tbc
# --- # 