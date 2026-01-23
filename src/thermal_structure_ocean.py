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

from .package_import import *
# modules
from .compute_material_property import heat_capacity,heat_conductivity,density
from .phase_db import PhaseDataBase
from .utils import timing_function, print_ph
from .compute_material_property import compute_thermal_properties
from .scal import Scal
from .numerical_control import NumericalControls,ctrl_LHS

start       = timing.time()
zeros       = np.zeros
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
    
        
# we will solve the system Mx = D with
# x = the temperature vector (our unknowns)
# M = our matrix of coefficients in front of the temperature (known)
# D = the right hand side vector (known)

# first, we need to build M and D
# M and D are different in the predictor and corrector step 
#-----------------------------------------------------------------------------------------
# Melting parametrisation
def compute_initial_geotherm(Tp:float
                             ,z: NDArray[np.float64]
                             ,sc: Scal
                             ,option_melt: int = 1 ) -> NDArray[np.float64]:
    """Compute the initial geotherm
    Compute the initial geotherm assuming isentropic melting in case the adiabatic flag is active. 
    Args:
        Tp (float): Potential temperature (i.e., the prescribed maximum temperature of the model)
        z (NDArray[np.float64]): Depth vecor
        sc (Scaling): Scaling structure
        option_melt (int, optional): Melting flag (== Adiabatic flag). Defaults to 1.

    Returns:
        NDArray[np.float64]: Geotherm: if adiabatic is not active, a vector of the shape of z, with a constant temperature; if the adiabatic flag is active
        and isentropic adiabatic geotherm that accounts the melting processes following katz et al 2003, and Shorle et al 2014. 
    
    Note: the initial geotherm assume in anycase linear thermal coefficient. This simplify the calculation, and the literature references that have been used for computing
    the melting adiabat are not accounting for the non-linearities. 
    """
    
    if option_melt == 1: 
        
        from Melting_parametrisation import lithologies,compute_Fraction,runge_kutta_algori,SL,dSL,rcpx,compute_T_cpx
        
        TS       = lambda  P :SL(1085.7, 132.9,-5.1,P)

        dTS      = lambda  P :dSL(132.9,-5.1,P)

        TLL      = lambda  P :SL(1475,80,-3.2,P)

        dTLL     = lambda P  :dSL(80,-3.2,P)

        TL       = lambda  P :SL(1780,45.0,-2.0,P)

        dTL      = lambda  P :dSL(45.0,-2.0,P)

        Mcpx     = lambda  fr,P :rcpx(0.5,0.08,fr,P)

        TCpx     = lambda  fr,P :compute_T_cpx(Mcpx(fr,P),TS(P),TLL(P))

        #Ts = None,Tcpx=None,TLl=None,TL = None,MCpx = None,dTs = None, dTL = None, dTLl = None
        lhz = lithologies(Ts=TS,Tcpx=TCpx,TLl=TLL,TL=TL,MCpx=Mcpx,dTs=dTS,dTL=dTL,dTLl=dTLL)

        P = (z*sc.L * 9.81 * lhz.rho) 
        
        T = np.zeros(len(P))
        
        P = np.flip(P)
        
        Tp = Tp*sc.Temp

        T_start = (Tp) + 18/1e9*P[0]
        
        T[0] = T_start

        for i in range(len(P)-1):
            dP = P[i+1]- P[i]
            F0  = compute_Fraction(T[i],TL(P[i]),TLL(P[i]),TS(P[i]),TCpx(lhz.fr,P[i]),Mcpx(lhz.fr,P[i]))  
            dT = runge_kutta_algori(lhz,P[i],T[i],dP,F0,None,Tp)
            T[i+1] = T[i] + dT 
            if F0>0.0 and F0 <= lhz.Mcpx(lhz.fr,P[i]):
                c = 'r'
                alpha = 0.8 
            elif F0 > lhz.Mcpx(lhz.fr,P[i]):
                c = 'forestgreen'
                alpha = 0.9
            else:
                c='k'
                alpha = 0.3
        
        Told = np.flip(T)/sc.Temp
    else: 
        
        Told = np.ones(len(z))*Tp
        
    return Told 

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
    if flagNL == 0:
        for i in range(it):
            Cp[i], rho[i], k[i] = compute_thermal_properties(pdb,T[i],p[i],ph[i])
    else: 
        Cp[:] = CValues[0]; k[:] = CValues[1]; rho[:] = CValues[2] 

    return Cp,k,rho
#-----------------------------------------------------------------------------------------
@njit
def build_coefficient_matrix(A,
                             B,
                             D,
                             Q,
                             M,
                             pdb,
                             ph,
                             ctrl,
                             lhs,
                             TO,
                             TG,
                             TPr,
                             k_m,
                             density_m,
                             heat_capacity_m,
                             step,
                             lit_p,
                             sc,
                             ind_z,
                             Ttop,
                             Tmax):

    nz         = lhs.nz 
    dt         = lhs.dt 
    dz         = lhs.dz
    NLflag     = lhs.non_linearities
    CVal       = np.array([lhs.Cp,lhs.k,lhs.rho],dtype = np.float64)

    
    if step == 0:

        # predictor step

        # m = n

        heat_capacity_m,k_m,density_m=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,TG,lit_p,NLflag,CVal,ind_z)

        dz_m               = full((nz), dz) # current assumption: incompressible

    elif step == 1: 

        # corrector step 

        # m = n+1/2

        Cp_m0,k_m0,rho_m0=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,TG,lit_p,NLflag,CVal,ind_z)

        Cp_m1,k_m1,rho_m1=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,TG,lit_p,NLflag,CVal,ind_z)

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

                # calculate A  

                A[j] = (dt / ( density_m[j] * heat_capacity_m[j] * ( dz_m[j] + dz_m[j] ) ))

            elif j > 0:

                # calculate A 

                A[j] = (dt / ( density_m[j] * heat_capacity_m[j] * ( dz_m[j] + dz_m[j-1] ) ))  


            # ========== BUILD M ========== - coefficient matrix 

            # ========== boundary conditions ==========

            if (i == 0 and j == 0): 

                # boundary condition at the top 

                M[i,j] = 1. 

                D[j]   = Ttop 

            elif (i == nz-1 and j == nz-1):

                # boundary condition at the bottom 

                M[i,j] = 1. 

                D[j]   = Tmax 

            else:

                # ========== BUILD M ========== - coefficient matrix 

                if i - j == 1 and j < nz - 2:

                    # T_j+1 

                    if j > 0:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j]

                    elif j == 0:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j] ) / 2. ) / dz_m[j]

                elif i == j:

                    # diagonal: T_j

                    M[i,j] = 1. + A[j] * ( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] + ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1])

                elif i - j == -1 and j > 1 and j < nz:

                    # T_j-1

                    if j < nz - 1:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1]

                    elif j == nz - 1:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j] ) / 2. ) / dz_m[j]


                # ========== BUILD D ========== - right hand side vector 

                # D consists of multiple components

                # we say D = T + A * Q + B

                if j > 0 and j < nz-1:

                    Q[j] = ( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] )*TO[j+1] - (( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] ) + ( ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1] )) * TO[j] + ( ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1] )*TO[j-1]


                    # B - correction that represents the second term on the right-hand side on the equation 
                    if NLflag == 0:
                        rho_A =  density(pdb,TO[j],lit_p[j],ph[j])
                        Cp_A  = heat_capacity(pdb,TO[j],ph[j])
                    else: 
                        rho_A = lhs.rho 
                        Cp_A  = lhs.Cp
           
                    if step == 0:
                        if NLflag == 0:
                            rho_B = density(pdb,TO[j],lit_p[j],ph[j])
                            Cp_B = heat_capacity(pdb,TO[j],ph[j])
                        else: 
                            rho_B = lhs.rho 
                            Cp_B = lhs.Cp 
                        # B - predictor step 
                        B[j] = -TO[j] * ( rho_A * Cp_A - rho_B * Cp_B) / (rho_A * Cp_A)

                    elif step == 1: 
                        if NLflag == 0:
                            rho_B = density(pdb,TO[j],lit_p[j],ph[j])
                            Cp_B = heat_capacity(pdb,TO[j],ph[j])
                        else: 
                            rho_B = lhs.rho 
                            Cp_B = lhs.Cp 

                        # B - corrector step 

                        B[j] = - ((TPr[j] + TO[j]) * ( rho_B*Cp_B - rho_A*Cp_A ) / ( rho_B*Cp_A + rho_B*Cp_A))


                    D[j] = TO[j] + A[j] * Q[j] + B[j]
    return M,D

#-----------------------------------------------------------------------------------------
#@njit 
def compute_ocean_plate_temperature(ctrl:NumericalControls
                                    ,lhs:ctrl_LHS
                                    ,scal:Scal
                                    ,pdb:PhaseDataBase)->tuple[ctrl_LHS, NDArray[np.float64], NDArray[np.float64]]:
    
    """
    
    """

    # Spell out the structure 
    dz          = lhs.dz # m 
    dt          = lhs.dt # year
    end_time    = lhs.end_time    # million years 
    nz          = lhs.nz                 # size of the spatial array (in km; max depth, assuming dz = 1km) 
    depth_melt = lhs.depth_melt
    alpha_g    = lhs.alpha_g
    g          = ctrl.g
    Tmax       = ctrl.Tmax
    Ttop       = ctrl.Ttop
    van_keken_option = lhs.van_keken

    
    option_1D_solve = lhs.option_1D_solve 

    nt = int(end_time / dt + 1)

    t = zeros((1,nt))


    # ========== initial conditions ========== 


    z    = np.arange(0,nz*dz,dz)
    ph   = np.zeros([nz],dtype = np.int32) # I assume that everything is mantle 
    ph[z<6000/scal.L] = np.int32(1)

    if lhs.van_keken == 1: 
        from scipy import special
        Cp    = 1250/scal.Cp
        k     = 3.0/scal.k
        rho   = 3300/scal.rho
        kappa = k/rho/Cp 
        t     = 50 * scal.scale_Myr2sec/scal.T
        T_lhs = Ttop+(Tmax-Ttop) * special.erf(z /2 /np.sqrt(t * kappa))
        lhs.z[:] = -z[:] 
        lhs.LHS[:] = T_lhs[:]
        P = ctrl.g * z * rho
        if ctrl.adiabatic_heating == 1: 
            T_lhs = T_lhs * np.exp((3e-5/rho/Cp) * P )

        return lhs,[],[]


    

    
    Told = compute_initial_geotherm(ctrl.Tmax,z,scal,ctrl.adiabatic_heating)
    
    

    
    lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,Told,pdb)
    


    temperature       = zeros([nt,nz],dtype=np.float64)
    pressure          = zeros([nt,nz],dtype=np.float64)
    conductivity      = zeros([nt,nz],dtype=np.float64)
    capacity          = zeros([nt,nz],dtype=np.float64)
    density           = zeros([nt,nz],dtype=np.float64)

    T_1t = zeros([nt,nz],dtype=np.float64)
    temperature[0,:] = Told
    pressure[0,:] = lit_p
    # print(Told)


    k_m = zeros((nz),dtype=np.float64)
    heat_capacity_m = zeros((nz),dtype=np.float64)
    density_m = zeros((nz),dtype=np.float64)
    k_tmp = zeros((nz),dtype=np.float64)
    Cp_tmp = zeros((nz),dtype=np.float64)
    rho_tmp = zeros((nz),dtype=np.float64)

    Ttop = ctrl.Ttop
    Tmax = np.max(Told)

    ind_z_lit = np.where(z>=120e3/scal.L)[0][0] # I force the temperature to be T_max at this dept -> Otherwise I have a temperature jump, and i do not know how much
    for time in range(1,nt):
          
        t[0,time] = t[0,time] + time*dt
        TO = temperature[time-1,:] # temperature at the previous time step
        TG  = TO

        M = zeros((nz, nz),dtype=np.float64)     # pre-allocate M array


        A               = zeros((nz),dtype=np.float64)

        D               = zeros((nz),dtype=np.float64)     # pre-allocate D column vector

        Q               = zeros((nz),dtype=np.float64)

        B               = zeros((nz),dtype=np.float64)

        TPr = np.zeros(nz,dtype=np.float64)

        it = 0 
        res = 1.0
        while res > 1e-6 and it < 10:
            for step in range(2):   


                    M,D = build_coefficient_matrix(A,B,D,Q,M,pdb,ph,ctrl,lhs,TO,TG,TPr,k_m,density_m,heat_capacity_m,step,lit_p,scal,ind_z_lit,Ttop,Tmax)
                    # ========== Solve system of equations using numpy.linalg.solve ==========

                    if option_1D_solve == 1:
                        # brute force solve
                        Tnew = transpose(np.linalg.solve(M, D))    # solve Mx = D for x
                    elif option_1D_solve == 2:    
                        # make ordered banded diagonal matrix MD from M 
                        upper = 1
                        lower = 1
                        n = M.shape[1]
                        assert(all(M.shape ==(n,n)))

                        ab = zeros((2*n-1, n))

                        for i in range(n):
                            ab[i,(n-1)-i:] = diagonal(M,(n-1)-i)

                        for i in range(n-1): 
                            ab[(2*n-2)-i,:i+1] = diagonal(M,i-(n-1))

                        mid_row_inx = int(ab.shape[0]/2)
                        upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
                        upper_rows.reverse()
                        upper_rows.append(mid_row_inx)
                        lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
                        keep_rows = upper_rows+lower_rows
                        ab = ab[keep_rows,:]

                        Tnew = transpose(solve_banded((1,1),ab, D))

                    if step == 0:
                        TPr = Tnew 
                    elif step == 1 and it == 0:
                        T_1t[time,:] = Tnew
            res = np.linalg.norm(Tnew-TG,2)/np.linalg.norm(Tnew+TG,2)
            if np.isnan(res) == True:
                print("NaN detected in the residual")
                sys.exit(1)
            TG = Tnew * 0.8 + TG * (0.2) 
            lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,Told,pdb)
            it += 1

            # end for-loop predictor-corrector step 
        #print_ph('Time %d, it %d'%(time,it))
        lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,Told,pdb)
        Cp_tmp,k_tmp,rho_tmp = _compute_Cp_k_rho(ph,pdb,Cp_tmp,k_tmp,rho_tmp,Tnew,lit_p,lhs.non_linearities,[lhs.Cp,lhs.k,lhs.rho],0)
        temperature[time,:] = Tnew
        pressure[time,:]    = lit_p
        capacity[time,:]    = Cp_tmp
        conductivity[time,:] = k_tmp 
        density[time,:]      = rho_tmp
        
    # -> Save the actual LHS 
    # Correction 
    #temperature[:,-z<-120e3/scal.L] = ctrl.Tmax

    # Current age index
    current_age_index = np.where(t[0] >= lhs.c_age_plate)[0][0]
    lhs.LHS[:] = temperature[current_age_index]
    lhs.z[:] = - z

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



    return lhs, t, temperature




@timing_function
def compute_initial_LHS(ctrl,lhs,scal,pdb):
    
    if lhs.van_keken == 0 or lhs.non_linearities == 0 :
        lhs,t, temperature = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb)
    else:
        lhs,_,_ = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb)
        return lhs
    
    
    
    from scipy.interpolate import RegularGridInterpolator
 
    t_re = np.linspace(lhs.c_age_var[0],lhs.c_age_var[1], num = lhs.LHS_var.shape[1])
    T,Z  = np.meshgrid(t_re,lhs.z,indexing='ij')
    TT,ZZ = np.meshgrid(t,lhs.z,indexing='ij')
    interp_func = RegularGridInterpolator((t[0], lhs.z), temperature)
    points_coarse = np.column_stack((T.ravel(), Z.ravel()))
    lhs.LHS_var = interp_func(points_coarse).reshape(len(t_re), len(lhs.z))
    lhs.t_res_vec = t_re 
        
    
    return lhs
#-----------------------------------------------------------------------------------------



    #print("time: %.3f s" % (timing.time() - start))
    # print(Tnew)

    # ========== save thermal structure ==========
    #if os.path.isdir('databases') == False:
    #    os.makedirs('databases')

    #fname = 'databases/slab_k{0}_Cp{1}_rho{2}'.format(option_k,option_Cp,option_rho)
    #np.savez('%s.npz'%fname,temperature=temperature,t=t,z=z)
    
    
    """

    import cmcrameri as cmc
    
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "serif"
    
    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,Tem, levels=[0,100,200,300,500,600,700,800,900,1000,1100,1200,1300,1400], cmap='cmc.lipari')
    b = ax0.contour(TTime, -ZZ, Tem,levels=[100,200,300,500,600,700,800,900,1000,1100,1200,1300],colors='k')
    
    ax0.set_xlim(0,130)
    ax0.set_ylim(-130,0.0)

    plt.colorbar(a, label=r'T, $[^{\circ}C]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    plt.savefig('Temp_plate.png')

    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,Cp, levels=20, cmap='cmc.lipari')
    plt.colorbar(a, label=r'Cp, $[J/K/kg]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('Cp_plate.png')

    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,k,  levels=40, cmap='cmc.nuuk')


    plt.colorbar(a, label=r'k, $[W/K/m]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('Condu_plate.png')

    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,rho, levels=10, cmap='cmc.lipari')

    plt.colorbar(a, label=r'$\rho$, $[kg/m^3]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('rho_plate.png')
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,LP, levels=10, cmap='cmc.lipari')

    plt.colorbar(a, label=r'$P$, $[GPa]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('Pressure_plate.png')
    



    """