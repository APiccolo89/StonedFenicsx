
# modules
from stonedfenicsx.package_import import *
from .phase_db import PhaseDataBase
from stonedfenicsx.scal import Scal
# ---------------------------------------------------------------------------------
@dataclass
class Functions_material_properties_global():
    """Function containing the fem.function per each of the parameter of material properties
    Member variables:
    k0: constant conductivity
    fr: radiogenic flag
    k_a, k_b, k_c, k_d, k_e, k_f: conductivity parameters
    C0, C1, C2, C3, C4, C5: heat capacity parameters
    rho0, alpha0, alpha1, alpha2, Kb: density parameters and alpha parameters
    option_rho: flag to choose the density formulation
    radio: radiogenic heating
    Tref, R, A, B, T_A, x_A, T_B, x_B: parameters for the radiative conductivity    
    """
    # Heat Conductivity properties
    k0 : fem.Function = None 
    fr : fem.Function = None
    k_a: fem.Function = None
    k_b: fem.Function = None
    k_c: fem.Function = None
    k_d: fem.Function = None
    k_e: fem.Function = None
    k_f: fem.Function = None
    # Heat Capacity properties
    C0 : fem.Function = None
    C1 : fem.Function = None
    C2 : fem.Function = None
    C3 : fem.Function = None
    C4 : fem.Function = None
    C5 : fem.Function = None
    # Density properties
    rho0    : fem.Function = None
    alpha0  : fem.Function = None
    alpha1  : fem.Function = None
    alpha2  : fem.Function = None
    Kb      : fem.Function = None
    option_rho : fem.Function = None
    # Radiogenic heating properties
    radio   : fem.Function = None
    # Other properties 
    Tref    : float = 0.0
    R       : float = 8.3145
    A       : float = 0.0
    B       : float = 0.0
    T_A     : float = 0.0
    x_A     : float = 0.0
    T_B     : float = 0.0
    x_B     : float = 0.0
# ---------------------------------------------------------------------------------
def populate_material_properties_thermal(CP:Functions_material_properties_global
                                         ,pdb:PhaseDataBase
                                         ,phase:fem.Function)->Functions_material_properties_global:

    """    Initialize all thermal properties as FEniCS functions.
    Args:        
        CP (Functions_material_properties_global): class of fem.function that will contain all the material properties as fem.function
        pdb (PhaseDataBase): class containing the material properties as numpy arrays, indexed by phase ID
        phase (fem.Function): function containing the phase ID for each cell, used to index the material properties from the PhaseDataBase
    Returns:
        CP (Function_material_properties): update CP with the material properties as fem.function
        
    Note: The field in this version of the code are static. They are not advected. It is necessary to call it 
    once during a preprocessing step, after the phase is defined, and before the solver routine is called. 
        
    """

    ph = np.int32(phase.x.array)
    P0 = phase.function_space
    
    # Heat Conductivity properties
    CP.k0 = fem.Function(P0)  ; CP.k0.x.array[:]    =  pdb.k0[ph]
    CP.fr = fem.Function(P0)  ; CP.fr.x.array[:] =  pdb.radio_flag[ph]
    CP.k_a= fem.Function(P0)  ; CP.k_a.x.array[:] = pdb.k_a[ph]
    CP.k_b= fem.Function(P0)  ; CP.k_b.x.array[:] = pdb.k_b[ph]
    CP.k_c= fem.Function(P0)  ; CP.k_c.x.array[:] = pdb.k_c[ph]
    CP.k_d= fem.Function(P0)  ; CP.k_d.x.array[:] = pdb.k_d[ph]
    CP.k_e= fem.Function(P0)  ; CP.k_e.x.array[:] = pdb.k_e[ph]
    CP.k_f= fem.Function(P0)  ; CP.k_f.x.array[:] = pdb.k_f[ph]
    # Heat Capacity properties
    CP.C0  = fem.Function(P0) ; CP.C0.x.array[:] = pdb.C0[ph]
    CP.C1 = fem.Function(P0)  ; CP.C1.x.array[:] = pdb.C1[ph]
    CP.C2 = fem.Function(P0)  ; CP.C2.x.array[:] = pdb.C2[ph]
    CP.C3 = fem.Function(P0)  ; CP.C3.x.array[:] = pdb.C3[ph]
    CP.C4 = fem.Function(P0)  ; CP.C4.x.array[:] = pdb.C4[ph]
    CP.C5 = fem.Function(P0)  ; CP.C5.x.array[:] = pdb.C5[ph]
    # Density properties
    CP.rho0    = fem.Function(P0)     ; CP.rho0.x.array[:]    = pdb.rho0[ph]
    CP.alpha0  = fem.Function(P0)     ; CP.alpha0.x.array[:]  = pdb.alpha0[ph]
    CP.alpha1  = fem.Function(P0)     ; CP.alpha1.x.array[:]  = pdb.alpha1[ph]
    CP.alpha2  = fem.Function(P0)     ; CP.alpha2.x.array[:]  = pdb.alpha2[ph]
    CP.Kb      = fem.Function(P0)     ; CP.Kb.x.array[:]      = pdb.Kb[ph]
    CP.option_rho = fem.Function(P0)  ; CP.option_rho.x.array[:] = pdb.option_rho[ph]
    
    CP.radio   = fem.Function(P0)   ; CP.radio.x.array[:]     = pdb.radio[ph]
    CP.radio.x.scatter_forward()
    CP.Tref    = pdb.Tref
    CP.R       = pdb.R
    CP.A       = pdb.A
    CP.B       = pdb.B
    CP.T_A     = pdb.T_A
    CP.x_A     = pdb.x_A
    CP.T_B     = pdb.T_B
    CP.x_B     = pdb.x_B
    
    return CP

        
        
        
        
        
@dataclass
class Functions_material_rheology():
    """Initialise the rheological properties 
    
    Note: Stokes equation can be evaluated in a sub-domain (wedge). The class
    is separated from the main material properties dataclass for this reason. 
    """
    Bdif    : fem.Function = None
    Bdis    : fem.Function = None
    n       : fem.Function = None
    Edif    : fem.Function = None
    Edis    : fem.Function = None
    Vdif    : fem.Function = None
    Vdis    : fem.Function = None
    eta     : fem.Function = None
    option_eta : fem.Function = None
    eta_max : float = None
    R        : float = None

def populate_material_properties_rheology(CP:Functions_material_rheology
                                          ,pdb:PhaseDataBase
                                          ,phase:fem.Function)->Functions_material_rheology:
    """Change the rheological properties as fem.function, given the phase distribution and the PhaseDataBase.

    Args:
        CP (Functions_material_rheology): Rheological properties as fem.function
        pdb (PhaseDataBase): Phase Data Base containing the rheological properties as numpy arrays, indexed by phase ID
        phase (fem.Function): function containing the phase ID for each cell, used to index the material properties from the PhaseDataBase

    Returns:
        Functions_material_rheology: updated Functions_material_rheology with the rheological properties as fem.function
    """
    ph = np.int32(phase.x.array)
    P0 = phase.function_space
    
    CP.Bdif    = fem.Function(P0)     ; CP.Bdif.x.array[:]     = pdb.Bdif[ph]
    CP.Bdis    = fem.Function(P0)     ; CP.Bdis.x.array[:]     = pdb.Bdis[ph]
    CP.n       = fem.Function(P0)     ; CP.n.x.array[:]        = pdb.n[ph]
    CP.Edif    = fem.Function(P0)     ; CP.Edif.x.array[:]     = pdb.Edif[ph]
    CP.Edis    = fem.Function(P0)     ; CP.Edis.x.array[:]     = pdb.Edis[ph]
    CP.Vdif    = fem.Function(P0)     ; CP.Vdif.x.array[:]     = pdb.Vdif[ph]
    CP.Vdis    = fem.Function(P0)     ; CP.Vdis.x.array[:]     = pdb.Vdis[ph]
    CP.eta     = fem.Function(P0)     ; CP.eta.x.array[:]      = pdb.eta[ph]
    CP.option_eta = fem.Function(P0)  ; CP.option_eta.x.array[:] = pdb.option_eta[ph]
    CP.eta_max = pdb.eta_max
    CP.R       = pdb.R
    CP.eta.x.scatter_forward()
    return CP 
#---------------------------------------------------------------------------------
def heat_conductivity_FX(FG : Functions_material_properties_global
                         ,T : fem.Function 
                         ,p : fem.Function
                         ,Cp : fem.Expression
                         ,rho: fem.Expression) -> fem.Expression:
    """Function that computes the heat conductivity for a given Pressure and Temperature
       k = k0 + kb*exp(-(T-Tr)/kc)+kd*exp(-(T-Tr)/ke)*exp(kf*P) + rad * flag
       if the phase has a constant conductivity, k0 is defined and positive, otherwise is 0.0 
       while while the other properties are set to be 0.0 
       kb*exp(T-Tr/0)=0.0 so no active. 
    Args:
        FG (Functions_material_properties_global) : Precomputed function spaces with the material properties
        T (fem.Function)  : Temperature field or trial function
        p (_type_)  : Lithostatic pressure field 
        Cp (_type_) : Cp expression for the computation of the conductivity, it is an expression because it depends on T 
        rho (_type_): rho expression for the computation of the conductivity, it is an expression because it depends on T and P 

    Returns:
        k: fem.form expression for the heat conductivity, to be used in the weak formulation of the heat equation.
    """
    
    # Compute the radiative conductivity
    k_rad = FG.A * exp(-(T-FG.T_A)**2/ (2*FG.x_A ** 2 )) + FG.B * exp(-(T - FG.T_B)**2 / (2* FG.x_B**2))
    # Compute the lattice conductivity
    kappa_lat = FG.k_a + FG.k_b * exp(-(T-FG.Tref)/FG.k_c) + FG.k_d * exp(-(T-FG.Tref)/FG.k_e)
    # Compute the pressure dependence of the conductivity
    kappa_p   = exp(FG.k_f * p)  
    # Compute the total conductivity FG.k0 -> constant conductivity, if the phase has it, otherwise 0.0
    k = FG.k0  + (kappa_lat * kappa_p * Cp * rho + k_rad * FG.fr)  

    return k 
#---------------------------------------------------------------------------------
def heat_capacity_FX(FG : Functions_material_properties_global
                     ,T : fem.Function) -> fem.Expression: 
    """Derive the heat capacity expression

    Args:
        FG (Functions_material_properties_global): Precomputed function spaces with the material properties
        T (fem.Function): Temperature field or trial function

    Returns:
        fem.Expression: expression for the heat capacity, to be used in the weak formulation of the heat equation.
    """
    # General formula for the heat capacity, it is an expression because it depends on T. C0 = Cp in case the heat capacity is constant, otherwise the other parameters are active.
    C_p = FG.C0 + FG.C1 * (T**(-0.5)) + FG.C2 * T**(-2.) + FG.C3 * (T**(-3.)) + FG.C4 * T + FG.C5 * T**2

    return C_p
  
def compute_radiogenic(FG, hs): 
    hs.interpolate(FG.radio)
    return hs 
    
#---------------------------------------------------------------------------------
def density_FX(FG:Functions_material_properties_global
               ,T:fem.Function
               ,p:fem.Function)->fem.Expression:
    
    """Derive the density expression
    Args:
        FG (Functions_material_properties_global): Precomputed function spaces with the material properties
        T (fem.Function): Temperature field or trial function
        p (fem.Function): Pressure field or trial function
    Returns:
        fem.Expression: expression for the density, to be used in the weak formulation of the heat equation.
        
    """

    # Base density (with temperature dependence)
    temp_term = exp(- p * FG.alpha2)*(FG.alpha0 * (T - FG.Tref) + (FG.alpha1 / 2.0) * (T**2 - FG.Tref**2))
    rho_temp = FG.rho0 * (1-temp_term) 

    # Add pressure dependence if needed
    rho = conditional(
        eq(FG.option_rho, 0), FG.rho0,
        conditional(
            eq(FG.option_rho, 1), rho_temp,
            rho_temp * exp(p / FG.Kb)
        )
    )

    return rho 
#----------------------------------
def alpha_FX(FG : Functions_material_properties_global
             ,T : fem.Function
             ,p : fem.Function)->fem.Expression:
    """Derive the thermal expansivity expression
    Args:
        FG (Functions_material_properties_global): Precomputed function spaces with the material properties
        T (fem.Function): Temperature field or trial function
        p (fem.Function): Pressure field or trial function
    Returns:
        fem.Expression: expression for the thermal expansivity, to be used in the weak formulation of the heat equation.
        
    """

    # Base density (with temperature dependence)
    alpha =  exp(- p * FG.alpha2) * (FG.alpha0  + (FG.alpha1) * (T- FG.Tref))


    return alpha 

def cell_average_DG0(mesh, expr_ufl):
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    f0 = fem.Function(V0)

    w = ufl.TestFunction(V0)
    u = ufl.TrialFunction(V0)
    dx = ufl.dx(domain=mesh)

    a = fem.form(w * u * dx)
    L = fem.form(w * expr_ufl * dx)

    A = fem.petsc.assemble_matrix(a)
    A.assemble()
    b = fem.petsc.assemble_vector(L)

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.solve(b, f0.x.petsc_vec)
    f0.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
    return f0
#---------------------------------------------------------------------------------
def compute_viscosity_FX(e:fem.Expression
                        ,T_in:fem.Function
                        ,P_in:fem.Function
                        ,FR:Functions_material_rheology
                        ,sc:Scal)->fem.Expression:
    """Compute the viscosity for a given strain rate, Pressure and Temperature
    The viscosity is computed as a composite of the diffusion and dislocation creep, with a maximum viscosity cutoff.
    The composite is computed as the harmonic average of the diffusion and dislocation viscosity.
    Args:
        e (fem.Expression): strain rate invariant expression, to be computed in the weak formulation
        T_in (fem.Function): Temperature field 
        P_in (fem.Function): Pressure field 
        FR (Functions_material_rheology): Precomputed function spaces with the rheological properties
        sc (Scal): Scaling class containing the scaling factors for the problem, used to scale the input fields 
        to the non-dimensional values used in the rheological laws. 
    Returns:    
        fem.Expression: expression for the viscosity, to be used in the weak formulation of the Stokes equation.
    
    """    
    def ufl_pow(u, v, eps=0):
        return ufl.exp(v * ufl.ln(u + eps))
    
    def compute_eII(e):
        e_II  = sqrt(0.5*inner(e, e) + 1e-15)    
        return e_II
    
    e_II = compute_eII(e)
    # If your phase IDs are available per cell for mesh0:
    
    # Eta max 
    # strain indipendent  
    cdf = FR.Bdif * exp(-(FR.Edif + P_in * sc.stress * FR.Vdif )/(FR.R * T_in * sc.Temp)) ; cds = FR.Bdis * exp(-(FR.Edis + P_in * sc.stress * FR.Vdis)/(FR.R * T_in*sc.Temp))
    # compute tau guess
    n_co  = (1-FR.n)/FR.n
    n_inv = 1/FR.n 
    # Se esiste un cazzo di inferno in culo a Satana ci vanno quelli che hanno generato 
    # sto modo creativo di fare gli esponenti. 
    etads     = 0.5 * cds**(-n_inv) * e_II**n_co
    etadf     = 0.5 * cdf**(-1)
    eta_av    = 1 / (1 / etads + 1/etadf + 1/FR.eta_max)
    eta_df    = 1 / (1 / etadf + 1 / FR.eta_max) 
        
    # check if the option_eta -> constant or not, otherwise release the composite eta 
    eta = ufl.conditional(
        ufl.eq(FR.option_eta, 0.0), FR.eta,
        ufl.conditional(
            ufl.eq(FR.option_eta, 1.0), eta_df,
            eta_av
        )
    )

    return eta
#---------------------------------------------------------------------------------
def compute_plastic_strain(e_II:fem.Expression
                           ,T_in:fem.Function
                           ,P_in:fem.Function
                           ,pdb:PhaseDataBase
                           ,ph:int
                           ,phwz:fem.Function
                           ,sc)->tuple[fem.Expression, fem.Expression]:
    """





    """

    e_II = e_II 
    
    
    # If your phase IDs are available per cell for mesh0:
    
    # UNFORTUNATELY I AM STUPID and i do not have any idea how to scale the energies such that it would be easier to handle. Since the scale of force and legth is self-consistently related to mass, i do not know how to deal with the fucking useless mol in the energy of activation 
    T = T_in.copy()
    P = P_in.copy()
    T.x.array[:] = T.x.array[:]*sc.Temp  ;T.x.scatter_forward()
    P.x.array[:] = P.x.array[:]*sc.stress;P.x.scatter_forward()
    P0    = T.function_space
    
    # Gather material parameters as UFL expressions via indexing
    Bdif    = fem.Function(P0,name = 'Bdif')  ; Bdif.x.array[:]    =  pdb.Bdif[phwz]
    Bdis    = fem.Function(P0,name = 'Bdis')  ; Bdis.x.array[:]    =  pdb.Bdis[phwz]
    n       = fem.Function(P0,name = 'n')     ; n.x.array[:]       =  pdb.n[phwz]
    Edif    = fem.Function(P0,name = 'Edif')  ; Edif.x.array[:]    =  pdb.Edif[phwz]
    Edis    = fem.Function(P0,name = 'Edis')  ; Edis.x.array[:]    =  pdb.Edis[phwz]
    Vdif    = fem.Function(P0,name = 'Vdif')  ; Vdif.x.array[:]    =  pdb.Vdif[phwz]
    Vdis    = fem.Function(P0,name = 'Vdis')  ; Vdis.x.array[:]    =  pdb.Vdis[phwz]
    
    # In case the viscosity for the given phase is constant 
    eta_con     = fem.Function(P0) ; eta_con.x.array[:]     =  pdb.eta[phwz]
    # Option for eta for a given marker number ph 
    opt_eta = fem.Function(P0)  ; opt_eta.x.array[:] =  pdb.option_eta[phwz]
    # strain indipendent  
    cdf = Bdif * exp(-(Edif + P * Vdif )/(pdb.R * T)) ; cds = Bdis * exp(-(Edis + P * Vdis)/(pdb.R * T))
    # compute tau guess
    n_co  = (1-n)/n
    n_inv = 1/n 
    # Se esiste un cazzo di inferno in culo a Satana ci vanno quelli che hanno generato 
    # sto modo creativo di fare gli esponenti. 
    etads     = 0.5 * cds**(-n_inv) * e_II**n_co
    etadf     = 0.5 * cdf**(-1)
    eta_av    = 1 / (1 / etads + 1/etadf + 1/pdb.eta_max)
    
    # -> Compute the tau lim 
    tau_lim  = pdb.cohesion * cos(pdb.friction_angle) + P_in * sin (pdb.friction_angle)
    
    tau_vis  = 2 * eta_av * e_II
    
    
    # check if the option_eta -> constant or not, otherwise release the composite eta 
    
    tau_eff = ufl.conditional(tau_vis > tau_lim, tau_lim, tau_vis)

    e_plr2    = (e_II - (tau_eff / 2 /eta_av)) / e_II
    
    
    e_plr = ufl.conditional(e_plr2 < 0.0, 0.0, e_plr2)


    return e_plr, tau_eff
#-----------------------------------------------------------------------------
@njit
def heat_conductivity(pdb:PhaseDataBase
                      ,T:NDArray[np.float64]
                      ,p:NDArray[np.float64]
                      ,rho:NDArray[np.float64]
                      ,Cp:NDArray[np.float64]
                      ,ph:int)->NDArray[np.float64]:    

    """Compute the heat conductivity for a given Pressure and Temperature
    Args:
        pdb (PhaseDataBase): Phase Data Base containing the material properties as numpy arrays, indexed by phase ID
        T (NDArray[np.float64]): Temperature field as a numpy array
        p (NDArray[np.float64]): Pressure field as a numpy array
        rho (NDArray[np.float64]): Density field as a numpy array
        Cp (NDArray[np.float64]): Heat capacity field as a numpy array
        ph (int): phase ID for which to compute the conductivity, used to index the material properties from the PhaseDataBase
    Returns:
        NDArray[np.float64]: array containing the heat conductivity
    
    Function to compute the heat conductivity for a given Pressure and Temperature, used for the post-processing of the thermal properties.
    It is used for computing the initial conductivity field for the oceanic plate thermal boundary condition.
    """

    k_rad = pdb.A * np.exp(-(T-pdb.T_A)**2/ (2*pdb.x_A ** 2 )) + pdb.B * np.exp(-(T - pdb.T_B)**2 / (2* pdb.x_B**2))

    kappa_lat = pdb.k_a[ph] + pdb.k_b[ph] * np.exp(-(T-pdb.Tref)/pdb.k_c[ph]) + pdb.k_d[ph] * np.exp(-(T-pdb.Tref)/pdb.k_e[ph])
    
    kappa_p   = np.exp(pdb.k_f[ph] * p)  

    k = pdb.k0[ph] + kappa_lat * kappa_p * Cp * rho + k_rad * pdb.radio_flag[ph]

    return k 
#---------------------------------------------------------------------------------
@njit
def density(pdb:PhaseDataBase
            ,T:NDArray[np.float64]
            ,p:NDArray[np.float64]
            ,ph:int)->NDArray[np.float64]:
    """Compute the density for a given Pressure and Temperature, used for oceanic plate thermal boundary condition.
    
    Args:
        pdb (PhaseDataBase): Phase Data Base containing the material properties as numpy arrays, indexed by phase ID
        T (NDArray[np.float64]): Temperature field as a numpy array
        p (NDArray[np.float64]): Pressure field as a numpy array
        ph (int): phase ID for which to compute the density, used to index the material properties from the PhaseDataBase
    Returns:
        NDArray[np.float64]: array containing the density   
    Function to compute the density for a given Pressure and Temperature. 
    """
    rho_0 = pdb.rho0[ph] 
    
    if pdb.option_rho[ph] == 0:
        # constant variables 
        return rho_0 
    else :
        # calculate rho
        rho     = rho_0 * (1 - np.exp(- p * pdb.alpha2[ph])*( pdb.alpha0[ph] * (T - pdb.Tref) + (pdb.alpha1[ph]/2.) * ( T**2 - pdb.Tref**2 )))
        if pdb.option_rho[ph] == 2:
            # calculate the pressure dependence of the density
            Kb = pdb.Kb[ph]
            rho = rho * np.exp(p/Kb)    
    
    return rho

#---------------------------------------------------------------------------------
@njit
def heat_capacity(pdb:PhaseDataBase
                  ,T:NDArray[np.float64]
                  ,ph:int)->NDArray[np.float64]:
    """Compute heat capacity

    Args:
        pdb (PhaseDataBase): Phase Data Base containing the material properties as numpy arrays, indexed by phase ID
        T (NDArray[np.float64]): Temperature field as a numpy array
        ph (int): phase ID for which to compute the heat capacity, used to index the material properties from the PhaseDataBase

    Returns:
        NDArray[np.float64]: array containing the heat capacity
    """

    C_p = pdb.C0[ph] + pdb.C1[ph] * (T**(-0.5)) + pdb.C2[ph] * T**(-2.0) + pdb.C3[ph] * (T**(-3.)) + pdb.C4[ph]* T + pdb.C5[ph] * T**2
    
    return C_p
#---------------------------------------------------------------------------------
@njit 
def compute_thermal_properties(pdb,T,p,ph):
    
    Cp   = heat_capacity(pdb,T,ph)
    rho  = density(pdb,T,p,ph)
    k    = heat_conductivity(pdb,T,p,rho,Cp,ph)
    
    return Cp, rho, k 
#----------------------------------------------------------------------------------
