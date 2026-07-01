
# modules
from stonedfenicsx.config.phase_db import PhaseDataBase
from stonedfenicsx.config.scal import Scal
from ufl import cos, sin, tan, conditional, eq,exp, sqrt,inner
import ufl
import dolfinx.fem as fem 
import numpy as np 
from dataclasses import dataclass, InitVar
# ---
@dataclass
class MATERIALS:
    pdb : InitVar[PhaseDataBase]
    phase : InitVar[fem.Function]
# ---
# ---
@dataclass
class THERMALCACHED(MATERIALS):
    """Function containing the fem.function per each of the parameter of material properties
    Member variables:
    k0: constant conductivity
    rg_cached: radiogenic flag
    k_a, k_b, k_c, k_d, k_e, k_f: conductivity parameters
    C0, C1, C2, C3, C4, C5: heat capacity parameters
    rho0, alpha0, alpha1, alpha2, Kb: density parameters and alpha parameters
    option_rho: flag to choose the density formulation
    radio: radiogenic heating
    Tref, R, A, B, T_A, x_A, T_B, x_B: parameters for the radiative conductivity    
    """
    # Heat Conductivity properties
    k0 : fem.Function = None 
    rg_cached : fem.Function = None
    k_a: fem.Function = None
    k_b: fem.Function = None
    k_c: fem.Function = None
    k_d: fem.Function = None
    k_e: fem.Function = None
    k_f: fem.Function = None
    # Heat Capacity properties
    c0 : fem.Function = None
    c1 : fem.Function = None
    c2 : fem.Function = None
    c3 : fem.Function = None
    c4 : fem.Function = None
    c5 : fem.Function = None
    # Density properties
    rho0    : fem.Function = None
    alpha0  : fem.Function = None
    alpha1  : fem.Function = None
    alpha2  : fem.Function = None
    kb      : fem.Function = None
    option_rho : fem.Function = None
    # Radiogenic heating properties
    radiogenic   : fem.Function = None
    # Other properties
    temp_ref : float = 0.0
    gas_constant : float = 8.3145
    a_rad : float = 0.0
    b_rad : float = 0.0
    temp_a : float = 0.0
    x_a : float = 0.0
    temp_b : float = 0.0
    x_b : float = 0.0
    def __post_init__(self
                     ,pdb:PhaseDataBase
                     ,phase:fem.Function)->None:

        """    Initialize all thermal properties as FEniCS functions.
        Args:        
            self (THERMALCACHED): FEM functions for all thermal material properties
            pdb (PhaseDataBase): class containing the material properties as numpy arrays, indexed by phase ID
            phase (fem.Function): function containing the phase ID for each cell, used to index the material properties rg_cachedom the PhaseDataBase
        Returns:
            self (Function_material_properties): update self with the material properties as fem.function

        Note: The field in this version of the code are static. They are not advected. It is necessary to call it 
        once during a preprocessing step, after the phase is defined, and before the solver routine is called. 

        """

        ph = np.int32(phase.x.array)
        ph_fs = phase.function_space

        # Heat Conductivity properties
        self.k0 = fem.Function(ph_fs)  
        self.rg_cached = fem.Function(ph_fs)  
        self.k_a= fem.Function(ph_fs)  
        self.k_b= fem.Function(ph_fs)  
        self.k_c= fem.Function(ph_fs)  
        self.k_d= fem.Function(ph_fs)  
        self.k_e= fem.Function(ph_fs)  
        self.k_f= fem.Function(ph_fs)
        self.k0.x.array[:] =  pdb.k0[ph]
        self.rg_cached.x.array[:] =  pdb.radiative_conductivity[ph]
        self.k_a.x.array[:] = pdb.k_a[ph]
        self.k_b.x.array[:] = pdb.k_b[ph]
        self.k_c.x.array[:] = pdb.k_c[ph]
        self.k_d.x.array[:] = pdb.k_d[ph]
        self.k_e.x.array[:] = pdb.k_e[ph]
        self.k_f.x.array[:] = pdb.k_f[ph]
        # Heat Capacity properties
        self.c0 = fem.Function(ph_fs)
        self.c1 = fem.Function(ph_fs) 
        self.c2 = fem.Function(ph_fs) 
        self.c3 = fem.Function(ph_fs) 
        self.c4 = fem.Function(ph_fs) 
        self.c5 = fem.Function(ph_fs) 
        self.c0.x.array[:] = pdb.c0[ph]
        self.c1.x.array[:] = pdb.c1[ph]
        self.c2.x.array[:] = pdb.c2[ph]
        self.c3.x.array[:] = pdb.c3[ph]
        self.c4.x.array[:] = pdb.c4[ph]
        self.c5.x.array[:] = pdb.c5[ph]
        # Density properties
        self.rho0    =    fem.Function(ph_fs)   
        self.alpha0  =    fem.Function(ph_fs)   
        self.alpha1  =    fem.Function(ph_fs)   
        self.alpha2  =    fem.Function(ph_fs)   
        self.kb      =    fem.Function(ph_fs)   
        self.option_rho = fem.Function(ph_fs)
        self.rho0.x.array[:]    = pdb.rho0[ph]
        self.alpha0.x.array[:]  = pdb.alpha0[ph]
        self.alpha1.x.array[:]  = pdb.alpha1[ph]
        self.alpha2.x.array[:]  = pdb.alpha2[ph]
        self.kb.x.array[:]      = pdb.kb[ph]
        self.option_rho.x.array[:] = pdb.option_rho[ph]

        self.radiogenic   = fem.Function(ph_fs)
        self.radiogenic.x.array[:]     = pdb.radiogenic_heat[ph]
        self.radiogenic.x.scatter_forward()
        
        self.temp_ref    = pdb.temp_ref
        self.gas_constant       = pdb.gas_constant
        self.a_rad       = pdb.a_rad
        self.b_rad       = pdb.b_rad
        self.temp_a     = pdb.temp_a
        self.x_a     = pdb.x_a
        self.temp_b     = pdb.temp_b
        self.x_b     = pdb.x_b
# ---
# ---
@dataclass
class RHEOLOGYCACHED(MATERIALS):
    """Initialise the rheological properties 
    
    Note: Stokes equation can be evaluated in a sub-domain (wedge). The class
    is separated rg_cachedom the main material properties dataclass for this reason. 
    """
    b_dif    : fem.Function = None
    b_dis    : fem.Function= None
    n       : fem.Function = None
    e_dif   : fem.Function = None
    e_dis   : fem.Function = None
    v_dif   : fem.Function = None
    v_dis   : fem.Function = None
    eta     : fem.Function = None
    eta_def : fem.Function = None 
    option_eta : fem.Function = None
    eta_max : float = None
    gas_constant     : float = None

    def __post_init__(self,
                      pdb:PhaseDataBase
                      ,phase:fem.Function)->None:
        """Change the rheological properties as fem.function, given the phase distribution and the PhaseDataBase.

        Args:
            self (RHEOLOGYCACHED): FEM functions for all rheological properties
            pdb (PhaseDataBase): Phase Data Base containing the rheological properties as numpy arrays, indexed by phase ID
            phase (fem.Function): function containing the phase ID for each cell, used to index the material properties rg_cachedom the PhaseDataBase

        Returns:
            Modifies self in place with the rheological properties as fem.function.
        """
        ph = np.int32(phase.x.array)
        ph_fs = phase.function_space

        self.b_dif    = fem.Function(ph_fs)  
        self.b_dis    = fem.Function(ph_fs)  
        self.n        = fem.Function(ph_fs)   
        self.e_dif    = fem.Function(ph_fs)  
        self.e_dis    = fem.Function(ph_fs)  
        self.v_dif    = fem.Function(ph_fs)  
        self.v_dis    = fem.Function(ph_fs)  
        self.eta      = fem.Function(ph_fs)   
        self.option_eta = fem.Function(ph_fs)
        self.b_dif.x.array[:]     = pdb.b_dif[ph]
        self.b_dis.x.array[:]     = pdb.b_dis[ph]
        self.n.x.array[:]        = pdb.n[ph]
        self.e_dif.x.array[:]     = pdb.e_dif[ph]
        self.e_dis.x.array[:]     = pdb.e_dis[ph]
        self.v_dif.x.array[:]     = pdb.v_dif[ph]
        self.v_dis.x.array[:]     = pdb.v_dis[ph]
        self.eta.x.array[:]      = pdb.eta[ph]
        self.option_eta.x.array[:] = pdb.option_eta[ph]
        self.eta_max = pdb.eta_max
        self.eta_def = pdb.eta_def
        self.gas_constant       = pdb.gas_constant
        self.eta.x.scatter_forward()
# ---
def heat_conductivity_FX(scal_cached : THERMALCACHED
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
        scal_cached (THERMALCACHED) : Precomputed function spaces with the material properties
        T (fem.Function)  : Temperature field or trial function
        p (_type_)  : Lithostatic pressure field 
        Cp (_type_) : Cp expression for the computation of the conductivity, it is an expression because it depends on T 
        rho (_type_): rho expression for the computation of the conductivity, it is an expression because it depends on T and P 

    Returns:
        k: fem.form expression for the heat conductivity, to be used in the weak formulation of the heat equation.
    """
    
    # Compute the radiative conductivity
    k_rad = scal_cached.a_rad * exp(-(T-scal_cached.temp_a)**2/ (2*scal_cached.x_a ** 2 )) + scal_cached.b_rad * exp(-(T - scal_cached.temp_b)**2 / (2* scal_cached.x_b**2))
    # Compute the lattice conductivity
    kappa_lat = scal_cached.k_a + scal_cached.k_b * exp(-(T-scal_cached.temp_ref)/scal_cached.k_c) + scal_cached.k_d * exp(-(T-scal_cached.temp_ref)/scal_cached.k_e)
    # Compute the pressure dependence of the conductivity
    kappa_p   = exp(scal_cached.k_f * p)  
    # Compute the total conductivity scal_cached.k0 -> constant conductivity, if the phase has it, otherwise 0.0
    k = scal_cached.k0  + (kappa_lat * kappa_p * Cp * rho + k_rad * scal_cached.rg_cached)  

    return k 
# ---
def heat_capacity_FX(scal_cached : THERMALCACHED
                     ,T : fem.Function) -> fem.Expression: 
    """Derive the heat capacity expression

    Args:
        scal_cached (THERMALCACHED): Precomputed function spaces with the material properties
        T (fem.Function): Temperature field or trial function

    Returns:
        fem.Expression: expression for the heat capacity, to be used in the weak formulation of the heat equation.
    """
    # General formula for the heat capacity, it is an expression because it depends on T. C0 = Cp in case the heat capacity is constant, otherwise the other parameters are active.
    C_p = scal_cached.c0 + scal_cached.c1 * (T**(-0.5)) + scal_cached.c2 * T**(-2.) + scal_cached.c3 * (T**(-3.)) + scal_cached.c4 * T + scal_cached.c5 * T**2

    return C_p
  
def compute_radiogenic(scal_cached, hs): 
    hs.interpolate(scal_cached.radiogenic)
    return hs
# ---
def density_FX(scal_cached:THERMALCACHED
               ,T:fem.Function
               ,p:fem.Function)->fem.Expression:
    
    """Derive the density expression
    Args:
        scal_cached (THERMALCACHED): Precomputed function spaces with the material properties
        T (fem.Function): Temperature field or trial function
        p (fem.Function): Pressure field or trial function
    Returns:
        fem.Expression: expression for the density, to be used in the weak formulation of the heat equation.
        
    """

    # Base density (with temperature dependence)
    temp_term = exp(- p * scal_cached.alpha2)*(scal_cached.alpha0 * (T - scal_cached.temp_ref) + (scal_cached.alpha1 / 2.0) * (T**2 - scal_cached.temp_ref**2))
    rho_temp = scal_cached.rho0 * (1-temp_term)

    # Add pressure dependence if needed
    rho = conditional(
        eq(scal_cached.option_rho, 0), scal_cached.rho0,
        conditional(
            eq(scal_cached.option_rho, 1), rho_temp,
            rho_temp * exp(p / scal_cached.kb)
        )
    )

    return rho 
# ---
def alpha_FX(scal_cached : THERMALCACHED
             ,T : fem.Function
             ,p : fem.Function)->fem.Expression:
    """Derive the thermal expansivity expression
    Args:
        scal_cached (THERMALCACHED): Precomputed function spaces with the material properties
        T (fem.Function): Temperature field or trial function
        p (fem.Function): Pressure field or trial function
    Returns:
        fem.Expression: expression for the thermal expansivity, to be used in the weak formulation of the heat equation.
        
    """

    # Base density (with temperature dependence)
    alpha =  exp(- p * scal_cached.alpha2) * (scal_cached.alpha0  + (scal_cached.alpha1) * (T- scal_cached.temp_ref))


    return alpha 
# ---
def cell_average_DG0(mesh, expr_ufl):
    V0 = fem.functionspace(mesh, ("DG", 0))
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
# ---
def compute_viscosity_FX(e:fem.Expression
                        ,temp_in:fem.Function
                        ,pres_in:fem.Function
                        ,pdb:PhaseDataBase
                        ,rg_cached:RHEOLOGYCACHED)->fem.Expression:
    """Compute the viscosity for a given strain rate, Pressure and Temperature
    The viscosity is computed as a composite of the diffusion and dislocation creep, with a maximum viscosity cutoff.
    The composite is computed as the harmonic average of the diffusion and dislocation viscosity.
    Args:
        e (fem.Expression): strain rate invariant expression, to be computed in the weak formulation
        T_in (fem.Function): Temperature field 
        P_in (fem.Function): Pressure field 
        rg_cached (RHEOLOGYCACHED): Precomputed function spaces with the rheological properties
        sc (Scal): Scaling class containing the scaling factors for the problem, used to scale the input fields 
        to the non-dimensional values used in the rheological laws. 
    Returns:    
        fem.Expression: expression for the viscosity, to be used in the weak formulation of the Stokes equation.
    
    """    
    
    def compute_eii(e):
        e_ii  = sqrt(0.5*inner(e, e) + 1e-15)    
        return e_ii
    
    e_ii = compute_eii(e)
    
    # Eta max 
    # strain indipendent  
    cdf = rg_cached.b_dif * exp(-(rg_cached.e_dif + pres_in * pdb.pres_scal * rg_cached.v_dif) / (rg_cached.gas_constant * temp_in * pdb.temp_scal))
    cds = rg_cached.b_dis * exp(-(rg_cached.e_dis + pres_in * pdb.pres_scal * rg_cached.v_dis) / (rg_cached.gas_constant * temp_in * pdb.temp_scal))
    # compute tau guess
    n_co  = (1-rg_cached.n)/rg_cached.n
    n_inv = 1/rg_cached.n 
    # Se esiste un cazzo di inferno in culo a Satana ci vanno quelli che hanno generato 
    # sto modo creativo di fare gli esponenti. 
    etads     = 0.5 * cds**(-n_inv) * e_ii**n_co
    etadf     = 0.5 * cdf**(-1)
    eta_av    = 1 / (1 / etads + 1/etadf + 1/rg_cached.eta_max)
    eta_df    = 1 / (1 / etadf + 1 / rg_cached.eta_max) 
        
    # check if the option_eta -> constant or not, otherwise release the composite eta 
    eta = ufl.conditional(
        ufl.eq(rg_cached.option_eta, 0.0), rg_cached.eta,
        ufl.conditional(
            ufl.eq(rg_cached.option_eta, 1.0), eta_df,
            eta_av
        )
    )

    return eta
# ---
def compute_plastic_strain(e_ii:fem.Expression
                           ,temp_in:fem.Function
                           ,pres_in:fem.Function
                           ,pdb:PhaseDataBase
                           )->tuple[fem.Expression, fem.Expression]:
    """_summary_

    Args:
        e_ii (fem.Expression): _description_
        temp_in (fem.Function): _description_
        pres_in (fem.Function): _description_
        pdb (PhaseDataBase): _description_

    Returns:
        tuple[ufl.fem.Expression, ufl.fem.Expression]: _description_
    """
    
    # UNFORTUNATELY I AM STUPID and i do not have any idea how to scale the energies such that it would be easier to handle. Since the scale of force and legth is self-consistently related to mass, i do not know how to deal with the fucking useless mol in the energy of activation 
    temp = temp_in.copy()
    pres = pres_in.copy()
    temp.x.array[:] = temp.x.array[:]*pdb.temp_scal
    temp.x.scatter_forward()
    pres.x.array[:] = pres.x.array[:]*pdb.pres_scal
    pres.x.scatter_forward()
    # Gather material parameters as UFL expressions via indexing
    bdis =  pdb.bdis_wz
    n    =  pdb.n_wz
    edis =  pdb.edis_wz
    vdis =  pdb.vdis_wz
    eh2o = pdb.eh2o_wz
    vh2o = pdb.vH2o_wz
    temp_ref = pdb.temp_ref * pdb.temp_scale 
    pres_ref = pdb.pres_ref * pdb.pres_scale
    
    # strain indipendent  
    cds = bdis * exp(-(edis + pres * vdis)/(pdb.gas_constant * temp))
    # compute tau guess
    
    if pdb.water_cor == 2: 
        water = exp(-(eh2o+pres*vh2o)/(pdb.gas_constant * temp))/exp(-(eh2o+pres_ref*vh2o)/(pdb.gas_constant * temp_ref))
        cds = cds * water ** (pdb.r_wz)

    tau_vis  = cds ** (-1/n) * e_ii**(1/n)
        
    # -> Compute the tau lim 
    tau_lim  = pres_in * sin(pdb.phi)

    tau_eff = tau_vis * ufl.tanh(tau_lim/tau_vis)

    return tau_eff, tau_vis, tau_lim

# ---
# ---
# ---
