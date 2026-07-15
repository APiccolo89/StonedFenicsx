
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
    """Abstract base for cached material-property dataclasses.

    Defines the two InitVar arguments consumed by `__post_init__` in every
    subclass.  `pdb` carries the per-phase parameter arrays and `phase`
    carries the DG0 phase-ID field on the target sub-mesh.  Neither is
    stored as an instance attribute after construction; they are used only to
    populate the concrete fem.Function fields defined in each subclass.

    Args:
        pdb (PhaseDataBase): Material-property database with parameter arrays
            indexed by integer phase ID.
        phase (fem.Function): DG0 Function whose `.x.array[:]` holds the
            integer phase ID for every cell in the sub-mesh.
    """
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

        """Allocate and fill all thermal material-property fem.Functions.

        Uses the integer phase-ID array from `phase.x.array` to index into
        the per-phase parameter arrays in `pdb`, broadcasting a scalar per
        element into a DG0 fem.Function defined on the same function space as
        `phase`.  All functions are filled once and treated as static
        throughout the simulation (properties are not advected with the flow).

        Scalar parameters that are mesh-independent (reference temperature,
        gas constant, radiative conductivity coefficients) are stored as plain
        Python floats rather than fem.Functions.

        Args:
            pdb (PhaseDataBase): Material-property database; all arrays are
                already non-dimensionalised before this call.
            phase (fem.Function): DG0 phase-ID field on the target sub-mesh.
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
        """Allocate and fill all rheological material-property fem.Functions.

        Mirrors the pattern of THERMALCACHED.__post_init__: indexes `pdb`
        arrays by integer phase ID and writes the result into newly allocated
        DG0 fem.Functions on the same function space as `phase`.  Creep
        prefactors (b_dif, b_dis), stress exponent (n), activation energies
        (e_dif, e_dis), and activation volumes (v_dif, v_dis) are stored as
        Functions so they enter the UFL viscosity expression symbolically.
        Scalar upper bounds (eta_max, eta_def) and gas constant are kept as
        Python floats.

        Args:
            pdb (PhaseDataBase): Material-property database; all arrays are
                already non-dimensionalised before this call.
            phase (fem.Function): DG0 phase-ID field on the Stokes sub-mesh
                (wedge or slab).
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
    """Build the UFL expression for thermal conductivity as a function of T and P.

    Implements the composite conductivity law:
        k = k0 + (k_lat(T) * exp(k_f * P) * Cp * rho  +  k_rad(T) * flag_rad)

    where:
        k_lat(T) = k_a + k_b * exp(-(T - T_ref)/k_c) + k_d * exp(-(T - T_ref)/k_e)
        k_rad(T) = a_rad * exp(-(T - T_a)^2 / (2 x_a^2))
                 + b_rad * exp(-(T - T_b)^2 / (2 x_b^2))

    For phases with a constant conductivity, `k0 > 0` and the lattice/radiative
    terms are zero (their coefficients are 0 in the database).  For mantle
    phases, `k0 = 0` and the T- and P-dependent terms are active.

    Args:
        scal_cached (THERMALCACHED): Cached DG0 fem.Functions for all
            conductivity coefficients.
        T (fem.Function): Temperature field (dimensionless) — typically the
            Picard-frozen `T_k` or the UFL trial function.
        p (fem.Function): Lithostatic pressure field (dimensionless).
        Cp (fem.Expression): UFL heat-capacity expression from `heat_capacity_FX`;
            passed in to avoid recomputing it inside this function.
        rho (fem.Expression): UFL density expression from `density_FX`.

    Returns:
        fem.Expression: UFL expression for k, to be used directly in the
        bilinear form of the energy equation.
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
    """Build the UFL expression for heat capacity as a polynomial in T.

    Implements the Berman (1988) polynomial:
        Cp = c0 + c1 * T^(-0.5) + c2 * T^(-2) + c3 * T^(-3) + c4 * T + c5 * T^2

    For phases with a constant heat capacity only `c0` is non-zero; all other
    coefficients are set to 0 in the database.

    Args:
        scal_cached (THERMALCACHED): Cached DG0 fem.Functions for the heat-
            capacity polynomial coefficients c0–c5.
        T (fem.Function): Temperature field (dimensionless).

    Returns:
        fem.Expression: UFL expression for Cp, to be used in the energy
        bilinear form and passed to `heat_conductivity_FX`.
    """
    # General formula for the heat capacity, it is an expression because it depends on T. C0 = Cp in case the heat capacity is constant, otherwise the other parameters are active.
    C_p = scal_cached.c0 + scal_cached.c1 * (T**(-0.5)) + scal_cached.c2 * T**(-2.) + scal_cached.c3 * (T**(-3.)) + scal_cached.c4 * T + scal_cached.c5 * T**2

    return C_p
  
def compute_radiogenic(scal_cached:THERMALCACHED, hs:fem.Function) -> fem.Function:
    """Interpolate the cached radiogenic heating field into a pre-allocated Function.

    Copies the per-element radiogenic heat production (already stored as a
    DG0 fem.Function in `scal_cached.radiogenic`) into `hs` by direct
    interpolation, avoiding a new allocation.

    Args:
        scal_cached (THERMALCACHED): Thermal cache holding `radiogenic` as a
            DG0 fem.Function of non-dimensionalised heat production [W/m^3].
        hs (fem.Function): Pre-allocated target Function on the same function
            space; overwritten in-place.

    Returns:
        fem.Function: The updated `hs` object.
    """
    hs.interpolate(scal_cached.radiogenic)
    return hs
# ---
def density_FX(scal_cached:THERMALCACHED
               ,T:fem.Function
               ,p:fem.Function)->fem.Expression:
    """Build the UFL expression for density as a function of T and P.

    Selects one of three density formulations per element via UFL conditional
    branching on `scal_cached.option_rho`:
        0 -- constant:          rho = rho0
        1 -- T-dependent:       rho = rho0 * (1 - alpha_int(T, P))
        2 -- T- and P-dependent: rho = rho0 * (1 - alpha_int) * exp(P / kb)

    where the integrated thermal expansivity term is:
        alpha_int = exp(-P * alpha2) * (alpha0 * (T - T_ref)
                  + (alpha1/2) * (T^2 - T_ref^2))

    Args:
        scal_cached (THERMALCACHED): Cached DG0 fem.Functions for rho0,
            alpha0, alpha1, alpha2, kb, option_rho, and the reference
            temperature temp_ref.
        T (fem.Function): Temperature field (dimensionless).
        p (fem.Function): Lithostatic pressure field (dimensionless).

    Returns:
        fem.Expression: UFL expression for density, to be inserted into the
        energy and Stokes bilinear forms.
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
    """Build the UFL expression for thermal expansivity as a function of T and P.

    Implements the linearised Birch-Murnaghan expansivity:
        alpha(T, P) = exp(-P * alpha2) * (alpha0 + alpha1 * (T - T_ref))

    The pressure factor exp(-P * alpha2) accounts for compressional
    suppression of thermal expansion.  For constant-alpha phases only alpha0
    is non-zero.

    Args:
        scal_cached (THERMALCACHED): Cached DG0 fem.Functions for alpha0,
            alpha1, alpha2, and the reference temperature temp_ref.
        T (fem.Function): Temperature field (dimensionless).
        p (fem.Function): Lithostatic pressure field (dimensionless).

    Returns:
        fem.Expression: UFL expression for alpha, suitable for use in the
        buoyancy term of the Stokes momentum equation.
    """

    # Base density (with temperature dependence)
    alpha =  exp(- p * scal_cached.alpha2) * (scal_cached.alpha0  + (scal_cached.alpha1) * (T- scal_cached.temp_ref))


    return alpha 

# ---
def compute_viscosity_FX(e:fem.Expression
                        ,temp_in:fem.Function
                        ,pres_in:fem.Function
                        ,pdb:PhaseDataBase
                        ,rg_cached:RHEOLOGYCACHED)->fem.Expression:
    """Build the UFL expression for effective viscosity from strain rate, T, and P.

    Computes a composite diffusion + dislocation creep viscosity with a
    maximum-viscosity cutoff, selecting the formulation per element via UFL
    conditionals on `rg_cached.option_eta`:
        0 -- constant:          eta = rg_cached.eta
        1 -- diffusion only:    eta = harmonic(eta_df, eta_max)
        2 -- composite:         eta = harmonic(eta_df, eta_ds, eta_max)

    The creep flow factors (in SI, re-dimensionalised internally via
    pdb.temp_scal and pdb.pres_scal) are:
        C_df = b_dif * exp(-(E_dif + P * V_dif) / (R * T))
        C_ds = b_dis * exp(-(E_dis + P * V_dis) / (R * T))

    The dislocation viscosity is:
        eta_ds = 0.5 * C_ds^(-1/n) * e_II^((1-n)/n)

    where e_II = sqrt(0.5 * e:e + epsilon) is the second invariant of the
    strain-rate tensor.

    Args:
        e (fem.Expression): Full strain-rate tensor UFL expression
            (not the invariant); the invariant is computed internally.
        temp_in (fem.Function): Temperature field (dimensionless).
        pres_in (fem.Function): Pressure field (dimensionless).
        pdb (PhaseDataBase): Database carrying `temp_scal`, `pres_scal`, and
            `gas_constant` needed to re-dimensionalise the Arrhenius exponent.
        rg_cached (RHEOLOGYCACHED): Cached DG0 fem.Functions for all creep
            parameters and the `option_eta` selector.

    Returns:
        fem.Expression: UFL expression for eta, to be inserted into the Stokes
        bilinear form.
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
                           )->tuple[fem.Expression, fem.Expression, fem.Expression]:
    """Build UFL expressions for the effective shear stress in the plastic weak zone.

    Computes the visco-plastic effective stress using a dislocation-creep
    flow law for the weak-zone phase (separate parameters from the bulk
    mantle).  Re-dimensionalises temperature and pressure internally (copies
    are made to avoid modifying the solver fields) because activation energies
    are stored in SI (J/mol) and cannot be non-dimensionalised consistently
    with a mol-based gas constant.

    Optionally applies a water-fugacity correction to the creep prefactor
    when `pdb.water_cor == 2`:
        C_ds *= (f_H2O(P,T) / f_H2O(P_ref, T_ref))^r_wz

    The effective stress is regularised via a smooth tanh cap:
        tau_eff = tau_vis * tanh(tau_lim / tau_vis)

    where `tau_lim = P * sin(phi)` is the Drucker-Prager yield stress.

    Args:
        e_ii (fem.Expression): Second invariant of the strain-rate tensor
            (already computed, e.g. from `compute_strain_rate`).
        temp_in (fem.Function): Temperature field (dimensionless); copied and
            re-dimensionalised internally.
        pres_in (fem.Function): Pressure field (dimensionless); copied and
            re-dimensionalised internally.
        pdb (PhaseDataBase): Database with weak-zone rheology parameters
            (bdis_wz, n_wz, edis_wz, vdis_wz, eh2o_wz, vh2o_wz, r_wz, phi)
            and scaling factors (temp_scal, pres_scal, gas_constant).


    Args:
        e_ii (fem.Expression): Second invariant of the strain-rate tensor
            (already computed, e.g. from `compute_strain_rate`).
        temp_in (fem.Function): Temperature field (dimensionless); copied and
            re-dimensionalised internally.
        pres_in (fem.Function): Pressure field (dimensionless); copied and
            re-dimensionalised internally.
        pdb (PhaseDataBase): Database with weak-zone rheology parameters
            (bdis_wz, n_wz, edis_wz, vdis_wz, eh2o_wz, vh2o_wz, r_wz, phi)
            and scaling factors (temp_scal, pres_scal, gas_constant).

    Returns:
        tuple[fem.Expression, fem.Expression, fem.Expression]:
            tau_eff  -- regularised effective stress (UFL expression).
            tau_vis  -- viscous stress before plastic cap (UFL expression).
            tau_lim  -- Drucker-Prager yield stress (UFL expression).
    """
    
    # UNFORTUNATELY I AM STUPID and i do not have any idea how to scale the energies such that it would be easier to handle. Since the scale of force and legth is self-consistently related to mass, i do not know how to deal with the fucking useless mol in the energy of activation
    # NOTE: rescale as UFL expressions on temp_in/pres_in directly (do NOT .copy() into
    # a detached Function) -> compute_shear_heating() is only called once (its form is
    # cached and reused for the whole run), so tau_eff must keep a live UFL reference to
    # the solver's T/P Functions to pick up their updated values on each reassembly.
    temp = temp_in * pdb.temp_scal
    pres = pres_in * pdb.pres_scal
    # Gather material parameters as UFL expressions via indexing
    bdis =  pdb.bdis_wz
    n    =  pdb.n_wz
    edis =  pdb.edis_wz
    vdis =  pdb.vdis_wz
    eh2o = pdb.eh2o_wz
    vh2o = pdb.vh2o_wz
    temp_ref = pdb.temp_ref * pdb.temp_scal
    pres_ref = pdb.pres_ref * pdb.pres_scal
    
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
