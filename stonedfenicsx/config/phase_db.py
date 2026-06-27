"""Modules"""
from numba import float64,int32
from numba.experimental import jitclass
import numpy as np
from numpy.typing import NDArray
from stonedfenicsx.utils import print_ph
from stonedfenicsx.config.config_utils import update_ip_file
from pathlib import Path
import yaml
from dataclasses import dataclass, field, asdict,InitVar
from numba import njit

#TO DO in the future. 
#The jit class is a pain in the ass. I used to use the numba routine for creating the left boundary condition and right 
#boundary condition. I kept the Iris FD code, adapting and making usable, but, on the second thought, I can convert all these configuration
#routine into a fenicsx small 1D numerical code. In this way, it is possible to convert the PhaseDataBase into a dataclass and split into two 
#classes: one super-classes that contains the essential portion of the given problem (i.e., the intelectual hoax of the kynematic models), and 
#a more generic soul that stores the essential portion of any geodynamic code. 
#For example, I am envisioning that the configuration routine, can be used for having a compressible fenicsx tool to explore the magmatic conduit, 
#with compressibility and more nice physics behind. config module can be used in a more generic way, and adapted. For the current pubblication, 
#I cannot envision a plan to make as generic possible the code. The contract is limited, and I cannot follow the plan as I wanted. I just throw a few 
#lines, in case someone would like to use the stuff that I wrote, and find it easy enough to build their tools. 


# ---
# ---
# Global variable
_F_CODE = {"None": 0, "Simpleshear": 1, "Uniaxial": 2}
_WATER_CODE = {"None": 0, "COH": 1, "Fugacity": 2}
_PATH_DATA_BASE_ = Path(Path(__file__).parents[0],'material_properties_data_base')
_GAS_CONSTANT_ = 8.3145
_P_REF_ = np.float64(1e5)
_T_REF_ = np.float64(298.15)
_ETA_DEF_ = np.float64(1e20)
# ---

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

    e_dif : float
        Activation energy for diffusion creep [J/mol].
    v_dif : float
        Activation volume for diffusion creep [m³/mol].
    b_dif : float
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
    e_dis : float
        Activation energy for dislocation creep [J/mol].
    v_dis : float
        Activation volume for dislocation creep [m³/mol].
    b_dis : float
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
    radiogenic_heat: float
        Radiogenic heat production [W/m³] (or [Pa/s] if used as source in σ units).
    radiative_conductivity: float
        Activation flag for radiogenic heating / radiative conductivity
        (0.0 = off, 1.0 = on, or a more general scaling factor).

    Notes
    -----
    This class is intended as a flexible container for building a PhaseDataBase.
    For your current kinematic slab work it may be somewhat overkill, but it
    should be reusable for other problems.
    """

    name_phase: str = "Undefined Phase"
    id_ph: int = 0
    # Viscosity / rheology
    name_diffusion: str = "Constant"
    e_dif: float | None = None
    v_dif: float | None = None
    b_dif: float | None = None

    name_dislocation: str = "Constant"
    n: float | None = None
    e_dis: float | None = None
    v_dis: float | None = None
    b_dis: float | None = None

    eta: float = 1e20  # constant viscosity

    # Thermal properties
    cp: float = 1250.0
    k: float = 3.0
    rho0: float = 3300.0

    name_capacity: str = "Constant"
    name_conductivity: str = "Constant"
    name_alpha: str = "Constant"
    name_density: str = "Constant"
    alpha0: float = 3e-5  # constant thermal expansivity
    radiogenic_heat: float = 0.0
    # Internal heating
    radiative_conductivity: float = 0.0


# –----------------------------------------------------------------------------------
@dataclass(slots=True)
class PhInput:
    """Container of the phases"""

    shear_heating_disl_law: str = "WetQuartzite"
    shear_heating_disl_ch: float = 0.0
    shear_heating_disl_phi: float = 0.0
    subducting_plate_mantle: Phase = field(init=False)
    oceanic_crust: Phase = field(init=False)
    wedge_mantle: Phase = field(init=False)
    overriding_mantle: Phase = field(init=False)
    overriding_upper_crust: Phase = field(init=False)
    overriding_lower_crust: Phase = field(init=False)

spec_phase = [
    # Viscosity – Diffusion creep
    ("e_dif", float64[:]),    # Activation energy diffusion creep [J/mol]
    ("v_dif", float64[:]),    # Activation volume diffusion creep [m^3/mol]
    ("b_dif", float64[:]),    # Pre-exponential factor diffusion creep [Pa^-1 s^-1]

    # Viscosity – Dislocation creep
    ("e_dis", float64[:]),    # Activation energy dislocation creep [J/mol]
    ("v_dis", float64[:]),    # Activation volume dislocation creep [m^3/mol]
    ("b_dis", float64[:]),    # Pre-exponential factor dislocation creep [Pa^-n s^-1]
    ("n", float64[:]),       # Stress exponent
    ("eta", float64[:]),     # Constant viscosity [Pa s]
    ("option_eta", int32[:]),# Option for viscosity calculation

    # Heat capacity
    ("c0", float64[:]),      # Cp coefficient [J/mol/K]
    ("c1", float64[:]),      # Cp coefficient [J/mol/K^0.5]
    ("c2", float64[:]),      # Cp coefficient [J·K/mol]
    ("c3", float64[:]),      # Cp coefficient [J·K^2/mol]
    ("c4", float64[:]),      # Cp coefficient []
    ("c5", float64[:]),      # Cp coefficient []
    ("option_cp", int32[:]), # Option for Cp calculation
    
    # Thermal conductivity    
    ("k_a", float64[:]),     # Koefficient diffusivity  [m^2/s]
    ("k_b", float64[:]),     # Pressure-dependent coefficient [m^2/s]
    ("k_c", float64[:]),     # Radiative heat transfer coefficient [K]
    ("k_d", float64[:]),     # Radiative heat transfer coefficient [m^2/s]
    ("k_e", float64[:]),     # Radiative polynomial coefficients [W/m/K^3]
    ("k_f", float64[:]),     # Pressure dependency               [W/m/K/Pa]
    
    ("k0", float64[:]),     # Constant conductivity             [W/m/k]
    ("option_k", int32[:]),  # Option for conductivity calculation

    # Density parameters
    ("alpha0", float64[:]),       # Thermal expansivity [1/K]
    ("alpha1", float64[:]),      # Second-order thermal expansivity [1/K^2]
    ("alpha2", float64[:]),      # Pressure dependency of alpha [1/Pa]

    ("kb", float64[:]),          # Bulk modulus [Pa]
    ("rho0", float64[:]),        # Reference density [kg/m^3]
    ("option_rho", int32[:]),    # Option for density calculation
    
    # radiogenic heat
    ("radiogenic_heat",float64[:]),
    
    ("a_rad",float64), # Ref conductivity A
    ("b_rad",float64), # Ref conductivity B
    ("temp_a",float64), # T [K]
    ("temp_b",float64), # T [K]
    ("x_a",float64), # T [K]
    ("x_b",float64), # T [K]
    ("radiative_conductivity",float64[:]), # radio flag

    
    ("id_ph", int32[:]),              # phase number
    
    # Constants
    ("temp_ref", float64),      # Reference temperature [K]
    ("pres_ref", float64),      # Reference pressure [Pa]
    ("gas_constant", float64),          # Gas constant [J/mol/K]
    ("temp_scal",float64),     # T_scal [K] -> Important, as within the exponential soul of diffusion/dislocation creep  R is in mol I don't know how to make dimensionless
    ("pres_scal",float64),     # Same reason as before [Pa]
    ("eta_min",float64),    # minimum viscosity [Pas]
    ("eta_max",float64),    # max viscosity [Pas]
    ("eta_def",float64),    # default viscosity [Pas]

    # Weak Zone Parameters
    ('edis_wz',float64),
    ('vdis_wz',float64),
    ('n_wz',float64),
    ('bdis_wz',float64),
    ('eh2o_wz',float64),
    ('bh2o_wz',float64),
    ('ah2o_wz',float64),
    ('vh2o_wz',float64),
    ('water_cor',int32),
    ('r_wz',float64),
    ('vis_con_fl',int32),
    ('eta_wz',float64),
    ("phi",float64),
    ('cohesion',float64),
]

@jitclass(spec_phase)
class PhaseDataBase:
    """Phase database: 
    
    """
    def __init__(self
                 ,number_phases:int
                 ,eta_max:float
                 ,d=0.5):
        # Initialize individ_phual fields as arrays
        """_summary_

        Args:
            number_phases (int): number of phases 
            eta_max (float): maximum viscosity 
            d (float, optional): grain size for the radiative conductivity. Defaults to 0.5.

        Raises:
            ValueError: If the number of phases exceed 7, throw an error. 
        """
        
        
        if number_phases>8:
            raise ValueError("The number of phases should not exceed 7")
        
        
        self.temp_ref           = _T_REF_  # Reference temperature [K]
        self.pres_ref           = _P_REF_     # Reference pressure [Pa]
        self.gas_constant = _GAS_CONSTANT_  # Universal gas constant [J/(mol K)]
        self.eta_min        = 1e18    # Min viscosity [Pas]
        self.eta_max        = eta_max    # Max viscosity [Pas]
        self.eta_def        = _ETA_DEF_    # Default viscosity [Pas]
        self.temp_scal         = 1.      # Default temperature scale
        self.pres_scal         = 1.      # Default Pressure scale
        self.id_ph             = np.zeros(number_phases, dtype=np.int32)
        self.a_rad              = 1.8 * (1 - np.exp(-d**1.3 / 0.15)) - (1 - np.exp(-d**0.5 / 5.0))
        self.b_rad              = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-d**3 / 10.0)
        self.temp_a            = 490.0 + 1850.0 * np.exp(-d**0.315 / 0.825) + 875.0 * np.exp(-d / 0.18)
        self.temp_b            = 2700.0 + 9000.0 * np.exp(-d**0.5 / 0.205)
        self.x_a            = 167.5 + 505.0 * np.exp(-d**0.5 / 0.85)
        self.x_b            = 465.0 + 1700.0 * np.exp(-d**0.94 / 0.175)

    
        # Explanation: For testing the pressure and t scal are set to be 1.0 -> so, the software is not performing any 
        # scaling operation. 
        # -> When the property are automatically scaled these value will be update automatically. 
        
        
        # Viscosity data
        # Diffusion creep
        self.e_dif         = np.zeros(number_phases, dtype=np.float64)              # Activation energy diffusion creep [J/mol]
        self.v_dif         = np.zeros(number_phases, dtype=np.float64)              # Activation volume diffusion creep [m^3/mol]
        self.b_dif         = np.zeros(number_phases, dtype=np.float64)              # Pre-exponential factor diffusion creep [Pa^-1 s^-1]
        
        # Dislocation creep
        self.e_dis       = np.zeros(number_phases, dtype=np.float64)              # Activation energy dislocaiton creep [J/mol]
        self.v_dis       = np.zeros(number_phases, dtype=np.float64)              # Activation volume dislocation creep [m^3/mol]
        self.b_dis       = np.zeros(number_phases, dtype=np.float64)              # Pre-exponential factor [Pa^-n s^-1]
        self.n          = np.ones (number_phases, dtype=np.float64)              # stress exponent  []
        self.eta        = np.zeros(number_phases, dtype=np.float64)              # constant viscosity [Pa s] - in case of constant viscosity 
        self.option_eta = np.zeros(number_phases, dtype=np.int32)                 # Option for viscosity calculation
        
        # Thermal properties
        self.c0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat capacity [J/mol/K]
        self.c1         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [J/mol/K^0.5]  -> CONVERTED INTO J/kg/K^0.5       
        self.c2         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K)/mol]    -> CONVERTED INTO J/kg/K^2
        self.c3         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3
        self.c4         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3
        self.c5         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3
        
        self.option_cp  = np.zeros(number_phases, dtype=np.int32)                 # Option for heat capacity calculation
        
        # Thermal conductivity 
        self.k0 = np.zeros(number_phases, dtype=np.float64)               # Reference heat conductivity [W/m/K]
        self.k_a = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity [1/Pa]
        self.k_b = np.zeros(number_phases, dtype=np.float64)               # exponent
        # Radiative heat transfer
        self.k_c = np.ones(number_phases, dtype=np.float64)               # Radiative heat transfer constant [W/m/K]
        self.k_d = np.zeros(number_phases, dtype=np.float64)               # Radiative heat transfer constant [W/m/K^2]
        self.k_e = np.ones((number_phases), dtype=np.float64)             # Radiative heat transfer polynomial coefficients [W/m/K^3]        
        self.k_f = np.zeros((number_phases), dtype = np.float64)           # 
        # Radiative heat transfer 
        self.radiative_conductivity = np.zeros(number_phases, dtype=np.float64)                 # Option for heat conductivity calculation
        
        self.radiogenic_heat = np.zeros(number_phases, dtype=np.float64)               # Radiogenic

        # Density parameters 
        self.alpha0     = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity coefficient [1/K]   
        self.alpha1     = np.zeros(number_phases, dtype=np.float64)               # Second-order expansivity [1/K^2]
        self.alpha2     = np.zeros(number_phases, dtype=np.float64)               # Second-order expansivity [1/Pa]
        self.kb         = np.zeros(number_phases, dtype=np.float64)               # Bulk modulus [Pa]                
        self.rho0       = np.zeros(number_phases, dtype=np.float64)               # Reference density [kg/m^3] {In case of constant density}
        self.option_rho = np.zeros(number_phases, dtype=np.int32)                 # Option for density calculation
        
        # Virtual Shear zone
        self.edis_wz = 0.0
        self.vdis_wz = 0.0
        self.n_wz = 0.0
        self.bdis_wz = 0.0
        self.eh2o_wz = 0.0
        self.bh2o_wz = 0.0
        self.ah2o_wz = 0.0
        self.vh2o_wz = 0.0
        self.water_cor = 0
        self.vis_con_fl = 0
        self.r_wz = 0.0
        self.eta_wz    = 0.0
        self.phi = 0.0
        self.cohesion  = 0.0
# ---
# ---
def generate_phase(pdb:PhaseDataBase,
                    id_ph:int = -100,
                    name_diffusion:str = 'Constant',
                    e_dif:float| None = None,
                    v_dif:float | None = None,
                    b_dif:float | None = None,
                    name_dislocation:float = 'Constant',
                    n:float | None = None,
                    e_dis:float| None = None,
                    v_dis:float| None = None,
                    b_dis:float| None = None,
                    cp:float               = 1250,
                    k:float                = 3.0,
                    rho0:float             = 3300,
                    eta:float              = 1e20,
                    name_capacity:str      = 'Constant',
                    name_conductivity:str  = 'Constant',
                    name_alpha:str         = 'Constant',
                    name_density:str       = 'PT',
                    alpha0:float = 3e-5,
                    radiogenic_heat:float = 0.0,
                    radiative_conductivity:float = 0,
                    pressure_dependency:int=1)     -> PhaseDataBase:
    """ Generate a phase with the specified properties and add it to the phase database.
    Args:
        pdb (PhaseDataBase): The phase database to which the new phase will be added.
        id_ph (int): The **id_phentifier** for the new phase.
        name_diffusion (str): The name of the diffusion creep rheology to use.
        e_dif (float): Activation energy for diffusion creep [J/mol].
        v_dif (float): Activation volume for diffusion creep [m^3/mol].
        b_dif (float): Pre-exponential factor for diffusion creep [Pa^-1 s^-1].
        name_dislocation (str): The name of the dislocation creep rheology to use.
        n (float): Stress exponent for dislocation creep.
        e_dis (float): Activation energy for dislocation creep [J/mol].
        v_dis (float): Activation volume for dislocation creep [m^3/mol].
        B_dis (float): Pre-exponential factor for dislocation creep [Pa^-n s^-1].
        Cp (float): Heat capacity [J/kg/K]. 
        k (float): Thermal conductivity [W/m/K].
        rho0 (float): Reference density [kg/m^3]. It is a mandatory parameters in any case. 
        eta (float): Constant viscosity [Pa s]. 
        name_capacity (str): The name of the heat capacity model to use.
        name_conductivity (str): The name of the thermal conductivity model to use.
        name_alpha (str): The name of the thermal expansivity model to use.
        name_density (str): The name of the density model to use.
        radio (float): Radiogenic heat production [W/kg].
        radio_flag (float): Flag for radiative conductivity production calculation.
    Returns:
        PhaseDataBase: The updated phase database with the new phase added.
        
    PhaseDataBase is a class that contains the properties of the different phases. 
    Few parameters are constanst common to all the phases (e.g. eta_max,T_ref)
    The other parameters are vector of 1xnumber_phases, where number_phases is the number of phases that we want to consid_pher in the model.
    """
    
    pdb.id_ph[id_ph-1] = id_ph
    id_ph = id_ph - 1
    if name_diffusion != 'Constant':
        data_buf = read_rheology(name_diffusion,0)
        pdb.e_dif[id_ph] = data_buf.e
        pdb.v_dif[id_ph] = data_buf.v
        pdb.b_dif[id_ph] = data_buf.b
    if e_dif is not None:
        pdb.e_dif[id_ph] = e_dif
    elif v_dif is not None:
        pdb.v_dif[id_ph] = v_dif
    elif b_dif is not None:
        pdb.b_dif[id_ph] = b_dif
    if name_dislocation != 'Constant':
        buf_data = read_rheology(name_dislocation,1)
        pdb.e_dis[id_ph] = buf_data.e
        pdb.v_dis[id_ph] = buf_data.v
        pdb.b_dis[id_ph] = buf_data.b
        pdb.n[id_ph]    = buf_data.n
    if n is not None:
        pdb.n[id_ph] = n
        if pdb.b_dis[id_ph] != 0.0:
            print('Warning: Stress pre-exponential factor has inconsistent measure [Pa^-ns^-1] wrt the original flow law')
    if e_dis is not None: # if the user specify the activation energy, overwrite the value of the flow law
        pdb.e_dis[id_ph] = e_dis
    if v_dis is not None: # if the user specify the activation volume, overwrite the value of the flow law  
        pdb.v_dis[id_ph] = v_dis
    if b_dis is not None: # if the user specify the pre-exponential factor, overwrite the value of the flow law
        pdb.b_dis[id_ph] = b_dis
    
    if name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta == -1e23:
        print_ph("Warning: Both diffusion and dislocation creep are set to be constant, but the viscosity is not specified")
        print_ph(f'The software will use the default viscosity value of {pdb.eta_def:.1e} Pa s. \
                 If you want to specify a different value, please set the eta parameter.')
    
    if name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta == -1e23:
            pdb.eta[id_ph] = pdb.eta_def
    elif name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta != -1e23:
        pdb.eta[id_ph] = eta # in case of constant viscosity, this value will be used. In case of non-constant viscosity, this value will be ignored.
    else:
        pdb.eta[id_ph] = 0.0 # in case of non-constant viscosity, this value will be ignored. I set it to 0.0 to avoid_ph any confusion.

    
    option_rheology = None
    if name_diffusion == 'Constant' and name_dislocation == 'Constant':
        option_rheology = 0
    elif name_dislocation == 'Constant':
        option_rheology = 1
    elif name_dislocation != 'Constant' and name_diffusion != 'Constant':
        option_rheology = 2
    elif name_dislocation != 'Constant' and name_diffusion == 'Constant':
        option_rheology = 3
    pdb.option_eta[id_ph] = option_rheology
    pdb.radiogenic_heat[id_ph] = radiogenic_heat

    if name_capacity == 'Constant':
        pdb.c0[id_ph] = cp
        pdb.c1[id_ph] = 0.0
        pdb.c2[id_ph] = 0.0
        pdb.c3[id_ph] = 0.0
        pdb.c4[id_ph] = 0.0
        pdb.c5[id_ph] = 0.0
    else:
        buf_cp = read_capacity(name_capacity)
        pdb.c0[id_ph] = buf_cp.c0
        pdb.c1[id_ph] = buf_cp.c1
        pdb.c2[id_ph] = buf_cp.c2
        pdb.c3[id_ph] = buf_cp.c3
        pdb.c4[id_ph] = buf_cp.c4
        pdb.c5[id_ph] = buf_cp.c5

    if name_conductivity == 'Constant':
        buf_data_diffusivity = LatticeDiffusivity()
        buf_data_diffusivity.a = 0.0
        buf_data_diffusivity.b = 0.0
        buf_data_diffusivity.c = 1.0
        buf_data_diffusivity.d = 0.0
        buf_data_diffusivity.e = 1.0
        buf_data_diffusivity.f = 0.0
        buf_data_diffusivity.g = 1.0
    else: 
        buf_data_diffusivity = read_diffusivity(name_conductivity)
    if pressure_dependency == 0:
        buf_data_diffusivity.f = 0.0


    pdb.k_a[id_ph] = buf_data_diffusivity.a
    pdb.k_b[id_ph] = buf_data_diffusivity.b
    pdb.k_c[id_ph] = buf_data_diffusivity.c
    pdb.k_d[id_ph] = buf_data_diffusivity.d
    pdb.k_e[id_ph] = buf_data_diffusivity.e
    pdb.k_f[id_ph] = buf_data_diffusivity.f
    pdb.k0[id_ph] = k * buf_data_diffusivity.g
    pdb.radiative_conductivity[id_ph] = radiative_conductivity
    # Density
    if name_alpha != 'Constant':
        alpha = read_expansivity(name_alpha)
    else:
        alpha = ThermalExpansivity()
        alpha.alpha0 = alpha0
        alpha.alpha1 = 0.0
        alpha.alpha2 = 0.0
    if pressure_dependency == 0:
        alpha.alpha2 = 0.0
        kb = 1e30
    else:
        kb = (2*100e9*(1+0.25))/(3*(1-0.25*2))  # Bulk modulus [Pa]

    pdb.alpha0[id_ph]     = alpha.alpha0
    pdb.alpha1[id_ph]     = alpha.alpha1
    pdb.alpha2[id_ph]     = alpha.alpha2
    pdb.kb[id_ph]         = kb  # Bulk modulus [Pa]
    pdb.rho0[id_ph]       = rho0
    if name_density == 'Constant':
        pdb.option_rho[id_ph] = np.int32(0)
    else:
        pdb.option_rho[id_ph] = np.int32(2)

    return pdb


def generate_phase_database(pressure_dependency:int,eta_max:float, phin:PhInput) -> PhaseDataBase:
    """_summary_

    Args:
        pressure_dependency (int): deactivate the pressure dependency
        eta_max (float): maximum viscosity
        phin (PhInput): data class containing the information of the input phases 

    Returns:
        PhaseDataBase: data class that contains the information of the phases. 
    """

    pdb = PhaseDataBase(6, eta_max=eta_max)

    if pressure_dependency == 0:
        print_ph(
            "Pressure dependency of the material properties is deactivated. The material properties will not depend on the pressure and will be only temperature dependent."
        )
    else:
        print_ph(
            "Pressure dependency of the material properties is activated. The material properties will depend on the pressure and temperature."
        )

    phase = Phase()
    dict_ph_in = asdict(phin)
    for i in dict_ph_in.keys():
        
        phase = getattr(phin, i)

        if not isinstance(phase,Phase):
            continue

        print_ph(f"Generating phase {phase.id_ph} : {i}, Phase Name : {phase.name_phase}")

        print_ph("-----Rheological Parameters------")
        print_ph(
            f"Diffusion law  : {phase.name_diffusion if hasattr(phase, 'name_diffusion') else 'Constant'}"
        )
        if phase.e_dif is not None:
            print_ph(f"   e_dif : {phase.e_dif} ")
        if phase.v_dif is not None:
            print_ph(f"   v_dif : {phase.v_dif} ")
        if phase.b_dif is not None:
            print_ph(f"   b_dif : {phase.b_dif} ")

        print_ph(
            f"Dislocation law: {phase.name_dislocation if hasattr(phase, 'name_dislocation') else 'Constant'}"
        )
        if phase.n is not None:
            print_ph(f"   n    : {phase.n} ")
        if phase.e_dis is not None:
            print_ph(f"   e_dis : {phase.e_dis} ")
        if phase.v_dis is not None:
            print_ph(f"   v_dis : {phase.v_dis} ")
        if phase.b_dis is not None:
            print_ph(f"   b_dis : {phase.B_dis} ")
        if phase.name_diffusion == "Constant" and phase.name_dislocation == "Constant":
            print_ph(f"   eta  : {phase.eta} [Pas] ")

        print_ph("-----------------------------------")

        print_ph("-----Thermal Parameters------")
        print_ph(
            f"Density law       : {phase.name_density if hasattr(phase, 'name_density') else 'Constant'}"
        )
        print_ph(
            f"Thermal capacity  : {phase.name_capacity if hasattr(phase, 'name_capacity') else 'Constant'}"
        )
        print_ph(
            f"Thermal conductivity : {phase.name_conductivity if hasattr(phase, 'name_conductivity') else 'Constant'}"
        )
        print_ph(
            f"Thermal expansivity : {phase.name_alpha if hasattr(phase, 'name_conductivity') else 'Constant'}"
        )
        print_ph(
            f"Radiogenic heating:  {phase.radiogenic_heat if phase.radiogenic_heat !=0.0 else 'Radiogenic heating is not active'}"
        )

        if hasattr(phase, "radiative_conductivity"):
            print_ph(f"   radiative conductivity flag : {phase.radiative_conductivity} ")
        if hasattr(phase, "rho0"):
            print_ph(f"   rho0 : {phase.rho0} ")
        print_ph("-----------------------------------")
        if phase.name_capacity == "Constant":
            print_ph(f"Heat capacity {phase.cp} J/kg/K")
            print_ph("-----------------------------------")
        if phase.name_conductivity == "Constant":
            print_ph(f"Thermal conductivity {phase.k} W/m/K")
            print_ph("-----------------------------------")
        print_ph("\n")

        pdb = generate_phase(
            pdb,
            phase.id_ph,
            radiative_conductivity=phase.radiative_conductivity if hasattr(phase, "radiative_conductivity") else 0.0,
            rho0=phase.rho0 if hasattr(phase, "rho0") else 3300,
            name_diffusion=(
                phase.name_diffusion if hasattr(phase, "name_diffusion") else "Constant"
            ),
            name_dislocation=(
                phase.name_dislocation
                if hasattr(phase, "name_dislocation")
                else "Constant"
            ),
            name_alpha=phase.name_alpha if hasattr(phase, "name_alpha") else "Constant",
            name_capacity=(
                phase.name_capacity if hasattr(phase, "name_capacity") else "Constant"
            ),
            name_density=(
                phase.name_density if hasattr(phase, "name_density") else "Constant"
            ),
            name_conductivity=(
                phase.name_conductivity
                if hasattr(phase, "name_conductivity")
                else "Constant"
            ),
            e_dif=phase.e_dif if hasattr(phase, "e_dif") else None,
            v_dif=phase.v_dif if hasattr(phase, "v_dif") else None,
            b_dif=phase.b_dif if hasattr(phase, "b_dif") else None,
            n=phase.n if hasattr(phase, "n") else None,
            e_dis=phase.e_dis if hasattr(phase, "e_dis") else None,
            v_dis=phase.v_dis if hasattr(phase, "v_dis") else None,
            b_dis=phase.b_dis if hasattr(phase, "b_dis") else None,
            eta=phase.eta if hasattr(phase, "eta") else 1e20,
            k=phase.k,
            radiogenic_heat=phase.radiogenic_heat,
            pressure_dependency=pressure_dependency,
        )


    pdb = fill_up_weakzone_data(
        ch=phin.shear_heating_disl_ch,
        phi=np.radians(phin.shear_heating_disl_phi),
        eta_wz=1e18,
        dislocation_creep=phin.shear_heating_disl_law,
        pdb=pdb,
    )

    return pdb

def fill_up_weakzone_data(ch:float = 10e6
                      ,phi: float = np.radians(5)
                      ,eta_wz: float = 1e20
                      ,dislocation_creep: str = 'Constant'
                      ,pdb:PhaseDataBase = None)->PhaseDataBase: 
    """Function that updates the data of the shear zone that mimick the subduction interface. 
    Args:
        ch (float, optional): Cohesion. Defaults to 10e6.
        phi (float, optional): Friction angle. Defaults to np.radians(5).
        eta_wz (float, optional): Viscosity. Defaults to 1e20.
        dislocation_creep (str, optional): Dislocation creep law. Defaults to 'Constant'.
        pdb (PhaseDataBase, optional): Phase Defaults to None.

    Returns:
        PhaseDataBase: updated phasedatabase
    """


    pdb.cohesion = ch
    pdb.eta_wz = eta_wz
    rheo = read_rheology(dislocation_creep,1)
    pdb.edis_wz = rheo.e
    pdb.vdis_wz = rheo.v
    pdb.bdis_wz = rheo.b
    pdb.n_wz = rheo.n
    pdb.r_wz = rheo.r
    pdb.water_cor = rheo.water_cor
    pdb.eh2o_wz = rheo.eh2o
    pdb.vh2o_wz = rheo.vh2o
    if dislocation_creep == 'Constant':
        pdb.vis_con_fl = 1
    pdb.phi = phi
    pdb.cohesion = ch
    return pdb

@dataclass(slots=True)
class RheologicalFlowLaw:

    e: float = field(init=False)        # activation energy [J/mol]
    v: float = field(init=False)        # activation volume [m^3/mol]
    n: float = field(init=False)        # stress exponent
    m: float = field(init=False)        # grain-size exponent
    d: float = field(init=False)        # grain size [m]
    b: float = field(init=False)   # pre-exponential factor (raw)
    b_si: str = field(init=False)
    b_or: float= field(init=False)
    f: str = field(init=False)          # experiment type ('NoCorrection'|'SimpleShear'|'UniAxial')
    mpa: int = field(init=False)
    ah2o: float = field(init=False)
    bh2o: float = field(init=False)
    eh2o: float = field(init=False)
    vh2o: float = field(init=False)
    r: float = field(init=False)        # water exponent
    water_correction: str = field(init=False)  # 'None'|'COH'|'Fugacity'
    water_cor : int = field(init=False)
    ref: str = field(init=False)

    def apply_correction(self) -> None:
        """Correct the pre-exponential factor.
        from any unit of measure, convert the pre-exponential factor to Pa^{-n}s^{-1}
        """

        b = self.b
        self.b_si = b
        f_code = _F_CODE[self.f]
        self.water_cor = _WATER_CODE[self.water_correction]
        if f_code == 1:                 # simple shear
            b *= 2 ** (self.n - 1)
        elif f_code == 2:               # uniaxial
            b *= (3 ** (self.n + 1) / 2) / 2
        if self.mpa == 1:               # MPa -> Pa
            b *= 10 ** (-self.n * 6)
        if self.water_cor == 1:                 # COH
            water_con = 1000 ** self.r
        elif self.water_cor == 2:               # fugacity
            water_con = (self.ah2o * self.bh2o
                         * np.exp(-(self.eh2o + self.vh2o * _P_REF_) / (_GAS_CONSTANT_ * _T_REF_))) ** self.r
        else:
            water_con = 1.0
        self.b = b * self.d ** (-self.m) * water_con

# ---
def check_data_base(tag:str,property_m:str)->str: 
    """_summary_

    Args:
        tag (str): _description_
        property_m (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        str: _description_
    """

    # Read the dictionary
    with open(Path(_PATH_DATA_BASE_,'Data_Base_dictionaries.yml'),encoding='utf-8') as dictionary_db: 
        file_db = yaml.safe_load(dictionary_db)
        db = file_db[property_m]
        keys = db.keys()
        if tag not in keys:
            print_ph(f'Valid_ph {property_m} Name')
            for i in keys:
                print_ph(i)
            raise ValueError(f'{tag} is not a {property_m}.')
        else:
            name = db[tag]
    return name

# ---
def read_rheology(tag:str,dis_dif:0)->RheologicalFlowLaw:
    """Read rheology 

    Args:
        tag (str): the name of the rheology from the input file
        type (0): flag indicating if the rheology is diffusion or dislocation

    Raises:
        ValueError:f'{tag} is not a rheology flow law.') -> The name used in the input
        file or in the pre-processing script is wrong. 

    Returns:
        Rheological_flow_law: class containing all the rheological data for a given rheology. 
        
    """

    name = check_data_base(tag,'Rheology_name')
    

    with open(Path(_PATH_DATA_BASE_,'rheology_data_base_raw.yml'),encoding='utf-8') as dictionary_db:
        file = yaml.safe_load(dictionary_db)
        common = file['Common']
        if dis_dif == 0:
            db_rheo = file['Diffusion_creep']
        elif dis_dif == 1: 
            db_rheo = file['Dislocation_creep']
        else:
            raise ValueError('Wrong flag value')

    buf = RheologicalFlowLaw()
    
    buf = update_ip_file(buf,common)
    
    buf = update_ip_file(buf,db_rheo[name])
    
    buf.apply_correction()

    return buf
# ---
@dataclass(slots=True)
class CpBufDB:
    """Temporary data class for the heat capacity
    """
    c0 :float = field(init=False)
    c1 :float = field(init=False)
    c2 :float = field(init=False)
    c3 :float = field(init=False)
    c4 :float = field(init=False)
    c5 :float = field(init=False)

# ---
def read_the_file_cp(d_cp:dict,i:int)->float:
    """read the empirical parameter of heat capacity

    Args:
        d_cp (dict): heat capacity block
        i (int): parameter of heat capacity

    Returns:
        float: the actual value
    """
    c = float(d_cp[f'c{i}'])
    
    return c
# ---
def read_capacity(name:str)->CpBufDB:
    """Read the internal heat capacity database

    Args:
        name (str): name of the heat capacity from the input

    Returns:
        CpBufDB: temporary data class for storing the information
    """
    
    def release_crust_data(db)->tuple[float,float,float,float,float,float,float]: 
        fr_ol = 0.15
        fr_au = 0.2
        fr_pg = 0.65
        
        c0 = fr_ol * read_the_file_cp(db['olivine'],0) \
            + fr_au * read_the_file_cp(db['augite'],0) \
            + fr_pg *  read_the_file_cp(db['plagioclase'],0)
        
        c1 = fr_ol * read_the_file_cp(db['olivine'],1) \
            + fr_au * read_the_file_cp(db['augite'],1) \
            + fr_pg *  read_the_file_cp(db['plagioclase'],1)
    
        c2 = fr_ol * read_the_file_cp(db['olivine'],2) \
            + fr_au * read_the_file_cp(db['augite'],2) \
            + fr_pg *  read_the_file_cp(db['plagioclase'],2)

        c3 = fr_ol * read_the_file_cp(db['olivine'],3) \
            + fr_au * read_the_file_cp(db['augite'],3) \
            + fr_pg *  read_the_file_cp(db['plagioclase'],3)

        c4 = fr_ol * read_the_file_cp(db['olivine'],4) \
            + fr_au * read_the_file_cp(db['augite'],4) \
            + fr_pg *  read_the_file_cp(db['plagioclase'],4)
        
        c5 = fr_ol * read_the_file_cp(db['olivine'],5) \
            + fr_au * read_the_file_cp(db['augite'],5) \
            + fr_pg *  read_the_file_cp(db['plagioclase'],5)

        

        return c0,c1,c2,c3,c4,c5
    
    def release_mantle_data(d_cp0,x:float,y:float,names:list)->tuple[float,float,float,float,float,float,float]: 
        
        c0 = x * read_the_file_cp(d_cp0[names[0]],0) + y * read_the_file_cp(d_cp0[names[1]],0)
        c1 = x * read_the_file_cp(d_cp0[names[0]],1) + y * read_the_file_cp(d_cp0[names[1]],1)
        c2 = x * read_the_file_cp(d_cp0[names[0]],2) + y * read_the_file_cp(d_cp0[names[1]],2)
        c3 = x * read_the_file_cp(d_cp0[names[0]],3) + y * read_the_file_cp(d_cp0[names[1]],3)
        c4 = x * read_the_file_cp(d_cp0[names[0]],4) + y * read_the_file_cp(d_cp0[names[1]],4)
        c5 = x * read_the_file_cp(d_cp0[names[0]],5) + y * read_the_file_cp(d_cp0[names[1]],5)

        return c0,c1,c2,c3,c4,c5
    
    
    buf_cp = CpBufDB()
    
    with open(Path(_PATH_DATA_BASE_,'thermal_properties_data_base_raw.yml'),encoding='utf-8') as dictionary_db:
        
        file_db = yaml.safe_load(dictionary_db)['Heat_capacity']
        if name == 'Oceanic_crust':
            (buf_cp.c0,
             buf_cp.c1,
             buf_cp.c2,
             buf_cp.c3,
             buf_cp.c4,
             buf_cp.c5) = release_crust_data(file_db)
            
        elif (name in ('Mantle_Bernard_1988_FO_FA','Mantle_Bernard_1988_FO','Mantle_Bernard_1988_FA') 
              or name in ('Mantle_Bernard_Ar_199x_FA','Mantle_Bernard_Ar_199x_FO','Mantle_Bernard_Ar_199x_FO_FA')):
            has_fo = "FO" in name
            has_fa = "FA" in name
            if has_fa and has_fo:
                x = 0.9
                y = 1-x
            elif has_fa:
                x = 0.0
                y = 1.0
            elif has_fo:
                x = 1.0
                y = 0.0
            
            if name in ('Mantle_Bernard_1988_FO_FA','Mantle_Bernard_1988_FO','Mantle_Bernard_1988_FA'):
                (buf_cp.c0,
                 buf_cp.c1,
                 buf_cp.c2,
                 buf_cp.c3,
                 buf_cp.c4,
                 buf_cp.c5) = release_mantle_data(file_db,names=['Bermann_Fosterite','Bermann_Fayalite'],x=x,y=y)
                     
            elif name in ('Mantle_Bernard_Ar_199x_FA','Mantle_Bernard_Ar_199x_FO','Mantle_Bernard_Ar_199x_FO_FA'):
                
                (buf_cp.c0,
                 buf_cp.c1,
                 buf_cp.c2,
                 buf_cp.c3,
                 buf_cp.c4,
                 buf_cp.c5) = release_mantle_data(file_db,names=['Bermann_Aranovich_Fosterite','Bermann_Aranovich_Fayalite'],x=x,y=y)
        else:
            raise ValueError(f'{name} is not a valid_ph Heat capacity')
        
    return buf_cp
# ---
@dataclass(slots=True) 
class LatticeDiffusivity:
    """Mutable dataclass buffer for storing the value of the lattice diffusivity
    """
    a :float = field(init=False)
    b :float = field(init=False)
    c :float = field(init=False)
    d :float = field(init=False)
    e :float = field(init=False)
    f :float = field(init=False)
    g :float = 0.0

# --- 
def read_diffusivity(tag:str)->LatticeDiffusivity:
    """Function that reads the data from the diffusivity 

    Args:
        tag (str): _description_

    Returns:
        LatticeDiffusivity: _description_
    """
    
    name = check_data_base(tag,'Thermal_diffusivity')

    buf_dif = LatticeDiffusivity()
    
    with open(Path(_PATH_DATA_BASE_,'thermal_properties_data_base_raw.yml'),encoding='utf-8') as dictionary_db:
        
        db_diff = yaml.safe_load(dictionary_db)['Thermal_diffusivity']
    
    buf_dif = update_ip_file(buf_dif,db_diff[name])
        
    return buf_dif
# --- 
@dataclass(slots=True) 
class ThermalExpansivity:
    alpha0 :float = field(init=False)
    alpha1 :float = field(init=False)
    alpha2 :float = field(init=False)

# --- 
def read_expansivity(tag:str)->ThermalExpansivity:
    
    name = check_data_base(tag,'Thermal_expansivity')

    buf_dif = ThermalExpansivity() 
    
    with open(Path(_PATH_DATA_BASE_,'thermal_properties_data_base_raw.yml'),encoding='utf-8') as dictionary_db:
        
        db_diff = yaml.safe_load(dictionary_db)['Thermal_expansivity']
    
    buf_dif = update_ip_file(buf_dif,db_diff[name])
        
    return buf_dif
# ---
#-----------------------------------------------------------------------------
@njit
def heat_conductivity(pdb:PhaseDataBase
                      ,temp:NDArray[np.float64]|np.float64
                      ,pres:NDArray[np.float64]|np.float64
                      ,rho:NDArray[np.float64]|np.float64
                      ,cp:NDArray[np.float64]|np.float64
                      ,ph:int)->NDArray[np.float64]:

    """Compute the heat conductivity for a given Pressure and Temperature
    Args:
        pdb (PhaseDataBase): Phase Data Base containing the material properties as numpy arrays, indexed by phase ID
        T (NDArray[np.float64] | np.float64): Temperature field as a numpy array
        p (NDArray[np.float64] | np.float64): Pressure field as a numpy array
        rho (NDArray[np.float64] | np.float64): Density field as a numpy array
        Cp (NDArray[np.float64] | np.float64): Heat capacity field as a numpy array
        ph (int): phase ID for which to compute the conductivity, used to index the material properties from the PhaseDataBase
    Returns:
        NDArray[np.float64]: array containing the heat conductivity
    
    Function to compute the heat conductivity for a given Pressure and Temperature, used for the post-processing of the thermal properties.
    It is used for computing the initial conductivity field for the oceanic plate thermal boundary condition.
    """

    k_rad = pdb.a_rad * np.exp(-(temp-pdb.temp_a)**2/ (2*pdb.x_a ** 2 )) + pdb.b_rad * np.exp(-(temp - pdb.temp_b)**2 / (2* pdb.x_b**2))

    kappa_lat = pdb.k_a[ph] + pdb.k_b[ph] * np.exp(-(temp-pdb.temp_ref)/pdb.k_c[ph]) + pdb.k_d[ph] * np.exp(-(temp-pdb.temp_ref)/pdb.k_e[ph])
    
    kappa_p   = np.exp(pdb.k_f[ph] * pres)

    k = pdb.k0[ph] + kappa_lat * kappa_p * cp * rho + k_rad * pdb.radiative_conductivity[ph]

    return k
#---------------------------------------------------------------------------------
@njit
def density(pdb:PhaseDataBase
            ,temp:NDArray[np.float64]|np.float64
            ,pres:NDArray[np.float64]|np.float64
            ,ph:int)->NDArray[np.float64]:
    """Compute the density for a given Pressure and Temperature, used for oceanic plate thermal boundary condition.
    
    Args:
        pdb (PhaseDataBase): Phase Data Base containing the material properties as numpy arrays, indexed by phase ID
        T (NDArray[np.float64] | np.float64): Temperature field as a numpy array
        p (NDArray[np.float64] | np.float64): Pressure field as a numpy array
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
        rho     = rho_0 * (1 - np.exp(- pres * pdb.alpha2[ph])
                           *( pdb.alpha0[ph] * (temp - pdb.temp_ref) + (pdb.alpha1[ph]/2.) * ( temp**2 - pdb.temp_ref**2 )))
        if pdb.option_rho[ph] == 2:
            # calculate the pressure dependence of the density
            kb = pdb.kb[ph]
            rho = rho * np.exp(pres/kb)
    
    return rho

#---------------------------------------------------------------------------------
@njit
def heat_capacity(pdb:PhaseDataBase
                  ,temp:NDArray[np.float64]|np.float64
                  ,ph:int)->NDArray[np.float64]|np.float64:
    """Compute heat capacity

    Args:
        pdb (PhaseDataBase): Phase Data Base containing the material properties as numpy arrays, indexed by phase ID
        T (NDArray[np.float64] | np.float64): Temperature field as a numpy array
        ph (int): phase ID for which to compute the heat capacity, used to index the material properties from the PhaseDataBase

    Returns:
        NDArray[np.float64] | np.float64: array containing the heat capacity
    """

    cp = pdb.c0[ph] + pdb.c1[ph] * (temp**(-0.5)) + pdb.c2[ph] * temp**(-2.0) + pdb.c3[ph] * (temp**(-3.)) + pdb.c4[ph]* temp + pdb.c5[ph] * temp**2
    
    return cp

# ---
@njit
def compute_thermal_properties(pdb:PhaseDataBase,temp:NDArray[np.float64]|np.float64
                               ,pres:NDArray[np.float64]|np.float64,
                               ph:int)->tuple[NDArray[np.float64]|np.float64,NDArray[np.float64]|np.float64,NDArray[np.float64]|np.float64]:
    """_summary_

    Args:
        pdb (PhaseDataBase): _description_
        temp (NDArray[np.float64] | np.float64): _description_
        pres (NDArray[np.float64] | np.float64): _description_
        ph (int): _description_

    Returns:
        tuple[NDArray[np.float64]|np.float64,NDArray[np.float64]|np.float64,NDArray[np.float64]|np.float64]: _description_
    """
    
    cp   = heat_capacity(pdb,temp,ph)
    rho  = density(pdb,temp,pres,ph)
    k    = heat_conductivity(pdb,temp,pres,rho,cp,ph)
    
    return cp, rho, k

# ---

def compute_effective_stress(eps: float, eta: float) -> float:
    """
    Function to compute the effective stress using the reference viscosity and actual strain rate
    eps: strain rate
    eta: reference viscosity
    """
    return 2 * eta * eps


def effective_stress(tau_fr: NDArray[np.float64], tau_v: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Args:
        tau_fr (NDArray[np.float64]): array containing the frictional stress
        tau_v (NDArray[np.float64]): viscous stresses

    Returns:
        NDArray[np.float64]: effective stress
    """
    return tau_v * np.tanh(tau_fr / tau_v)


def compute_tau_fr(pres: NDArray[np.float64], phi: float) -> NDArray[np.float64]:
    """_summary_

    Args:
        pres (NDArray[np.float64]): lithostatic pressure
        phi (float): friction angle

    Returns:
        NDArray[np.float64]: failure shear stress
    """
    return pres * np.sin(np.radians(phi))


def convert_velocity_strain_rate(wz: float, velocity: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Args:
        wz (float): thickness of the weak zone
        velocity (NDArray[np.float64]): velocity of convergence

    Returns:
        strain rate: array with strain rate
    """
    scal = 1e-2 / 365.25 / 60 / 60 / 24
    v = velocity * scal
    eps = v / wz
    return eps






def test_phase_pdb():

    # Test rheology 
    rqrtz = read_rheology('Wet_Quartzite_2001_Dislocation_creep',1)
    rolivinedsl = read_rheology('Hirth_wet_Dislocation_creep',1)
    rolivinedff = read_rheology('Hirth_wet_Diffusion_creep',0)
    # Test Heat Capacity 
    cp0 = read_capacity('Mantle_Bernard_Ar_199x_FA')
    cp1 = read_capacity('Mantle_Bernard_Ar_199x_FO')
    cp2 = read_capacity('Mantle_Bernard_Ar_199x_FO_FA')    
    cp3 = read_capacity('Mantle_Bernard_1988_FA')
    cp4 = read_capacity('Mantle_Bernard_1988_FO')
    cp5 = read_capacity('Mantle_Bernard_1988_FO_FA')
    cp6 = read_capacity('Crust')
    # Thermal diffusivity 
    dif_0 = read_diffusivity('Mantle_Richards_2018')
    dif_1 = read_diffusivity('Crust_Richards_2018')
    # Thermal expansivity
    alpha_0 = read_expansivity('Mantle')
    alpha_1 = read_expansivity('Oceanic_crust')

        
    return 0

def compute_effective_stress_rheological(A:RheologicalFlowLaw,eps:float,pres:NDArray[np.float64],temp:NDArray[np.float64])->NDArray[np.float64]:
    
    # strain indipendent  
    cds = A.b * np.exp(-(A.e + pres * A.v)/(_GAS_CONSTANT_ * temp)) 
    # compute tau guess
    
   
    water = np.exp(-(A.eh2o+pres *A.vh2o)/(_GAS_CONSTANT_ * temp))/np.exp(-(A.eh2o+_P_REF_*A.vh2o)/(_GAS_CONSTANT_ * _T_REF_))
    cds = cds * water ** (A.r)
    
    tau_eff_v = cds ** (-1/A.n) * eps**(1/A.n)
    
    
    return tau_eff_v 


def compute_bn(eps_ref:float,tau_ref:float,n)->float:
    
    return eps_ref * tau_ref**(-n)

def compute_stress(eps:NDArray[np.float64],bn:float,n:float)->NDArray[np.float64]:


    return bn**(-1/n)*eps**(1/n)




@dataclass
class Profile_Slab: 
    z: InitVar[NDArray[np.float32]]
    a: np.float32
    k:float = 3.1
    Tp: float = 1350+273.15
    kappa: float = 1e-7
    crd: NDArray[np.float32] = field(init=False)
    temp_slab: NDArray[np.float32] = field(init=False)
    dip : NDArray[np.float32] = field(init=False)
    length :NDArray[np.float32] =field(init=False)
    def __post_init__(self,z:NDArray[np.float32])->None:
        """ post init routine: generate the quadratic surface
        of the slab using a vector (1,n) and the curvature of
        the parabula.

        Args:
            z (NDArray[np.float32]): z coordinates

        """
        # Compute the coordinate x
        x = np.abs(np.sqrt(z/self.a))
        # Compute the dip angle (radians)
        dip = self.compute_dip(x,z) # len = len(x)-1
        # Compute the cumulative distance
        x,z,dist = self.cumulative_distance(x,z)# len = len(x)-1
        # Create the main vector
        self.crd = np.zeros([2,len(x)],dtype=np.float32)
        self.dip = dip 
        self.length = dist * 1e3
        # Correct the coordinate 
        # -> Necessary because the new x,z has as initial coordinate (x[0]+x[1])/2
        self.crd[0,:] = (x - x[0]) * 1e3
        self.crd[1,:] = (z - z[0]) * 1e3
        return None
    @staticmethod
    def cumulative_distance(x:NDArray[np.float32],z:NDArray[np.float32])->tuple[NDArray[np.float32],NDArray[np.float32],NDArray[np.float32]]:

        dist = np.cumsum(np.sqrt(np.diff(x)**2+np.diff(z)**2))
        x_m = (x[:-1]+x[1:])/2 
        z_m = (z[:-1]+z[1:])/2 
        
        return x_m,z_m,dist
    @staticmethod 
    def compute_dip(x:NDArray[np.float32],z:NDArray[np.float32])->NDArray[np.float32]:
        
        dx = np.diff(x)
        dz = np.diff(z)
        dip = np.arctan2(dz,dx)
    
        return dip
    def compute_slab_surface_temperature(self,gamma:float,vel:float,age:float,bq:float)->None:
        vel = convert_velocity(vel)
        age = convert_age(age)
        Pe = (vel * self.crd[1,:]**2)/self.kappa/self.length
        Q = (self.k*self.Tp)/(np.sqrt(np.pi * self.kappa * (age + self.length/vel)))
        Sq = np.cos(self.dip)+ bq*np.sqrt(Pe)

        self.temp_slab = 273.15 + (Q*self.crd[1,:])/(self.k*Sq)     
    

def convert_velocity(vel:NDArray[np.float64])->NDArray[np.float64]:
    
    vel = vel * 1e-2/365.25/60/60/24
    
    return vel 

def convert_age(age:float)->float: 
    
    age = age * 1e6 * 365.25*60*60*24
    
    return age


    
         

def shear_heating_slab_interface():
    from scipy.special import gamma
    import matplotlib.pyplot as plt
    n_fk = 4.0
    phi = 5.0
    wz = 500
    slab = Profile_Slab(np.linspace(0,80,1000),3.5/1e3)   
    m = 3
    bq = gamma(m/2+1)/gamma(m/2+1/2)
    vel  = np.linspace(1.0,10,10)
    eps = convert_velocity(vel)/wz
    rqrzt =  read_rheology('Wet_Quartzite_2001_Dislocation_creep',1)
    bn = compute_bn(1e-11,120e6,n_fk)
    pres = slab.crd[1,:]*3300*9.81
    tau_fr = pres*np.sin(np.radians(phi))
    tau_min = 0.01e6
    
    fig = plt.figure()
    ax = fig.gca()
    
    fig2 = plt.figure()
    ax1 = fig2.gca()
    
    fig3 = plt.figure()
    ax2 = fig3.gca()
    
    for i in enumerate(eps): 
        slab.compute_slab_surface_temperature(bq,vel[i[0]],30,bq)
        tau_v = compute_stress(i[1],bn,n_fk)
        tau_v_rh = compute_effective_stress_rheological(rqrzt,i[1],pres,slab.temp_slab)
        p_crit = tau_v/np.sin(np.radians(phi))
        p_norm = (pres-p_crit)/p_crit 
        tau_v_dmn = tau_min+(tau_v-tau_min)*np.exp(-5.0*p_norm)
        tau_eff_fk = tau_v_dmn * np.tanh(tau_fr/tau_v_dmn)
        tau_eff_rh = tau_v_rh * np.tanh(tau_fr/tau_v_rh)
        ax.plot(tau_eff_fk/1e6,-slab.crd[1,:]/1e3, c='forestgreen',linewidth=i[0]*0.1+0.1)
        ax.plot(tau_eff_rh/1e6,-slab.crd[1,:]/1e3,c='firebrick',linewidth=i[0]*0.1+0.1)
        ax1.plot(tau_eff_rh*convert_velocity(vel[i[0]]),-slab.crd[1,:]/1e3,c='firebrick',linewidth=i[0]*0.1+.1)
        ax1.plot(tau_eff_fk*convert_velocity(vel[i[0]]),-slab.crd[1,:]/1e3,c='forestgreen',linewidth=i[0]*0.1+.1)
        ax2.plot(tau_eff_rh/pres,-slab.crd[1,:]/1e3,c='firebrick',linewidth=i[0]*0.1+.1)
        ax2.plot(tau_eff_fk/pres,-slab.crd[1,:]/1e3,c='forestgreen',linewidth=i[0]*0.1+.1)
        
        
        print(f'max tau fake rheology {np.nanmax(tau_eff_fk/1e6):.3f}, mean {np.nanmean(tau_eff_fk/1e6):.3f}')
        print(f'max tau real rheology {np.nanmax(tau_eff_rh/1e6):.3f}, mean {np.nanmean(tau_eff_rh/1e6):.3f}')
    ax.set_ylabel('Depth [km]')
    ax1.set_ylabel('Depth [km]')
    ax2.set_ylabel('Depth [km]')
    ax2.set_xlabel(r'$\mu_{eff}$ []')
    ax.set_xlabel(r'$\tau$ [MPa]')
    ax1.set_xlabel(r'$log_{10} \Psi$ [W/m2]')
    ax1.set_xscale('log')
    fig.savefig('tau_eff.png')
    fig2.savefig('shear_heating.png')
    fig3.savefig('effective_mu.png')
    print('')


    return 0



if __name__ == '__main__':
    
    shear_heating_slab_interface()
    
    pass