"""Modules"""
from numba import float64,int32
from numba.experimental import jitclass
import numpy as np
from stonedfenicsx.utils import print_ph
from stonedfenicsx.config.input_parser import Phase,PhInput,update_ip_file
from pathlib import Path
import yaml
from dataclasses import dataclass, field

# Global variable
_F_CODE = {"None": 0, "Simpleshear": 1, "Uniaxial": 2}
_WATER_CODE = {"None": 0, "COH": 1, "Fugacity": 2}
_PATH_DATA_BASE_ = Path(Path(__file__).parents[0],'material_properties_data_base')
_GAS_CONSTANT_ = 8.3145
_P_REF_ = 1e5 
_T_REF_ = 298.15

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
    ("temp_A",float64), # T [K]
    ("temp_B",float64), # T [K]
    ("x_a",float64), # T [K]
    ("x_b",float64), # T [K]
    ("radiative_conductivity",float64[:]), # radio flag 

    
    ("id", int32[:]),              # phase number
    
    # Constants
    ("temp_ref", float64),      # Reference temperature [K]
    ("pres_ref", float64),      # Reference pressure [Pa]
    ("gas_constant", float64),          # Gas constant [J/mol/K]
    ("temp_Scal",float64),     # T_scal [K] -> Important, as within the exponential soul of diffusion/dislocation creep  R is in mol I don't know how to make dimensionless
    ("pres_Scal",float64),     # Same reason as before [Pa]
    ("eta_min",float64),    # minimum viscosity [Pas]
    ("eta_max",float64),    # max viscosity [Pas]
    ("eta_def",float64),    # default viscosity [Pas]
    
    # Weak Zone Parameters 
    ('edis_wz',float64),
    ('vdis_wz',float64),
    ('n_wz',float64),
    ('bdis_wz',float64),
    ('eH20_wz',float64),
    ('bH20_wz',float64),
    ('aH20_wz',float64),
    ('bH20_wz',float64),
    ('water_cor',int32),
    ('r_wz',float64),
    ('vis_con_fl',int32),
    ('eta_wz',float64),
    ("phi",float64),
    ('cohesion',float64),
]

#-----------------------------------------------------------------------------------------------------------
@jitclass(spec_phase)
class PhaseDataBase:
    """Phase database: 
    
    """
    def __init__(self
                 ,number_phases:int
                 ,eta_max:float
                 ,d=0.5):
        # Initialize individual fields as arrays
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
        
        
        self.temp_ref           = _P_REF_  # Reference temperature [K]
        self.pres_ref           = _T_REF_     # Reference pressure [Pa]
        self.gas_constant = _GAS_CONSTANT_  # Universal gas constant [J/(mol K)]
        self.eta_min        = 1e18    # Min viscosity [Pas]
        self.eta_max        = 1e25    # Max viscosity [Pas]
        self.eta_def        = 1e21    # Default viscosity [Pas]
        self.temp_scal         = 1.      # Default temperature scale
        self.pres_scal         = 1.      # Default Pressure scale 
        self.id             = np.zeros(number_phases, dtype=np.int32)
        self.a_rad              = 1.8 * (1 - np.exp(-d**1.3 / 0.15)) - (1 - np.exp(-d**0.5 / 5.0))
        self.b_rad              = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-d**3 / 10.0)
        self.temp_a            = 490.0 + 1850.0 * np.exp(-d**0.315 / 0.825) + 875.0 * np.exp(-d / 0.18)
        self.temp_b            = 2700.0 + 9000.0 * np.exp(-d**0.5 / 0.205)
        self.x_a            = 167.5 + 505.0 * np.exp(-d**0.5 / 0.85)
        self.x_b            = 465.0 + 1700.0 * np.exp(-d**0.94 / 0.175)
        self.eta_max = eta_max

    
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
        self.cohesion  = 0
#---------------------------------------------------------------------------------
def generate_phase_database(pressure_dependency:int,eta_max:float, phin:PhInput) -> PhaseDataBase:

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
    dict_ph_in = Phin.__dict__
    for i, ph in dict_ph_in.items():
        phase = getattr(Phin, i)
        print_ph(f"Generating phase {phase.id} : {i}, Phase Name : {phase.name_phase}")

        print_ph("-----Rheological Parameters------")
        print_ph(
            f"Diffusion law  : {phase.name_diffusion if hasattr(phase, 'name_diffusion') else 'Constant'}"
        )
        if phase.Edif != -1e23:
            print_ph(f"   Edif : {phase.Edif} ")
        if phase.Vdif != -1e23:
            print_ph(f"   Vdif : {phase.Vdif} ")
        if phase.Bdif != -1e23:
            print_ph(f"   Bdif : {phase.Bdif} ")

        print_ph(
            f"Dislocation law: {phase.name_dislocation if hasattr(phase, 'name_dislocation') else 'Constant'}"
        )
        if phase.n != -1e23:
            print_ph(f"   n    : {phase.n} ")
        if phase.Edis != -1e23:
            print_ph(f"   Edis : {phase.Edis} ")
        if phase.Vdis != -1e23:
            print_ph(f"   Vdis : {phase.Vdis} ")
        if phase.Bdis != -1e23:
            print_ph(f"   Bdis : {phase.Bdis} ")
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
            f"Radiogenic heating:  {phase.Hr if phase.Hr !=0.0 else 'Radiogenic heating is not active'}"
        )

        if hasattr(phase, "radio_flag"):
            print_ph(f"   radiative conductivity flag : {phase.radio_flag} ")
        if hasattr(phase, "rho0"):
            print_ph(f"   rho0 : {phase.rho0} ")
        print_ph("-----------------------------------")
        if phase.name_capacity == "Constant":
            print_ph(f"Heat capacity {phase.Cp} J/kg/K")
            print_ph("-----------------------------------")
        if phase.name_conductivity == "Constant":
            print_ph(f"Thermal conductivity {phase.k} W/m/K")
            print_ph("-----------------------------------")
        print_ph("\n")

        pdb = _generate_phase(
            pdb,
            phase.id,
            radio_flag=phase.radio_flag if hasattr(phase, "radio_flag") else 0.0,
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
            Edif=phase.Edif if hasattr(phase, "Edif") else -1e23,
            Vdif=phase.Vdif if hasattr(phase, "Vdif") else -1e23,
            Bdif=phase.Bdif if hasattr(phase, "Bdif") else -1e23,
            n=phase.n if hasattr(phase, "n") else -1e23,
            Edis=phase.Edis if hasattr(phase, "Edis") else -1e23,
            Vdis=phase.Vdis if hasattr(phase, "Vdis") else -1e23,
            Bdis=phase.Bdis if hasattr(phase, "Bdis") else -1e23,
            eta=phase.eta if hasattr(phase, "eta") else 1e20,
            k=phase.k,
            radio=phase.Hr,
            Pressure_dependency=IP.Pressure_dependency,
        )

        # Update the rheological data of the virtual weak zone.

    pdb = fill_up_weakzone_data(
        ch=IP.cohesion,
        phi=np.radians(IP.phi),
        eta_wz=IP.eta_wz,
        dislocation_creep=IP.dislocation_creep_wz,
        pdb=pdb,
    )

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
    b: float = field(init=False)        # corrected pre-exponential factor
    ref: str = field(init=False)

    def apply_correction(self) -> None:

        b = self.b
        self.b_si = b 
        f_code = _F_CODE[self.f]
        w_code = _WATER_CODE[self.water_correction]
        if f_code == 1:                 # simple shear
            b *= 2 ** (self.n - 1)
        elif f_code == 2:               # uniaxial
            b *= (3 ** (self.n + 1) / 2) / 2
        if self.mpa == 1:               # MPa -> Pa
            b *= 10 ** (-self.n * 6)
        if w_code == 1:                 # COH
            water_con = 1000 ** self.r
        elif w_code == 2:               # fugacity
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
            print_ph(f'Valid {property_m} Name')
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


def _generate_phase(pdb:PhaseDataBase,
                    id:int                 = -100,
                    name_diffusion:str     = 'Constant',
                    e_dif:float             = -1e23, 
                    v_dif:float             = -1e23,
                    b_dif:float             = -1e23, 
                    name_dislocation:float = 'Constant',
                    n:float                = -1e23,
                    e_dis:float             = -1e23,
                    v_dis:float             = -1e23, 
                    b_dis:float             = -1e23, 
                    cp:float               = 1250,
                    k:float                = 3.0,
                    rho0:float             = 3300,
                    eta:float              = 1e20,
                    name_capacity:str      = 'Constant',
                    name_conductivity:str  = 'Constant',
                    name_alpha:str         = 'Constant',
                    name_density:str       = 'PT',
                    radiogenic_heat:float = 0.0,                    
                    radiative_conductivity:float = 0,
                    pressure_dependency:int=1)     -> PhaseDataBase:
    """ Generate a phase with the specified properties and add it to the phase database.
    Args:
        pdb (PhaseDataBase): The phase database to which the new phase will be added.
        id (int): The **identifier** for the new phase.
        name_diffusion (str): The name of the diffusion creep rheology to use.
        Edif (float): Activation energy for diffusion creep [J/mol].
        Vdif (float): Activation volume for diffusion creep [m^3/mol].
        Bdif (float): Pre-exponential factor for diffusion creep [Pa^-1 s^-1].
        name_dislocation (str): The name of the dislocation creep rheology to use.
        n (float): Stress exponent for dislocation creep.
        Edis (float): Activation energy for dislocation creep [J/mol].
        Vdis (float): Activation volume for dislocation creep [m^3/mol].
        Bdis (float): Pre-exponential factor for dislocation creep [Pa^-n s^-1].
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
    The other parameters are vector of 1xnumber_phases, where number_phases is the number of phases that we want to consider in the model.
    """
    
    
    if (name_diffusion not in Dic_rheo) or (name_dislocation not in Dic_rheo):
        print_ph('The avaiable options are: ')
        for k in Dic_rheo: 
            print_ph('%s; '%k)
        
        if (name_diffusion not in Dic_rheo):
            raise ValueError("Error: %s is not a heat a Rheological option"%name_diffusion)
        else: 
            raise ValueError("Error: %s is not a heat a Rheological option"%name_dislocation)

            
    
    pdb.id[id-1] = id 
    id = id - 1 
    if name_diffusion != 'Constant':
        data_buf = _check_rheological(name_diffusion)
        pdb.e_dif[id] = data_buf.e
        pdb.v_dif[id] = data_buf.v
        pdb.b_dif[id] = data_buf.b 
    if e_dif != -1e23: 
        pdb.Edif[id] = e_dif
    elif v_dif !=-1e23:  
        pdb.Vdif[id] = v_dif
    elif b_dif != -1e23:
        pdb.Bdif[id] = b_dif 
    if name_dislocation != 'Constant':
        buf_data = _check_rheological(name_dislocation)
        pdb.e_dis[id] = buf_data.e
        pdb.v_dis[id] = buf_data.v
        pdb.b_dis[id] = buf_data.b
        pdb.n[id]    = buf_data.n
    if n!= -1e23: 
        pdb.n[id] = n 
        if pdb.b_dis[id] != 0.0: 
            print('Warning: Stress pre-exponential factor has inconsistent measure [Pa^-ns^-1] wrt the original flow law')
    if e_dis != -1e23: # if the user specify the activation energy, overwrite the value of the flow law
        pdb.Edis[id] = e_dis
    if v_dis !=-1e23: # if the user specify the activation volume, overwrite the value of the flow law  
        pdb.Vdis[id] = v_dis
    if b_dis != -1e23: # if the user specify the pre-exponential factor, overwrite the value of the flow law
        pdb.b_dis[id] = b_dis
    
    if name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta == -1e23:
        print_ph("Warning: Both diffusion and dislocation creep are set to be constant, but the viscosity is not specified")
        print_ph(f'The software will use the default viscosity value of {pdb.eta_def:.1e} Pa s. \
                 If you want to specify a different value, please set the eta parameter.')
    
    if name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta == -1e23:
            pdb.eta[id] = pdb.eta_def
    elif name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta != -1e23:
        pdb.eta[id] = eta # in case of constant viscosity, this value will be used. In case of non-constant viscosity, this value will be ignored.
    else:
        pdb.eta[id] = 0.0 # in case of non-constant viscosity, this value will be ignored. I set it to 0.0 to avoid any confusion.

    
    option_rheology = None
    if name_diffusion == 'Constant' and name_dislocation == 'Constant': 
        option_rheology = 0
    elif name_dislocation == 'Constant': 
        option_rheology = 1
    elif name_dislocation != 'Constant' and name_diffusion != 'Constant': 
        option_rheology = 2
    elif name_dislocation != 'Constant' and name_diffusion == 'Constant': 
        option_rheology = 3
    pdb.option_eta[id] = option_rheology
    pdb.radiogenic_heat[id] = radiogenic_heat
    # Heat capacity
    # To remove
    if name_capacity not in Dic_Cp:
        print_ph('The avaiable options are: ')
        for k in Dic_Cp: 
            print_ph('%s; '%k)
        
        raise ValueError(f"Error Phase id = {id:d}: {name_capacity} is not a heat Conductivity option")
    
    pdb.C0[id],pdb.C1[id],pdb.C2[id],pdb.C3[id],pdb.C4[id],pdb.C5[id] = release_heat_capacity_parameters(Dic_Cp[name_capacity], Cp)
    
    
    
    if name_conductivity not in Dic_conductivity:
        print_ph('The avaiable options are: ')
        for k in Dic_conductivity: 
            print_ph('%s; '%k)
        
        raise ValueError(f"Error Phase id = {id:d}: {name_conductivity} is not a heat Conductivity option")
    
    buf_data_diffusivity = _check_diffusivity(name_conductivity)
    if pressure_dependency == 0: 
        buf_data_diffusivity.f = 0.0
        
    
    pdb.k_a[id] = buf_data_diffusivity.a
    pdb.k_b[id] = buf_data_diffusivity.b
    pdb.k_c[id] = buf_data_diffusivity.c
    pdb.k_d[id] = buf_data_diffusivity.d
    pdb.k_e[id] = buf_data_diffusivity.e
    pdb.k_f[id] = buf_data_diffusivity.f
    pdb.k0[id] = k * buf_data_diffusivity.g
    pdb.radiative_conductivity[id] = radiative_conductivity
    
    # Density
    
    if name_alpha not in Dic_alpha:
        print_ph('The avaiable options are: ')
        for k in Dic_conductivity: 
            print_ph('%s; '%k)

        raise ValueError(f"Error Phase id = {id:d}: {name_conductivity} is not a heat Conductivity option")
    alpha = _check_alpha(name_alpha)
    if pressure_dependency == 0:
        alpha.alpha2 = 0.0
        kb = 1e30
    else: 
        kb = (2*100e9*(1+0.25))/(3*(1-0.25*2))  # Bulk modulus [Pa]
    
    pdb.alpha0[id]     = alpha.alpha0
    pdb.alpha1[id]     = alpha.alpha1
    pdb.alpha2[id]     = alpha.alpha2
    pdb.kb[id]         = kb  # Bulk modulus [Pa]
    pdb.rho0[id]       = rho0
    if name_density == 'Constant':
        pdb.option_rho[id] = np.int32(0)
    else: 
        pdb.option_rho[id] = np.int32(2)
    
    return pdb

def test_phase_pdb():
    
    # Test rheology 
    rqrtz = read_rheology('Wet_Quartzite_2001_Dislocation_creep',1)
    rolivinedsl = read_rheology('Hirth_wet_Dislocation_creep',1)
    rolivinedff = read_rheology('Hirth_wet_Diffusion_creep',0)

    
    
    return 0

if __name__ == '__main__':
    
    test_phase_pdb()
    
    pass