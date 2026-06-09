from numba import float64,int32
from numba.experimental import jitclass
import numpy as np
from stonedfenicsx.utils import print_ph
from stonedfenicsx.config.input_parser import Phase,PhInput,Input

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
    
    ("A",float64), # Ref conductivity A 
    ("B",float64), # Ref conductivity B 
    ("T_A",float64), # T [K]
    ("T_B",float64), # T [K]
    ("x_A",float64), # T [K]
    ("x_B",float64), # T [K]
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
        
        
        self.temp_ref           = 298.15  # Reference temperature [K]
        self.pres_ref           = 1e5     # Reference pressure [Pa]
        self.gas_constant             = 8.3145  # Universal gas constant [J/(mol K)]
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
def generate_phase_database(Pressure_dependency:int,eta_max:float, Phin:PhInput) -> PhaseDataBase:
    from stonedfenicsx.material_property.phase_db import fill_up_weakzone_data

    pdb = PhaseDataBase(6, eta_max=IP.eta_max)

    if IP.Pressure_dependency == 0:
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

#----------------------------------------------------------------------------
class Rheological_flow_law():
    """
    Class that contains the rheological flow law parameters. 
    """
    """Class that stores the information of the rheological flow law 
    
    E = Activation energy [joule/mol] 
    V = Activation volume [m3/mol]
    m = grain size exponent [nd]
    d = grain size 
    B = Pre-exponential factor [Pa^-n,s^-1]
    R = Gas constant
    q = Peirls creep exponent
    gamma = Peirls creep exponent
    taup = Peirls creep critical stress [Pa]
    ref = ref of of the current rheological law [Title and Doi]
    """
    
    
    def __init__(self
                 ,E:float=0.0
                 ,V:float=0.0
                 ,n:float=0.0
                 ,m:float=0.0
                 ,d0:float=1.0
                 ,B:float=0.0
                 ,B_SI:str = 'None'
                 ,F:str='NoCorrection'
                 ,MPa:int=0
                 ,r:float=0
                 ,aH20:float=1.0
                 ,EH20:float=0.0
                 ,VH20:float = 0.0 
                 ,BH20:float = 1.0
                 ,water_correction:str = 'None'
                 ,ref:str = ''):
        """_summary_

        Args:
            E (float, optional): Activation Energy [J/mol]. Defaults to 0.0.
            V (float, optional): Activation Volume [m^3/mol]. Defaults to 0.0.
            n (float, optional): Stress Exponent. Defaults to 0.0.
            m (float, optional): Grain Size exponent. Defaults to 0.0.
            d0 (float, optional): Initial Grain Size [m]. Defaults to 0.0.
            B (float, optional): Pre-exponential Factor. Defaults to 0.0.
            F (float, optional): Experimental Correction [0 = No Correction, 1 = Simple Shear, 2 = UniAxial]. Defaults to 0.
            MPa (int, optional): Conversion Flag (MPa^n/s->Pa^n/s). Defaults to 0.
            r (float, optional): Water exponent. Defaults to 0.
            water_correction(int): Water correction is an ad hoc parameters that allows to compute the wet/dislocation creep olivine with concentration (constant). 
                                 : rather necessary, as there is a big problem with unit of measure otherwise. 
            ref (str, optional): Reference metadata. Defaults to ''.
        """
        Dictionary_correction = {'NoCorrection':0,
                                 'SimpleShear':1,
                                 'UniAxial':2}
        
        Dictionary_water_correction = {'None':0,
                                       'COH':1,
                                       'Fugacity':2}
        
        self.E = E
        self.V = V
        self.n = n
        self.m = m
        self.d = d0
        self.R = 8.3145
        self.B_or = B # Original value 
        self.B_or_SI = B_SI
        self.F = F 
        self.MPa = MPa
        self.aH20 = aH20
        self.BH20 = BH20
        self.EH20 = EH20
        self.VH20 = VH20 
        self.water_corr = Dictionary_water_correction[water_correction]
        self.r  = r 
        self.B = self._correction(B,Dictionary_correction[F],n,m,MPa,d0,Dictionary_water_correction[water_correction],r)
        self.ref = ref
    #-----------------------------------------------------------------------------------------------------------
    def _correction(self,B,F=0,n=1,m=0,MPa=0,d0=1,water_correction =0, r=0):
        """Correction for the rheological flow law parameters, to account for the typology of the experiment, the unit of measure and the water content and grain size.
        Args:
            B (float): pre-exponential factor [xPa^-n s^-1]-> converted to Pa^-n s^-1
            F (int, optional): typology of the experiment. Defaults to 0. 0: no correction, 1: simple shear, 2: uniaxial.
            n (float, optional): stress exponent. Defaults to 1.
            m (float, optional): grain size exponent. Defaults to 0.
            MPa (int, optional): unit of measure. Defaults to 0. 0: Pa, 1: MPa.
            d0 (float, optional): reference grain size. Defaults to 0.
            r (float, optional): water content exponent. Defaults to 0.
            water (float, optional): water content. Defaults to 0.
        Returns:   
            float: corrected pre-exponential factor
        """
        
        
        # Correct for accounting the typology of the experiment
        if F == 1: # Simple shear
            B = B*2**(n-1)
        if F == 2 : # Uniaxial
            B = B*(3**(n+1)/2)/2
        # Convert the unit of measure
        if MPa == 1:
            B = B*10**(-n*6)
        if water_correction==1: 
            water_con = 1000**r 
        elif water_correction==2: 
            water_con = self.aH20 * self.BH20 * np.exp(- (self.EH20+self.VH20*1e5)/(self.R * 298.15))
            water_con = water_con ** r
        else: 
            water_con = 1.0 
            
        B = B*d0**(-m) * water_con
        return B 

#-----------------------------------------------------------------------------------------------------------

def _check_rheological(tag:str) -> Rheological_flow_law:
    """
    Retrieve rheological flow-law parameters and return a structured data object
    for insertion into the phase database.

    The function interprets the provided ``tag`` and extracts the corresponding
    rheological flow-law parameters from internally defined datasets, packaging
    them into a ``Rheological_flow_law`` data class used by the material property
    system.

    Args:
        tag (str):
            Name of the rheological flow-law model to retrieve.

    Returns:
        Rheological_flow_law:
            Data object containing the parameters of the selected rheological
            flow law, formatted for direct use in the phase database.
    """


    if tag == '':
        # empty rheological flow law to fill it
        return Rheological_flow_law()
    else: 
        A = getattr(RB,Dic_rheo[tag])
        return A 

def _generate_phase(PD:PhaseDataBase,
                    id:int                 = -100,
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
                    name_density:str       = 'PT',
                    radio:float = 0.0,                    
                    radio_flag:float = 0,
                    Pressure_dependency:int=1)     -> PhaseDataBase:
    """ Generate a phase with the specified properties and add it to the phase database.
    Args:
        PD (PhaseDataBase): The phase database to which the new phase will be added.
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

            
    
    PD.id[id-1] = id 
    id = id - 1 
    if name_diffusion != 'Constant':
        A = _check_rheological(name_diffusion)
        PD.Edif[id] = A.E 
        PD.Vdif[id] = A.V
        PD.Bdif[id] = A.B 
    if Edif != -1e23: 
        PD.Edif[id] = Edif 
    elif Vdif !=-1e23:  
        PD.Vdif[id] = Vdif 
    elif Bdif != -1e23:
        PD.Bdif[id] = Bdif  
    if name_dislocation != 'Constant':
        A = _check_rheological(name_dislocation)
        PD.Edis[id] = A.E 
        PD.Vdis[id] = A.V
        PD.Bdis[id] = A.B 
        PD.n[id]    = A.n 
    if n!= -1e23: 
        PD.n[id] = n 
        if PD.Bdis[id] != 0.0: 
            print('Warning: Stress pre-exponential factor has inconsistent measure [Pa^-ns^-1] wrt the original flow law')
    if Edis != -1e23: # if the user specify the activation energy, overwrite the value of the flow law
        PD.Edis[id] = Edis 
    if Vdis !=-1e23: # if the user specify the activation volume, overwrite the value of the flow law  
        PD.Vdis[id] = Vdis 
    if Bdis != -1e23: # if the user specify the pre-exponential factor, overwrite the value of the flow law
        PD.Bdis[id] = Bdis  
    
    if name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta == -1e23:
        raise Warning("Warning: Both diffusion and dislocation creep are set to be constant, but the viscosity is not specified")
        print_ph(f'The software will use the default viscosity value of {PD.eta_def:.1e} Pa s. If you want to specify a different value, please set the eta parameter.')
    
    if name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta == -1e23:
            PD.eta[id] = PD.eta_def
    elif name_diffusion == 'Constant' and name_dislocation == 'Constant' and eta != -1e23:
        PD.eta[id] = eta # in case of constant viscosity, this value will be used. In case of non-constant viscosity, this value will be ignored.
    else: 
        PD.eta[id] = 0.0 # in case of non-constant viscosity, this value will be ignored. I set it to 0.0 to avoid any confusion.

    
 
    if name_diffusion == 'Constant' and name_dislocation == 'Constant': 
        option_rheology = 0
    elif name_dislocation == 'Constant': 
        option_rheology = 1
    elif name_dislocation != 'Constant' and name_diffusion != 'Constant': 
        option_rheology = 2 
    elif name_dislocation != 'Constant' and name_diffusion == 'Constant': 
        option_rheology = 3 
    

        
    PD.option_eta[id] = option_rheology
    
    PD.radio[id] = radio 
    # Heat capacity

    
    if name_capacity not in Dic_Cp:
        print_ph('The avaiable options are: ')
        for k in Dic_Cp: 
            print_ph('%s; '%k)
        
        raise ValueError(f"Error Phase id = {id:d}: {name_capacity} is not a heat Conductivity option")
    
    PD.C0[id],PD.C1[id],PD.C2[id],PD.C3[id],PD.C4[id],PD.C5[id] = release_heat_capacity_parameters(Dic_Cp[name_capacity], Cp)
    
    
    
    if name_conductivity not in Dic_conductivity:
        print_ph('The avaiable options are: ')
        for k in Dic_conductivity: 
            print_ph('%s; '%k)
        
        raise ValueError(f"Error Phase id = {id:d}: {name_conductivity} is not a heat Conductivity option")
    
    TD = _check_diffusivity(name_conductivity)
    if Pressure_dependency == 0: 
        TD.f = 0.0
        
    
    PD.k_a[id] = TD.a 
    PD.k_b[id] = TD.b 
    PD.k_c[id] = TD.c 
    PD.k_d[id] = TD.d 
    PD.k_e[id] = TD.e 
    PD.k_f[id] = TD.f 
    PD.k0[id] = k * TD.g 
    PD.radio_flag[id] = radio_flag 
    
    # Density
    
    if name_alpha not in Dic_alpha:
        print_ph('The avaiable options are: ')
        for k in Dic_conductivity: 
            print_ph('%s; '%k)

        raise ValueError(f"Error Phase id = {id:d}: {name_conductivity} is not a heat Conductivity option")
    alpha = _check_alpha(name_alpha)
    if Pressure_dependency == 0: 
        alpha.alpha2 = 0.0
        Kb = 1e30
    else: 
        Kb = (2*100e9*(1+0.25))/(3*(1-0.25*2))  # Bulk modulus [Pa]
    
    PD.alpha0[id]     = alpha.alpha0
    PD.alpha1[id]     = alpha.alpha1
    PD.alpha2[id]     = alpha.alpha2
    PD.Kb[id]         = Kb  # Bulk modulus [Pa]
    PD.rho0[id]       = rho0
    if name_density == 'Constant':
        PD.option_rho[id] = np.int32(0) 
    else: 
        PD.option_rho[id] = np.int32(2) 
    
    return PD 