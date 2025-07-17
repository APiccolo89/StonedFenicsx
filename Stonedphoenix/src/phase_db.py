

from dataclasses import dataclass, field
import numpy as np
from numba.experimental import jitclass
from numba import float64, int32
from typing import Tuple, List
from typing import Optional
from numba import njit, prange
 
Dic_rheo ={'Hirth_Dry_Olivine_diff':  'Dislocation_DryOlivine',
          'Hirth_Dry_Olivine_disl' :  'Diffusion_DryOlivine',
          'Van_Keken_diff'         :  'Diffusion_vanKeken',
          'Van_Keken_disl'         :  'Dislocation_vanKeken',
          'Hirth_Wet_Olivine_diff' :  'Diffusion_WetOlivine',
          'Hirth_Wet_Olivine_disl' :  'Dislocation_WetOlivine'}


class Rheological_data_Base():
    """
    Global data base of the rheology employed for the project 
    """
    def __init__(self):
        # Dislocation creep
        # Dry Olivine
        E = 540e3
        V = 0.0
        n = 3.5
        m = 0.0
        B = (2*28968.6)**(-n)
        r = 1.0
        d0 = 1
        water = 1.0
        q   = -1e23
        taup = -1e23
        gamma = -1e23
        self.Dislocation_vanKeken = Rheological_flow_law(E,V,n,m,d0,B,0,0,r,water,q,gamma,taup)
        #Dry Olivine Hirth 
        E = 530.0e3
        V = 15e-6
        n = 3.5 
        m = 0.0
        B = 1.1e5
        r = 1.0
        d0 = 1
        water = 1.0
        q   = -1e23
        taup = -1e23
        gamma = -1e23 
        self.Dislocation_DryOlivine = Rheological_flow_law(E,V,n,m,d0,B,1,1,r,water,q,gamma,taup)
        # Wet Olivine
        E = 520.0e3
        V = 22e-6
        n = 3.5 
        m = 0.0
        B = 1600
        r = 1.2
        d0 = 1
        water = 1000.0
        self.Dislocation_WetOlivine = Rheological_flow_law(E,V,n,m,d0,B,1,1,r,water,q,gamma,taup)
        # Wet Plagio
        E = 345.0e3
        V = 38e-6
        n = 3.0 
        m = 0.0
        B = 1.5849
        r = 1.0
        d0 = 1
        water = 158.4893
        self.Dislocation_WetPlagio  = Rheological_flow_law(E,V,n,m,d0,B,2,1,r,water,q,gamma,taup)
        # Diffusion creep 
        E = 375.0e3
        V = 5e-6
        n = 1.0 
        m = 3.0
        B = 1.5e9
        r = 1.0
        d0 = 10e3
        water = 1.0
        self.Diffusion_DryOlivine   = Rheological_flow_law(E,V,n,m,d0,B,1,1,r,water,q,gamma,taup)
        E = 375.0e3
        V = 10e-6
        n = 1.0 
        m = 3.0
        B = 2.5e7
        r = 0.8
        d0 = 10e3
        water = 1000
        self.Diffusion_WetOlivine   = Rheological_flow_law(E,V,n,m,d0,B,1,1,r,water,q,gamma,taup)
        E = 159.0e3
        V = 38e-6
        n = 1.0 
        m = 3.0
        B = 0.1995
        r = 1.0
        d0 = 100
        water = 158.4893
        self.Diffusion_WetPlagio    = Rheological_flow_law(E,V,n,m,d0,B,2,1,r,water,q,gamma,taup)
        E = 335e3
        V = 0. 
        n = 1.0
        m = 3.0
        B = (1/1.32043e9/2)
        r = 1.0
        d0 = 1
        water = 1.0
        self.Diffusion_vanKeken = Rheological_flow_law(E,V,n,m,d0,B,0,0,r,water,q,gamma,taup)



class Rheological_flow_law():
    """
    Class that contains the rheological flow law parameters. 
    """
    def __init__(self,E=0.0,V=0.0,n=0.0,m=0.0,d0=0.0,B=0.0,F=0,MPa=0,r=0,water=1.0,q=0,gamma=0,taup=0):
        self.E = E
        self.V = V
        self.n = n
        self.m = m
        self.d = d0
        self.B = self._correction(B,F,n,m,MPa,d0,r,water)
        self.R = 8.3145
        self.q  = q
        self.gamma = gamma
        self.taup = taup
    def _correction(self,B,F=0,n=1,m=0,MPa=0,d0=0,r=0,water=0):
        # Correct for accounting the typology of the experiment
        if F == 1: # Simple shear
            B = B*2**(n-1)
        if F == 2 : # Uniaxial
            B = B*(3**(n+1)/2)/2
        # Convert the unit of measure
        if MPa == 1:
            B = B*10**(-n*6); 
        # Implicitly account the water content and the grain size
        B = B*d0**(-m)*water**(r)
        return B 


def _check_rheological(tag:str) -> Rheological_flow_law:


    if tag == '':
        # empty rheological flow law to fill it
        return Rheological_flow_law()
    else: 
        RB = Rheological_data_Base()
        A = getattr(RB,Dic_rheo[tag])
        return A 



spec_phase = [
    # Diffusion creep
    ("Edif", float64[:]),    # Activation energy diffusion creep [J/mol]
    ("Vdif", float64[:]),    # Activation volume diffusion creep [m^3/mol]
    ("Bdif", float64[:]),    # Pre-exponential factor diffusion creep [Pa^-1 s^-1]

    # Dislocation creep
    ("Edis", float64[:]),    # Activation energy dislocation creep [J/mol]
    ("Vdis", float64[:]),    # Activation volume dislocation creep [m^3/mol]
    ("Bdis", float64[:]),    # Pre-exponential factor [Pa^-n s^-1]
    ("n",    float64[:]),    # Stress exponent
    ("eta",  float64[:]),    # Constant viscosity [Pa s]
    ("option_eta", int32[:]),# Option for viscosity calculation

    # Heat capacity
    ("Cp", float64[:]),      # Constant heat capacity [J/kg K]
    ("C0", float64[:]),      # Reference Cp [J/mol/K]
    ("C1", float64[:]),      # Cp term [J/mol/K]
    ("C2", float64[:]),      # Cp term [J/mol/K]
    ("C3", float64[:]),      # Cp term [J/mol/K]
    ("option_Cp", int32[:]), # Option for Cp calculation

    # Thermal conductivity
    ("k", float64[:]),       # Constant thermal conductivity [W/m/K]
    ("k0", float64[:]),      # Reference conductivity [W/m/K]
    ("a", float64[:]),       # Thermal expansivity [1/Pa]
    ("T", float64[:]),       # Reference temperature [K]
    ("option_k", int32[:]),  # Option for conductivity calculation

    # Density parameters
    ("alpha", float64[:]),       # Thermal expansivity coefficient
    ("alpha2", float64[:]),      # Second-order expansivity
    ("Kb", float64[:]),          # Bulk modulus
    ("rho0", float64[:]),        # Reference density
    ("option_rho", int32[:]),    # Option for density calculation
]

@jitclass(spec_phase)
class PhaseDataBase:
    def __init__(self, number_phases):
        # Initialize individual fields as arrays
        if number_phases>8: 
            raise ValueError("The number of phases should not exceed 7")
        
        # Viscosity data 
        # Diffusion creep
        self.Edif       = np.zeros(number_phases, dtype=np.float64)              # Activation energy diffusion creep [J/mol]
        self.Vdif       = np.zeros(number_phases, dtype=np.float64)              # Activation volume diffusion creep [m^3/mol]
        self.Bdif       = np.zeros(number_phases, dtype=np.float64)              # Pre-exponential factor diffusion creep [Pa^-1 s^-1]
        
        # Dislocation creep
        self.Edis       = np.zeros(number_phases, dtype=np.float64)              # Activation energy dislocaiton creep [J/mol]
        self.Vdis       = np.zeros(number_phases, dtype=np.float64)              # Activation volume dislocation creep [m^3/mol]
        self.Bdis       = np.zeros(number_phases, dtype=np.float64)              # Pre-exponential factor [Pa^-n s^-1]
        self.n          = np.ones (number_phases, dtype=np.float64)              # stress exponent  []
        self.eta        = np.zeros(number_phases, dtype=np.float64)              # constant viscosity [Pa s] - in case of constant viscosity 
        self.option_eta = np.zeros(number_phases, dtype=np.int32)                 # Option for viscosity calculation

        # Thermal properties
        self.Cp         = np.zeros(number_phases, dtype=np.float64)               # Heat capacity [J/kg K] {In case of constant heat capacity}
        self.C0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat capacity [J/mol/K]
        self.C1         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [J/mol/K^0.5]         
        self.C2         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K)/mol]   
        self.C3         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]
        self.option_Cp  = np.zeros(number_phases, dtype=np.int32)                 # Option for heat capacity calculation
        
        # Thermal conductivity 
        self.k          = np.zeros(number_phases, dtype=np.float64)               # Heat conductivity [W/m/K] {In case of constant heat conductivity}
        self.k0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat conductivity [W/m/K]
        self.a          = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity [1/Pa]
        self.T          = np.ones (number_phases, dtype=np.float64)*298.15        # Reference temperature [K] -> By DEFAULT 298.15 K {25+273.15}K
        self.option_k   = np.zeros(number_phases, dtype=np.int32)                 # Option for heat conductivity calculation
        
        # Density parameters 
        self.alpha      = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity coefficient [1/K]   
        self.alpha2     = np.zeros(number_phases, dtype=np.float64)               # Second-order expansivity [1/K^2]
        self.Kb         = np.zeros(number_phases, dtype=np.float64)               # Bulk modulus [Pa]                
        self.rho0       = np.zeros(number_phases, dtype=np.float64)               # Reference density [kg/m^3] {In case of constant density}
        self.option_rho = np.zeros(number_phases, dtype=np.int32)                 # Option for density calculation





def _generate_phase(PD:PhaseDataBase,
                    number_phases:int,
                    id             = -100,
                    name_diffusion = '',
                    Edif = -1e23, 
                    Vdif = -1e23,
                    Bdif = -1e23, 
                    name_dislocation = '',
                    n    = -1e23,
                    Edis = -1e23,
                    Vdis = -1e23, 
                    Bdis = -1e23, 
                    Cp   = 1171.52,
                    k    = 3.138,
                    alpha = 3e-5,
                    beta = 1e-12,
                    rho   = 3300,
                    eta = -1e23):
    """
    Generate phase: 
    id : phase id number [0->n] 
    name_diffusion : name of the diffusion/dislocation creep law 
    Edif: data of the diffusion creep [Energy of activation]  
    Vdif: data of the diffusion creep [Activation volume]
    Bdif: preexponential factor diffusion
    name_dislocation: name of the dislocation creep law
    Edis: data of dislocation [Energy of activation]
    Vdis: data of dislocation [Activation volume]
    Bdis: preexponential factor dislocation 
    n   : stress exponent 
    Cp  : heat capacity 
    k   : heat conductivity 
    alpha:  thermal expansivity
    beta : compressibility 
    rho : density 
    => output -> update the id_th phase_db 
    """
    if name_diffusion != 'constant':
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
    if name_dislocation != 'constant':
        A = _check_rheological(name_dislocation)
        PD.Edis[id] = A.E 
        PD.Vdis[id] = A.V
        PD.Bdis[id] = A.B 
        PD.n[id] = A.n 
    if n!= -1e23: 
        PD.n[id] = n 
        if PD.Bdis[id] != 0.0: 
            print('Warning: Stress pre-exponential factor has inconsistent measure [Pa^-ns^-1] wrt the original flow law')
    elif Edis != -1e23: 
        PD.Edis[id] = Edis 
    elif Vdis !=-1e23:  
        PD.Vdis[id] = Vdis 
    elif Bdis != -1e23:
        PD.Bdis[id] = Bdis  
    
    PD.Cp[id] = Cp
    PD.alpha[id] = alpha
    PD.k[id] = k 
    PD.beta[id] = beta 
    PD.rho[id] = rho 
    PD.eta[id] = eta 
    return PD 


#@dataclass(slots=True)
#class phase_data_base:
#    """
#    Class that contains the data for the phase database. 
#    """
#    number_phases: int
#    phase_db: np.ndarray = field(init=False)
#
#    def __post_init__(self):
#        """
#        Initialize the phase database after the dataclass is created.
#        """
#        dtype = [('id', np.int32),
#                 ('Edif', np.float64),
#                 ('Vdif', np.float64),
#                 ('Bdif', np.float64),
#                 ('n', np.float64),
#                 ('Edis', np.float64),
#                 ('Vdis', np.float64),
#                 ('Bdis', np.float64),
#                 ('Cp', np.float64),
#                 ('k', np.float64),
#                 ('alpha', np.float64),
#                 ('beta',np.float64),
#                 ('rho', np.float64),
#                 ('eta', np.float64),]
#        self.phase_db = np.zeros(self.number_phases, dtype=dtype)
#    
#    def _generate_phase(self,
#                        id             = -100,
#                        name_diffusion = '',
#                        Edif = -1e23, 
#                        Vdif = -1e23,
#                        Bdif = -1e23, 
#                        name_dislocation = '',
#                        n    = -1e23,
#                        Edis = -1e23,
#                        Vdis = -1e23, 
#                        Bdis = -1e23, 
#                        Cp   = 1171.52,
#                        k    = 3.138,
#                        alpha = 3e-5,
#                        beta = 1e-12,
#                        rho   = 3300,
#                        eta = -1e23):
#        """
#        Generate phase: 
#        id : phase id number [0->n] 
#        name_diffusion : name of the diffusion/dislocation creep law 
#        Edif: data of the diffusion creep [Energy of activation]  
#        Vdif: data of the diffusion creep [Activation volume]
#        Bdif: preexponential factor diffusion
#        name_dislocation: name of the dislocation creep law
#        Edis: data of dislocation [Energy of activation]
#        Vdis: data of dislocation [Activation volume]
#        Bdis: preexponential factor dislocation 
#        n   : stress exponent 
#        Cp  : heat capacity 
#        k   : heat conductivity 
#        alpha:  thermal expansivity
#        beta : compressibility 
#        rho : density 
#        => output -> update the id_th phase_db 
#        """
#        if name_diffusion != 'constant':
#            A = _check_rheological(name_diffusion)
#            self.phase_db['Edif'][self.phase_db['id']==id] = A.E 
#            self.phase_db['Vdif'][self.phase_db['id']==id] = A.V
#            self.phase_db['Bdif'][self.phase_db['id']==id] = A.B 
#
#        if Edif != -1e23: 
#            self.phase_db['Edif'][self.phase_db['id']==id] = Edif 
#        elif Vdif !=-1e23:  
#            self.phase_db['Vdif'][self.phase_db['id']==id] = Vdif 
#        elif Bdif != -1e23:
#            self.phase_db['Bdif'][self.phase_db['id']==id] = Bdif  
#
#        if name_dislocation != 'constant':
#            A = _check_rheological(name_dislocation)
#            self.phase_db['Edis'][self.phase_db['id']==id] = A.E 
#            self.phase_db['Vdis'][self.phase_db['id']==id] = A.V
#            self.phase_db['Bdis'][self.phase_db['id']==id] = A.B 
#            self.phase_db['n'][self.phase_db['id']==id] = A.n 
#
#        if n!= -1e23: 
#            self.phase_db['n'][self.phase_db['id']==id] = n 
#            if self.phase_db['Bdis'][self.phase_db['id']==id] != 0.0: 
#                print('Warning: Stress pre-exponential factor has inconsistent measure [Pa^-ns^-1] wrt the original flow law')
#        elif Edis != -1e23: 
#            self.phase_db['Edis'][self.phase_db['id']==id] = Edis 
#        elif Vdis !=-1e23:  
#            self.phase_db['Vdis'][self.phase_db['id']==id] = Vdis 
#        elif Bdis != -1e23:
#            self.phase_db['Bdis'][self.phase_db['id']==id] = Bdis  
#        
#        self.phase_db['Cp'][self.phase_db['id']==id] = Cp
#        self.phase_db['alpha'][self.phase_db['id']==id] = alpha
#        self.phase_db['k'][self.phase_db['id']==id] = k 
#        self.phase_db['beta'][self.phase_db['id']==id] = beta 
#        self.phase_db['rho'][self.phase_db['id']==id] = rho 
#        self.phase_db['eta'][self.phase_db['id']==id] = eta 
#
#
#        return self 
#
#

