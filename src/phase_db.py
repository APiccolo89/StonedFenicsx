

from dataclasses import dataclass, field
import numpy as np
from numba.experimental import jitclass
from numba import float64, int32
from typing import Tuple, List
from typing import Optional
from numba import njit, prange
#-----
Dic_rheo ={'Constant'              :  'Linear', 
          'Hirth_Dry_Olivine_diff' :  'Dislocation_DryOlivine',
          'Hirth_Dry_Olivine_disl' :  'Diffusion_DryOlivine',
          'Van_Keken_diff'         :  'Diffusion_vanKeken',
          'Van_Keken_disl'         :  'Dislocation_vanKeken',
          'Hirth_Wet_Olivine_diff' :  'Diffusion_WetOlivine',
          'Hirth_Wet_Olivine_disl' :  'Dislocation_WetOlivine'
          }
#-----
Dic_conductivity ={'Constant'     :  'Constant',
                   'Mantle'       :  'Fosterite_Fayalite90',
                   'OceanicCrust' :  'Oceanic_Crust'}
#-----
Dic_Cp ={'Constant'                                   : 'Constant',
         'Berman_Forsterite'                          : 'BFo',
         'Berman_Fayalite'                            : 'BFay',
         'Berman_Aranovich_Forsterite'                : 'BAFo',
         'Berman_Aranovich_Fayalite'                  : 'BaFa',
         'Berman_Fo_Fa_01'                            : 'BFoFay01',
         'Bermann_Aranovich_Fo_Fa_0_1'                : 'BAFoFa_0_1',
         'Oceanic_crust'                              : 'OceanicCrust',
         'ContinentalCrust'                           : 'Crust'}
#-----
#-----------------------------------------------------------------------------------------------------------

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



class Thermal_diffusivity():
    """
    # General form of the equation employed for thermal conductivity: 
    Richard: 
    a = 0.565/1e6  
    b = 0.67/1e6
    c = 590.0
    d = 1.4/1e6
    e = 135.0
    T = T-273.15
    
    D = a + b * np.exp(-T/c) + d * np.exp(-T/e)
    -> in m^2/s 
    1) Data from Fosterite 
    
    """
    def __init__(self):
        # Fosterite Fayalite Richard et al 2020 {other reference}
        a = 0.565/1e6  # m^2/s  
        b = 0.67/1e6   # m^2/s
        c = 590.0      # K 
        d = 1.4/1e6    # m^2/s   
        e = 135.0      # K 
        self.Fosterite_Fayalite90 = Lattice_Diffusivity(a,b,c,d,e)
        # Augite 
        a  =  0.59/1e6 
        b  =  1.03/1e6
        c  =  386.0 
        d  =  0.928/1e6
        e  =  125.0   
        
        self.Augite = Lattice_Diffusivity(a,b,c,d,e)
        # AnAb 
        a = 0.36/1e6 
        b = 0.4/1e6 
        c = 300.0 
        d = 0.0 
        e = 1.0 
        
        self.AnAb = Lattice_Diffusivity(a,b,c,d,e)
        # Crust 
        a = 0.432/1e6
        b = 0.44/1e6
        c = 380.0
        d = 0.305/1e6 
        e = 145.0
        
        self.Oceanic_Crust  = Lattice_Diffusivity(a,b,c,d,e)
    


class Lattice_Diffusivity():
    def __init__(self,a=0.0,b=0.0,c=1.0,d=0.0,e=1.0,f=0.0,g=1.0):
        
        self.a = a 
        
        self.b = b 
        
        self.c = c 
        
        self.d = d 
        
        self.e = e
        
        if a != 0: # I refuse to put a different coefficient for each mineral.  
        
            f = 0.05/1e9 
        
        self.f = f
        
        if a != 0: # Switch for heat conductivity 
            
            g = 0.0
        
        self.g = g





        

#-----------------------------------------------------------------------------------------------------------

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
    #-----------------------------------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------------------------

def _check_rheological(tag:str) -> Rheological_flow_law:


    if tag == '':
        # empty rheological flow law to fill it
        return Rheological_flow_law()
    else: 
        RB = Rheological_data_Base()
        A = getattr(RB,Dic_rheo[tag])
        return A 


def _check_diffusivity(tag:str) ->Lattice_Diffusivity:


    if tag == 'Constant':
        # empty rheological flow law to fill it
        return Lattice_Diffusivity()
    else: 
        RB = Thermal_diffusivity()
        A = getattr(RB,Dic_conductivity[tag])
        return A 

#-----------------------------------------------------------------------------------------------------------

spec_phase = [
    # Viscosity – Diffusion creep
    ("Edif", float64[:]),    # Activation energy diffusion creep [J/mol]
    ("Vdif", float64[:]),    # Activation volume diffusion creep [m^3/mol]
    ("Bdif", float64[:]),    # Pre-exponential factor diffusion creep [Pa^-1 s^-1]

    # Viscosity – Dislocation creep
    ("Edis", float64[:]),    # Activation energy dislocation creep [J/mol]
    ("Vdis", float64[:]),    # Activation volume dislocation creep [m^3/mol]
    ("Bdis", float64[:]),    # Pre-exponential factor dislocation creep [Pa^-n s^-1]
    ("n", float64[:]),       # Stress exponent
    ("eta", float64[:]),     # Constant viscosity [Pa s]
    ("option_eta", int32[:]),# Option for viscosity calculation

    # Heat capacity
    ("C0", float64[:]),      # Cp coefficient [J/mol/K]
    ("C1", float64[:]),      # Cp coefficient [J/mol/K^0.5]
    ("C2", float64[:]),      # Cp coefficient [J·K/mol]
    ("C3", float64[:]),      # Cp coefficient [J·K^2/mol]
    ("C4", float64[:]),      # Cp coefficient []
    ("C5", float64[:]),      # Cp coefficient []
    ("option_Cp", int32[:]), # Option for Cp calculation
    
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
    ("Kb", float64[:]),          # Bulk modulus [Pa]
    ("rho0", float64[:]),        # Reference density [kg/m^3]
    ("option_rho", int32[:]),    # Option for density calculation
    
    # radiogenic heat
    ("radio",float64[:]), 
    
    
    
    ("A",float64), # Ref conductivity A 
    ("B",float64), # Ref conductivity B 
    ("T_A",float64), # T [K]
    ("T_B",float64), # T [K]
    ("x_A",float64), # T [K]
    ("x_B",float64), # T [K]
    ("radio_flag",float64[:]), # radio flag 

    
    ("id", int32[:]),              # phase number
    
    # Constants
    ("Tref", float64),      # Reference temperature [K]
    ("Pref", float64),      # Reference pressure [Pa]
    ("R", float64),          # Gas constant [J/mol/K]
    ("T_Scal",float64),     # T_scal [K] -> Important, as within the exponential soul of diffusion/dislocation creep  R is in mol I don't know how to make dimensionless
    ("P_Scal",float64),     # Same reason as before [Pa]
    ("eta_min",float64),    # minimum viscosity [Pas]
    ("eta_max",float64),    # max viscosity [Pas]
    ("eta_def",float64),    # default viscosity [Pas]
    ("friction_angle",float64),
    ('cohesion',float64)
]   

#-----------------------------------------------------------------------------------------------------------
@jitclass(spec_phase)
class PhaseDataBase:
    def __init__(self, number_phases,friction_angle,d=0.5):
        # Initialize individual fields as arrays
        if number_phases>8: 
            raise ValueError("The number of phases should not exceed 7")
        

        
        self.Tref           = 298.15  # Reference temperature [K]
        self.Pref           = 1e5     # Reference pressure [Pa]
        self.R              = 8.3145  # Universal gas constant [J/(mol K)]
        self.eta_min        = 1e18    # Min viscosity [Pas]
        self.eta_max        = 1e26    # Max viscosity [Pas]
        self.eta_def        = 1e21    # Default viscosity [Pas]
        self.T_Scal         = 1.      # Default temperature scale
        self.P_Scal         = 1.      # Default Pressure scale 
        self.friction_angle = friction_angle
        self.id             = np.zeros(number_phases, dtype=np.int32)
        self.cohesion       = 10e6 
        self.A              = 1.8 * (1 - np.exp(-d**1.3 / 0.15)) - (1 - np.exp(-d**0.5 / 5.0))
        self.B              = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-d**3 / 10.0)
        self.T_A            = 490.0 + 1850.0 * np.exp(-d**0.315 / 0.825) + 875.0 * np.exp(-d / 0.18)
        self.T_B            = 2700.0 + 9000.0 * np.exp(-d**0.5 / 0.205)
        self.x_A            = 167.5 + 505.0 * np.exp(-d**0.5 / 0.85)
        self.x_B            = 465.0 + 1700.0 * np.exp(-d**0.94 / 0.175) 
        
        
        
        
        # Explanation: For testing the pressure and t scal are set to be 1.0 -> so, the software is not performing any 
        # scaling operation. 
        # -> When the property are automatically scaled these value will be update automatically. 
        
        
        # Viscosity data 
        # Diffusion creep
        self.Edif         = np.zeros(number_phases, dtype=np.float64)              # Activation energy diffusion creep [J/mol]
        self.Vdif         = np.zeros(number_phases, dtype=np.float64)              # Activation volume diffusion creep [m^3/mol]
        self.Bdif         = np.zeros(number_phases, dtype=np.float64)              # Pre-exponential factor diffusion creep [Pa^-1 s^-1]
        
        # Dislocation creep
        self.Edis       = np.zeros(number_phases, dtype=np.float64)              # Activation energy dislocaiton creep [J/mol]
        self.Vdis       = np.zeros(number_phases, dtype=np.float64)              # Activation volume dislocation creep [m^3/mol]
        self.Bdis       = np.zeros(number_phases, dtype=np.float64)              # Pre-exponential factor [Pa^-n s^-1]
        self.n          = np.ones (number_phases, dtype=np.float64)              # stress exponent  []
        self.eta        = np.zeros(number_phases, dtype=np.float64)              # constant viscosity [Pa s] - in case of constant viscosity 
        self.option_eta = np.zeros(number_phases, dtype=np.int32)                 # Option for viscosity calculation

        # Thermal properties
        self.C0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat capacity [J/mol/K]
        self.C1         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [J/mol/K^0.5]  -> CONVERTED INTO J/kg/K^0.5       
        self.C2         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K)/mol]    -> CONVERTED INTO J/kg/K^2
        self.C3         = np.zeros(number_phases, dtype=np.float64)    # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3
        self.C4         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3
        self.C5         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3

        
        self.option_Cp  = np.zeros(number_phases, dtype=np.int32)                 # Option for heat capacity calculation
        
        # Thermal conductivity 
        self.k0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat conductivity [W/m/K]
        self.k_a        = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity [1/Pa]
        self.k_b        = np.zeros(number_phases, dtype=np.float64)               # exponent
        # Radiative heat transfer
        self.k_c        = np.ones(number_phases, dtype=np.float64)               # Radiative heat transfer constant [W/m/K]
        self.k_d        = np.zeros(number_phases, dtype=np.float64)               # Radiative heat transfer constant [W/m/K^2]
        self.k_e        = np.ones((number_phases), dtype=np.float64)             # Radiative heat transfer polynomial coefficients [W/m/K^3]        
        self.k_f        = np.zeros((number_phases), dtype = np.float64)           # 
        # Radiative heat transfer 
        self.radio_flag   = np.zeros(number_phases, dtype=np.float64)                 # Option for heat conductivity calculation
        
        self.radio      = np.zeros(number_phases, dtype=np.float64)               # Radiogenic

        # Density parameters 
        self.alpha0     = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity coefficient [1/K]   
        self.alpha1     = np.zeros(number_phases, dtype=np.float64)               # Second-order expansivity [1/K^2]
        self.Kb         = np.zeros(number_phases, dtype=np.float64)               # Bulk modulus [Pa]                
        self.rho0       = np.zeros(number_phases, dtype=np.float64)               # Reference density [kg/m^3] {In case of constant density}
        self.option_rho = np.zeros(number_phases, dtype=np.int32)                 # Option for density calculation
        

#-----------------------------------------------------------------------------------------------------------
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
                    rho0:float              = 3300,
                    eta:float              = 1e20,
                    name_capacity:str      = 'Constant',
                    name_conductivity:str  = 'Constant',
                    name_density:str       = 'PT',
                    radio:float = 0.0,
                    radio_flag:float = 0)     -> PhaseDataBase:
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
    elif Edis != -1e23: 
        PD.Edis[id] = Edis 
    elif Vdis !=-1e23:  
        PD.Vdis[id] = Vdis 
    elif Bdis != -1e23:
        PD.Bdis[id] = Bdis  
    PD.eta[id] = eta
 
    if name_diffusion == 'Constant' and name_dislocation == 'Constant': 
        option_rheology = 0
    elif name_dislocation == 'Constant': 
        option_rheology = 1
    else: 
        option_rheology = 2 
        
    PD.option_eta[id] = option_rheology
    
    PD.radio[id] = radio 
    # Heat capacity
    
    PD.C0[id],PD.C1[id],PD.C2[id],PD.C3[id],PD.C4[id],PD.C5[id] = release_heat_capacity_parameters(Dic_Cp[name_capacity], Cp)
    
    
    TD = _check_diffusivity(name_conductivity)
    PD.k_a[id] = TD.a 
    PD.k_b[id] = TD.b 
    PD.k_c[id] = TD.c 
    PD.k_d[id] = TD.d 
    PD.k_e[id] = TD.e 
    PD.k_f[id] = TD.f 
    PD.k0[id] = k * TD.g 
    d = 0.5
    PD.radio_flag[id] = radio_flag 
    
    
    # Density
    PD.alpha0[id]     = 2.832e-5
    PD.alpha1[id]     = 3.79e-8 
    PD.Kb[id]         = (2*100e9*(1+0.25))/(3*(1-0.25*2))  # Bulk modulus [Pa]
    PD.rho0[id]       = rho0
    if name_density == 'Constant':
        PD.option_rho[id] = np.int32(0) 
    else: 
        PD.option_rho[id] = np.int32(2) 
    
    
    return PD 

    
#-----------------------------------------------------------------------------------------------------------

# Deprecated, but better to keep here. 
def release_heat_conductivity_parameters(option_k: int) -> Tuple[float, float]:
    '''
    So, long story short: Hofmeister 1999 conductivity formulation is basically:
    $k(T,P) = k_298(298/T)^a * exp([-(4/gamma+1/3)*\integral(0,T)alpha(\theta)d\theta])$ 
    In the formulation that I found in Tosi, and also in Xu, they use this except with the exp 
    term. 
    Hofmeister 1999, claims that the k_tot(T,P) = k(T,P)+k_rad(T,P) -> So, I will use as 
    constitutive model the $k(T,P) = (k_298 + k_298 * a * P)(298/T)^a$ Assuming that k_298 and a 
    are accouting the exponential term. On the other hand, I need to check in Xu 2004, if it is the case. 
    '''
    
    if option_k == 1 or option_k == 3: 
        k    = 4.10 
        n    = 0.493   
        a    = 0.032/1e9*k #[Convert from 1/GPa to 1/Pa]
    elif option_k == 2 or option_k == 3 :
        k  = 2.47 # [Wm-1K-1]
        a = 0.33/1e9 # [Wm-1K-1GPa-1] GPA PD!!!!
        n  = 0.48 
    
        
    else:
        raise ValueError("The option for heat conductivity is not implemented")

    return k, a, n 
#-----------------------------------------------------------------------------------------------------------
def release_heat_capacity_parameters(tag:str,C0:float)->Tuple[float,float,float,float]: 
    """_summary_
    Cp = C0 + C1 **(-1/2) + C2 ** (-2) + C3 ** (-3) ** C4 (1) ** C5 (2) 
    Args:
        tag (str): _description_
        C0 (float): _description_

    Returns:
        Tuple[float,float,float,float]: _description_
        
        'Constant',

    """
    # Molar weight of fosterite and fayalite
    
    C0 = 0.0; C1 = 0.0; C2 = 0.0; C3 = 0.0; C4 = 0.0 ; C5 = 0.0
    
    if tag == 'Constant': 
        C0 = C0  
        C1 = 0.0 
        C2 = 0.0 
        C3 = 0.0 
        C4 = 0.0 
        C5 = 0.0 
        
    else: 
        if tag == 'BFo' or tag == 'BFay' or tag == 'BFoFay01':
            
            flag = [0,None]           
      
            if tag == 'BFo': 
                flag[1] = 'Fosterite'
            elif tag == 'BFa': 
                flag[1] = 'Fayalite'
            else: 
                flag[1] = 'Mix'
            
            C0, C1, C2, C3, C4, C5 =  mantle_heat_capacity(flag)

        elif tag == 'BAFo' or tag == 'BAFay' or tag == 'BAFoFay01':
            
            flag = [1,None]           
      
            if tag == 'BAFo': 
                flag[1] = 'Fosterite'
            elif tag == 'BAFa': 
                flag[1] = 'Fayalite'
            else: 
                flag[1] = 'Mix'
        
            C0, C1, C2, C3, C4, C5 =  mantle_heat_capacity(flag)
        if tag == 'OceanicCrust': 
            '''
            This is really annoying: Cp calculation is always Cp = C0 + C1T^x ... 
            The author of several paper seems to not understand to use a single consistent 
            equations ... 
            '''
            C0, C1, C2, C3 , C4 , C5 = compute_oceanic_crust()
        elif tag == 'Crust':
            pass
        

    return C0, C1, C2, C3, C4, C5




#----------------------------------------------------------------------------------------------------------

def compute_oceanic_crust() -> Tuple[float,float,float,float,float,float]:
    
    """_summary_
    Following Fred Richard 2020, with an adaptation, such as it makes sense. 
    Returns:
        _type_: _description_
    """
    # Olivine
    # ---- 
    C0_Ol = 1.6108 * 1e3 
    C1_Ol = -1.24788 * 1e4 
    C2_Ol = 0.0 
    C3_Ol = -1.728477*1e9 
    C4_Ol = 0.0 
    C5_Ol = 0.0 
    # ---
    # Clino ( Augite) 
    # --- 
    C0_au = 2.1715   * 1e3 
    C1_au = -2.22716 * 1e4
    C2_au = 1.1333   * 1e6 
    C3_au = 0.0 
    C4_au = -4.555 * 1e-1 
    C5_au = 1.299 * 1e-4
    # ---
    # Plagioclase 
    # --- 
    C0_pg = 1.85757  * 1e3 
    C1_pg = -1.64946 * 1e4
    C2_pg = -5.061   * 1e6 
    C3_pg = 0.0 
    C4_pg = -3.324 * 1e-1 
    C5_pg = 1.505 * 1e-4
    fr_ol = 0.15; fr_au = 0.2; fr_pg = 0.65
    # --- 
    
    C0 = fr_ol * C0_Ol + fr_au * C0_au + fr_pg * C0_pg
    C1 = fr_ol * C1_Ol + fr_au * C1_au + fr_pg * C1_pg
    C2 = fr_ol * C2_Ol + fr_au * C2_au + fr_pg * C2_pg
    C3 = fr_ol * C3_Ol + fr_au * C3_au + fr_pg * C3_pg
    C4 = fr_ol * C4_Ol + fr_au * C4_au + fr_pg * C4_pg
    C5 = fr_ol * C5_Ol + fr_au * C5_au + fr_pg * C5_pg
    
    # --- 
    
    return C0, C1, C2, C3, C4, C5


def mantle_heat_capacity(flag: list) -> Tuple[float,float,float,float,float,float]:
    
    # To do in the future: generalise -> Database of heat capacity parameters as a function of the major mineral molar composition
    # The law of the heat capacity is only temperature dependent, and it is a polynomial. For now I keep the vision of Iris, and introduce a few options later on 
    

    mmfo = 1/(140.691/1000)
    mmfa = 1/(203.771/1000)

    if flag[0] == 0:
        # Berman 1988 
        # forsterite 
        C0_fo = mmfo * 238.64    
        C1_fo = mmfo * -20.013e2 
        C3_fo = mmfo * -11.624e7 
    
        # fayalite 
        C0_fa = mmfa * 248.93    
        C1_fa = mmfa * -19.239e2 
        C3_fa = mmfa * -13.910e7 
    elif flag[0] == 1:
        # Berman & Aranovich 1996 
        # forsterite 
        C0_fo = mmfo * 233.18
        C1_fo = mmfo * -18.016e2
        C3_fo = mmfo * -26.794e7
    
        # fayalite 
        C0_fa = mmfa * 252.
        C1_fa = mmfa * -20.137e2
        C3_fa = mmfa * -6.219e7   
    
    
    if flag[1] == 'Fosterite': 
        # forsterite 
        x = 0.
    if flag[1] == 'Fayalite': 
        # fayalite 
        x = 1.
    if flag[1] == 'Mix': 
        # molar fraction of fayalite is 0.11 
        x = 0.11 
    
    C0 = (1-x)*C0_fo + x*C0_fa
    C1 = (1-x)*C1_fo + x*C1_fa
    C2 = 0.0
    C3 = (1-x)*C3_fo + x*C3_fa
    C4 = 0.0 
    C5 = 0.0 
    
    return C0, C1, C2, C3, C4, C5
      
    

