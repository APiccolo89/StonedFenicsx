

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
    ("Cp", float64[:]),      # Constant heat capacity [J/kg/K]
    ("C0", float64[:]),      # Cp coefficient [J/mol/K]
    ("C1", float64[:]),      # Cp coefficient [J/mol/K^0.5]
    ("C2", float64[:]),      # Cp coefficient [J·K/mol]
    ("C3", float64[:]),      # Cp coefficient [J·K^2/mol]
    ("option_Cp", int32[:]), # Option for Cp calculation

    # Thermal conductivity
    ("k", float64[:]),       # Constant thermal conductivity [W/m/K]
    ("k0", float64[:]),      # Reference thermal conductivity [W/m/K]
    ("a", float64[:]),       # Pressure-dependent coefficient [1/Pa]
    ("k_b", float64[:]),     # Radiative heat transfer coefficient [W/m/K]
    ("k_c", float64[:]),     # Radiative heat transfer coefficient [W/m/K^2]
    ("k_d", float64[:, :]),  # Radiative polynomial coefficients [W/m/K^3]
    ("option_k", int32[:]),  # Option for conductivity calculation

    # Density parameters
    ("alpha0", float64[:]),       # Thermal expansivity [1/K]
    ("alpha1", float64[:]),      # Second-order thermal expansivity [1/K^2]
    ("Kb", float64[:]),          # Bulk modulus [Pa]
    ("rho0", float64[:]),        # Reference density [kg/m^3]
    ("option_rho", int32[:]),    # Option for density calculation

    # Constants
    ("Tref", float64),      # Reference temperature [K]
    ("Pref", float64),      # Reference pressure [Pa]
    ("R", float64),          # Gas constant [J/mol/K]
    ("T_Scal",float64),     # T_scal -> Important, as within the exponential soul of diffusion/dislocation creep  R is in mol I don't know how to make dimensionless
    ("P_scal",float64),     # Same reason as before
]   

#-----------------------------------------------------------------------------------------------------------

@jitclass(spec_phase)
class PhaseDataBase:
    def __init__(self, number_phases):
        # Initialize individual fields as arrays
        if number_phases>8: 
            raise ValueError("The number of phases should not exceed 7")
        
        self.Tref        = 298.15  # Reference temperature [K]
        self.Pref        = 1e5     # Reference pressure [Pa]
        self.R           = 8.3145  # Universal gas constant [J/(mol K)]
        
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
        self.Cp         = np.zeros(number_phases, dtype=np.float64)               # Heat capacity [J/kg K] {In case of constant heat capacity}
        self.C0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat capacity [J/mol/K]
        self.C1         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [J/mol/K^0.5]  -> CONVERTED INTO J/kg/K^0.5       
        self.C2         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K)/mol]    -> CONVERTED INTO J/kg/K^2
        self.C3         = np.zeros(number_phases, dtype=np.float64)               # Temperature dependent heat capacity [(J*K^2)/mol]  -> CONVERTED INTO J/kg/K^3
        self.option_Cp  = np.zeros(number_phases, dtype=np.int32)                 # Option for heat capacity calculation
        
        # Thermal conductivity 
        self.k          = np.zeros(number_phases, dtype=np.float64)               # Heat conductivity [W/m/K] {In case of constant heat conductivity}
        self.k0         = np.zeros(number_phases, dtype=np.float64)               # Reference heat conductivity [W/m/K]
        self.a          = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity [1/Pa]
        # Radiative heat transfer
        self.k_b        = np.zeros(number_phases, dtype=np.float64)               # Radiative heat transfer constant [W/m/K]
        self.k_c        = np.zeros(number_phases, dtype=np.float64)               # Radiative heat transfer constant [W/m/K^2]
        self.k_d        = np.zeros((number_phases, 4), dtype=np.float64)             # Radiative heat transfer polynomial coefficients [W/m/K^3]        
    
        # Radiative heat transfer 
        self.option_k   = np.zeros(number_phases, dtype=np.int32)                 # Option for heat conductivity calculation

        
        # Density parameters 
        self.alpha0      = np.zeros(number_phases, dtype=np.float64)               # Thermal expansivity coefficient [1/K]   
        self.alpha1     = np.zeros(number_phases, dtype=np.float64)               # Second-order expansivity [1/K^2]
        self.Kb         = np.zeros(number_phases, dtype=np.float64)               # Bulk modulus [Pa]                
        self.rho0       = np.zeros(number_phases, dtype=np.float64)               # Reference density [kg/m^3] {In case of constant density}
        self.option_rho = np.zeros(number_phases, dtype=np.int32)                 # Option for density calculation

#-----------------------------------------------------------------------------------------------------------
def _generate_phase(PD:PhaseDataBase,
                    id:int                 = -100,
                    name_diffusion:str     = '',
                    Edif:float             = -1e23, 
                    Vdif:float             = -1e23,
                    Bdif:float             = -1e23, 
                    name_dislocation:float = '',
                    n:float                = -1e23,
                    Edis:float             = -1e23,
                    Vdis:float             = -1e23, 
                    Bdis:float             = -1e23, 
                    Cp:float               = 1171.52,
                    k:float                = 3.138,
                    rho:float              = 3300,
                    eta:float              = -1e23,
                    option_rheology:float  = 0,
                    option_Cp:int       = 0,
                    option_k:int         = 0,
                    option_rho:int = 0  )     -> PhaseDataBase:
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
    PD.option_eta[id] = option_rheology
    
    
    # Heat capacity
    if option_Cp == 0:
        # constant heat capacity 
        PD.Cp[id] = Cp
    elif option_Cp > 0 and option_Cp < 7:
        # Compute the material the effective material property 
        PD.C0[id],PD.C1[id],PD.C2[id],PD.C3[id] = release_heat_capacity_parameters(option_Cp)
    
    PD.option_Cp[id] = option_Cp
    
    # Heat Conductivity
    if option_k == 0:
        # constant heat conductivity 
        PD.k[id] = k
    elif option_k != 0:
        # Compute the material the effective material property 
        PD.k0[id],PD.a[id]  = release_heat_conductivity_parameters(option_k)
    
    elif option_k == 3:

        PD.k_b[id]    = 5.3
        PD.k_c[id]    = 0.0015
        PD.k_d[id,:]  = np.array([1.753e-2, -1.0365e-4, 2.2451e-7, -3.4071e-11], dtype=np.float64) 
    
    PD.option_k[id]   = option_k    
    
    # Density
    PD.alpha0[id]      = 2.832e-5
    PD.alpha1[id]     = 3.79e-8 
    PD.Kb[id]         = (2*100e9*(1+0.25))/(3*(1-0.25*2))  # Bulk modulus [Pa]
    PD.rho0[id]       = rho
    
    PD.option_rho[id] = option_rho
    
    return PD 

    
#-----------------------------------------------------------------------------------------------------------


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
    
    if option_k == 1: 
        k    = 4.10 
        n    = 0.493   
        a    = 0.032/1e9*k #[Convert from 1/GPa to 1/Pa]
    elif option_k == 2:
        k  = 2.47 # [Wm-1K-1]
        a = 0.33/1e9 # [Wm-1K-1GPa-1] GPA PD!!!!
        n  = 0.48 
    
        
    else:
        raise ValueError("The option for heat conductivity is not implemented")

    return k, a
#-----------------------------------------------------------------------------------------------------------

def release_heat_capacity_parameters(option_C_p: int) -> Tuple[float, float, float, float]:
    
    # To do in the future: generalise -> Database of heat capacity parameters as a function of the major mineral molar composition
    # The law of the heat capacity is only temperature dependent, and it is a polynomial. For now I keep the vision of Iris, and introduce a few options later on 
    

    mmfo = 1/(140.691/1000)
    mmfa = 1/(203.771/1000)

    if option_C_p > 0 and option_C_p < 4:
        # Berman 1988 
        # forsterite 
        C0_fo = mmfo * 238.64    
        C1_fo = mmfo * -20.013e2 
        C3_fo = mmfo * -11.624e7 
    
        # fayalite 
        C0_fa = mmfa * 248.93    
        C1_fa = mmfa * -19.239e2 
        C3_fa = mmfa * -13.910e7 
    elif option_C_p > 3 and option_C_p < 7:
        # Berman & Aranovich 1996 
        # forsterite 
        C0_fo = mmfo * 233.18
        C1_fo = mmfo * -18.016e2
        C3_fo = mmfo * -26.794e7
    
        # fayalite 
        C0_fa = mmfa * 252.
        C1_fa = mmfa * -20.137e2
        C3_fa = mmfa * -6.219e7   
    
    
    if option_C_p == 1 or option_C_p == 4: 
        # forsterite 
        x = 0.
    if option_C_p == 2 or option_C_p == 5: 
        # fayalite 
        x = 1.
    if option_C_p == 3 or option_C_p == 6: 
        # molar fraction of fayalite is 0.11 
        x = 0.11 
    
    C0 = (1-x)*C0_fo + x*C0_fa
    C1 = (1-x)*C1_fo + x*C1_fa
    C2 = 0.0
    C3 = (1-x)*C3_fo + x*C3_fa
    
    return C0, C1, C2, C3
      
    

