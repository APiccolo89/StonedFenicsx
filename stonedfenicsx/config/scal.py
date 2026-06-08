from stonedfenicsx.utils import timing_function, print_ph
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input
from stonedfenicsx.numerical_control import time_dependent_evolution
from dataclasses import dataclass,field

@dataclass(slots=True)
class Scal: 
    """Scaling class: 
        Class that stores the scaling value. From the input characteristic length, stress, viscosity and temperature
        it derives the other scaling for the other SI units.  
    """
   
    length:float = 1 # Length
    temp:float = 1000 # Temperature
    eta:float = 1e24 # Viscosity
    stress:float  = 1e9 # Stress

    time: float = field(init=False)         # [s]
    mass: float = field(init=False)         # [kg]
    ac: float = field(init=False)           # [m/s^2]
    rho: float = field(init=False)          # [kg/m^3]
    force: float = field(init=False)        # [N]
    energy: float = field(init=False)       # [J]
    watt: float = field(init=False)         # [W]
    strain_rate: float = field(init=False)  # [1/s]
    k: float = field(init=False)            # [W/(m K)]
    cp: float = field(init=False)           # [J/(kg K)]
    scale_vel: float = 1e-2 / 365.25 / 24 / 60 / 60 # scaling velocity from cm/yr to m/s
    scale_myr2sec: float = 1e6 * 365.25 * 60 * 60 * 24 # scale Myr to second 
    def compute_the_derivative_scal(self):
        """_summary_

        Returns:
            self: _description_
        """
        self.time = self.eta/self.stress # Time [eta/stress=Pas/Pa=s]
        self.mass = (self.stress*self.length**2) * \
            self.time**2 / self.length # Scaling mass [Pa L**2 = N = kgm/s2 /s2 /m = kg]
        self.ac = self.length/self.time**2 # acceleration
        self.rho = self.mass/self.length**3 # density 
        self.force = self.mass * self.ac # Force 
        self.energy = self.force * self.length # Energy
        self.watt = self.energy / self.time # Power
        self.strain_rate = 1 / self.time # Stress
        self.k  = self.watt / (self.length * self.temp) # Conductivity 
        self.cp = self.energy/(self.temp * self.mass) # Heat capacity 
        
        return self

def _scaling_material_properties(pdb,sc:Scal): 
    # scal the references values   
    pdb.Tref /= sc.Temp 
    pdb.Pref /= sc.stress
    pdb.T_Scal = sc.Temp 
    pdb.P_Scal = sc.stress 
    pdb.cohesion /= sc.stress 
    
    # Radiative conductivity parameters 
    
    # The formula used by Richard and Grose yields a conductivity, thus, 
    # thi
    pdb.A /= sc.k
    pdb.B /= sc.k 
    pdb.T_A /= sc.Temp
    pdb.T_B /= sc.Temp 
    pdb.x_A /= sc.Temp 
    pdb.x_B /= sc.Temp
    
    # Viscosity
    pdb.eta /= sc.eta
    pdb.eta_min /= sc.eta
    pdb.eta_max /= sc.eta
    pdb.eta_def /= sc.eta
    
    # B_dif/disl 
    scal_bdsl = sc.stress**(-pdb.n)*sc.T**(-1)    # Pa^-ns-1
    scal_bdif = (sc.stress*sc.T)**(-1)
    pdb.bdif /= scal_bdif 
    pdb.bdis /= scal_bdsl
    pdb.bdis_wz /= scal_bdif

    

    
    # Scal the heat capacity 
    scal_c1 = sc.Energy/sc.M/sc.Temp**(0.5) 
    scal_c2 = (sc.Energy*sc.Temp)/sc.M
    scal_c3 = (sc.Energy*sc.Temp**2)/sc.M
    scal_c4 = (sc.Energy)/sc.M/sc.Temp**2
    scal_c5 = (sc.Energy)/sc.M/sc.Temp**3

    pdb.c0 /= sc.cp
    pdb.c1 /= scal_c1
    pdb.c2 /= scal_c2
    pdb.c3 /= scal_c3
    pdb.c4 /= scal_c4
    pdb.c5 /= scal_c5
    
    # conductivity      D = a + b * np.exp(-T/c) + d * np.exp(-T/e)
   
    pdb.k0 /= sc.k

    pdb.k_a /= sc.L**2/sc.T           
    pdb.k_b /= sc.L**2/sc.T           
    pdb.k_c /= sc.Temp           
    pdb.k_d /= sc.L**2/sc.T           
    pdb.k_e /= sc.Temp           
    pdb.k_f /= sc.k/sc.stress

    pdb.alpha0 /= 1/sc.Temp
    pdb.alpha1 /= 1/sc.Temp**2
    pdb.alpha2 /= 1/sc.stress
    pdb.Kb /= sc.stress
    pdb.rho0 /= sc.rho
    scal_radio = sc.Watt/sc.L**3
    
    pdb.radio /= scal_radio
    
    pdb.bdis_wz /= scal_bdsl
    if MPI.COMM_WORLD.rank == 0: 
        print('{ :  -   > Scaling <  -  : }')
        print('         Material properties has been scaled following: ')
        print(f'         L [length]   = {sc.L:.3f} [m]')
        print(f'         Stress       = {sc.stress:.3f} [Pa]')
        print(f'         eta          = {sc.eta:.3e} [Pas]')
        print(f'         Temp         = {sc.Temp:.2f} [K]')
        print('The other unit of measure are derived from this set of scaling')
        print('{ <  -   : Scaling :  -  > }')

    return pdb 

def scale_parameters(lhs,scal):
    scal_factor = (scal.scale_Myr2sec / scal.T)
    lhs.end_time = lhs.end_time * scal_factor
    lhs.dt = lhs.dt * scal_factor
    lhs.c_age_plate = lhs.c_age_plate * scal_factor
    lhs.c_age_var = lhs.c_age_var * scal_factor  
    lhs.dz = lhs.dz / scal.L 
    lhs.alpha_g = lhs.alpha_g / (1 / scal.Temp)
    lhs.k = lhs.k / scal.k
    lhs.Cp = lhs.Cp / scal.Cp
    lhs.rho = lhs.rho / scal.rho
    return lhs
    
def scaling_control_parameters(ctrl,scal):
    
    ctrl.temp_top /= scal.temp 
    ctrl.temp_max /= scal.temp 
    ctrl.v_s = (ctrl.v_s * scal.scale_vel)/(scal.L/scal.T)
    ctrl.g = ctrl.g / (scal.L/scal.T**2)
    ctrl.wz_tk = ctrl.wz_tk / scal.L 
    ctrl.slab_age = ctrl.slab_age * (scal.scale_Myr2sec/scal.T)
    ctrl.time_max = ctrl.time_max * (scal.scale_Myr2sec/scal.T) 
    ctrl.dt = ctrl.dt * (scal.scale_Myr2sec/scal.T)

    return ctrl
    
    
def _scaling_mesh(M,scal):

    M.geometry.x[:] /= scal.length 
   
    return M

def dimensionless_ginput(g_input:Geom_input,sc:Scal):
    """_summary_

    Args:
        g_input (Geom_input): geometrical input
        sc (Scal): scaling object

    Returns:
        g_input: scaled geometrical input
    """
    g_input.x /= sc.length # main grid coordinate
    g_input.y /= sc.length  
    g_input.cr /= sc.length # crust 
    g_input.ocr /= sc.length # oceanic crust
    g_input.lit_mt /= sc.length # lithosperic mantle  
    g_input.wz_tk /= sc.length # weak zone 
    g_input.ns_depth /= sc.length # total lithosphere thickness
    g_input.decoupling /= sc.length # decoupling depth -> i.e. where the weak zone is prolonged 
    g_input.resolution_normal /= sc.lenght  # To Do
    g_input.resolution_refine /= sc.length  # To Do
    g_input.trans /= sc.length # the transition between coupled and uncoupled
    g_input.lab_d /= sc.length # Astenosphere-lithosphere 

    return g_input

def scal_time_class(t_lhs:time_dependent_evolution, sc: Scal)->time_dependent_evolution:

    t_lhs.age_plate *= sc.scale_Myr2sec * 1/sc.T 
    t_lhs.vel_plate *= sc.scale_vel * (sc.T/sc.L)
    t_lhs.t_age *= sc.scale_Myr2sec * 1/sc.T 
    t_lhs.t_vel *= sc.scale_Myr2sec * 1/sc.T 
    return t_lhs