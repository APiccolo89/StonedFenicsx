from .package_import import *
from .utils import timing_function, print_ph
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input

data_scal = [('L',float64),
             ('v',float64),
             ('rho',float64),
             ('M',float64),
             ('T',float64),
             ('Temp',float64),
             ('stress',float64),
             ('Watt',float64),
             ('Force',float64),
             ('Cp',float64),
             ('k',float64),
             ('eta',float64),
             ('cm2myr',float64),
             ('strain',float64),
             ('ac',float64),
             ('Energy',float64),
             ('scale_Myr2sec',float64),
             ('scale_vel',float64)]

@jitclass(data_scal)
class Scal: 
    def __init__(self,L=1.0,
                 stress = 1e9,
                 eta = 1e22,
                 Temp  = 1.0):
        self.L = L # Length 
        self.Temp = Temp # Temperature
        self.eta = eta # Viscosity
        self.stress  = stress # Stress 
        self.T = self.eta/self.stress # Time [eta/stress=Pas/Pa=s]
        self.M = (self.stress*self.L**2)*self.T**2/self.L # Scaling mass [Pa L**2 = N = kgm/s2 /s2 /m = kg]
        self.ac = self.L/self.T**2 # acceleration
        self.rho = self.M/self.L**3 # density 
        self.Force = self.M*self.ac # Force 
        self.Energy = self.Force*self.L # Energy
        self.Watt = self.Energy/self.T # Power
        self.strain  = 1/self.T # Stress
        self.k  = self.Watt/(self.L*self.Temp) # Conductivity 
        self.Cp = self.Energy/(self.Temp*self.M) # Heat capacity 
        self.scale_vel = 1e-2 / 365.25 / 24 / 60 / 60 # scaling velocity from cm/yr to m/s
        self.scale_Myr2sec = 1e6 * 365.25 * 60 * 60 * 24 # scale Myr to second 


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
    scal_Bdsl = sc.stress**(-pdb.n)*sc.T**(-1)    # Pa^-ns-1
    scal_Bdif = (sc.stress*sc.T)**(-1)
    pdb.Bdif /= scal_Bdif 
    pdb.Bdis /= scal_Bdsl
    

    
    # Scal the heat capacity 
    scal_c1 = sc.Energy/sc.M/sc.Temp**(0.5) 
    scal_c2 = (sc.Energy*sc.Temp)/sc.M
    scal_c3 = (sc.Energy*sc.Temp**2)/sc.M
    scal_c4 = (sc.Energy)/sc.M/sc.Temp**2
    scal_c5 = (sc.Energy)/sc.M/sc.Temp**3

    pdb.C0 /= sc.Cp
    pdb.C1 /= scal_c1 
    pdb.C2 /= scal_c2 
    pdb.C3 /= scal_c3 
    pdb.C4 /= scal_c4 
    pdb.C5 /= scal_c5
    
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
    
    if MPI.COMM_WORLD.rank == 0: 
    
        print('{ :  -   > Scaling <  -  : }')
        print('         Material properties has been scaled following: ')
        print('         L [length]   = %.3f [m]'%sc.L) 
        print('         Stress       = %.3f [Pa]'%sc.stress)
        print('         eta          = %.3e [Pas]'%sc.eta)
        print('         Temp         = %.2f [K]'%sc.Temp)
        print('The other unit of measure are derived from this set of scaling')
        print('{ <  -   : Scaling :  -  > }')

    return pdb 

def _scale_parameters(lhs,scal):
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
    
def _scaling_control_parameters(ctrl,scal):
    
    ctrl.Ttop /= scal.Temp 
    ctrl.Tmax /= scal.Temp 
    ctrl.v_s = (ctrl.v_s * scal.scale_vel)/(scal.L/scal.T)
    ctrl.g = ctrl.g / (scal.L/scal.T**2)
    ctrl.wz_tk = ctrl.wz_tk / scal.L 
    ctrl.slab_age = ctrl.slab_age * (scal.scale_Myr2sec/scal.T)
    ctrl.time_max = ctrl.time_max * (scal.scale_Myr2sec/scal.T) 
    ctrl.dt = ctrl.dt * (scal.scale_Myr2sec/scal.T)

    return ctrl  
    
    
def _scaling_mesh(M,scal):
    
    M.geometry.x[:] /= scal.L 
    
    return M 

def dimensionless_ginput(g_input:Geom_input,sc:Scal):
    g_input.x /= sc.L # main grid coordinate
    g_input.y /= sc.L   
    g_input.cr /= sc.L # crust 
    g_input.ocr /= sc.L # oceanic crust
    g_input.lit_mt /= sc.L # lithosperic mantle  
    g_input.wz_tk /= sc.L # weak zone 
    g_input.ns_depth /= sc.L # total lithosphere thickness
    g_input.decoupling /= sc.L # decoupling depth -> i.e. where the weak zone is prolonged 
    g_input.resolution_normal /= sc.L  # To Do
    g_input.resolution_refine /= sc.L  # To Do
    g_input.trans /= sc.L # the transition between coupled and uncoupled
    g_input.lab_d /= sc.L # Astenosphere-lithosphere 
    
    return g_input 