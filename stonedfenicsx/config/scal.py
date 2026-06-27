"""Modules"""
from stonedfenicsx.utils import timing_function
from stonedfenicsx.config.numerical_control import (CtrlKy,
                                                    CtrlTemperatureBC,
                                                    NumericalControls,
                                                    SimulationControls)
from stonedfenicsx.config.phase_db import PhaseDataBase
from dataclasses import dataclass,field
from stonedfenicsx.config.geometry import GeomInput,Mesh
_KELVIN_ = 273.15
_KM_2_M_ = 1000

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
        """During the configuration it fills the remnants unit of measure

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

@timing_function
def scaling_simulation_physical(
    ctrl_sim: SimulationControls, pdb: PhaseDataBase, mesh: Mesh
,sc:Scal) -> tuple[SimulationControls, PhaseDataBase, Mesh]:
    
    ctrl_sim.ctrl_ky = scale_kinematic_bc(ctrl_ky=ctrl_sim.ctrl_ky,sc=sc)
    ctrl_sim.ctrl_tbc = scale_parameters(ctrl_tbc=ctrl_sim.ctrl_tbc,sc=sc)
    ctrl_sim.ctrl = scaling_control_parameters(ctrl=ctrl_sim.ctrl,sc=sc)
    mesh = scaling_mesh(mesh=mesh,sc=sc)
    pdb = scaling_material_properties(pdb=pdb,sc=sc)
    return ctrl_sim, pdb, mesh


def scaling_material_properties(pdb:PhaseDataBase,sc:Scal)->PhaseDataBase: 
    # scal the references values   
    pdb.temp_ref /= sc.temp
    pdb.pres_ref /= sc.stress
    pdb.temp_scal = sc.temp
    pdb.pres_scal = sc.stress
    pdb.cohesion /= sc.stress
    
    pdb.a_rad /= sc.k
    pdb.b_rad /= sc.k
    pdb.temp_a /= sc.temp
    pdb.temp_b /= sc.temp
    pdb.x_a /= sc.temp
    pdb.x_b /= sc.temp
    
    # Viscosity
    pdb.eta /= sc.eta
    pdb.eta_min /= sc.eta
    pdb.eta_max /= sc.eta
    pdb.eta_def /= sc.eta
    
    # B_dif/disl
    scal_bdsl = sc.stress**(-pdb.n)*sc.time**(-1)    # Pa^-ns-1
    scal_bdif = (sc.stress*sc.time)**(-1)
    pdb.b_dif /= scal_bdif
    pdb.b_dis /= scal_bdsl

    # Scal the heat capacity
    scal_c1 = sc.energy/sc.mass/sc.temp**(0.5)
    scal_c2 = (sc.energy*sc.temp)/sc.mass
    scal_c3 = (sc.energy*sc.temp**2)/sc.mass
    scal_c4 = (sc.energy)/sc.mass/sc.temp**2
    scal_c5 = (sc.energy)/sc.mass/sc.temp**3

    pdb.c0 /= sc.cp
    pdb.c1 /= scal_c1
    pdb.c2 /= scal_c2
    pdb.c3 /= scal_c3
    pdb.c4 /= scal_c4
    pdb.c5 /= scal_c5
    # conductivity      D = a + b * np.exp(-T/c) + d * np.exp(-T/e)
    
    pdb.k0 /= sc.k

    pdb.k_a /= sc.length**2/sc.time      
    pdb.k_b /= sc.length**2/sc.time
    pdb.k_c /= sc.temp
    pdb.k_d /= sc.length**2/sc.time
    pdb.k_e /= sc.temp
    pdb.k_f /= sc.k/sc.stress

    pdb.alpha0 /= 1/sc.temp
    pdb.alpha1 /= 1/sc.temp**2
    pdb.alpha2 /= 1/sc.stress
    pdb.kb /= sc.stress
    pdb.rho0 /= sc.rho
    scal_radio = sc.watt/sc.length**3
    
    pdb.radiogenic_heat /= scal_radio
    scal_bdsl = sc.stress**(-pdb.n_wz)*sc.time**(-1)    # Pa^-ns-1

    pdb.bdis_wz /= scal_bdsl

    return pdb

def scale_parameters(ctrl_tbc:CtrlTemperatureBC,sc:Scal)->CtrlTemperatureBC:
    """_summary_

    Args:
        lhs (_type_): lhs:dataclass that handles the thermal boundary condition
        scal (Scal): scal the scaling class

    Returns:
        lhs(): the scaled lhs dataclass 
    """
    
    scal_factor = (sc.scale_myr2sec / sc.time)
    ctrl_tbc.temp_top = (ctrl_tbc.temp_top+_KELVIN_)/ sc.temp
    ctrl_tbc.temp_max = (ctrl_tbc.temp_max+_KELVIN_)/ sc.temp
    ctrl_tbc.end_time = ctrl_tbc.end_time * scal_factor
    ctrl_tbc.dt = ctrl_tbc.dt * scal_factor
    ctrl_tbc.slab_age = ctrl_tbc.slab_age * scal_factor
    ctrl_tbc.interval_time = ctrl_tbc.interval_time * scal_factor
    ctrl_tbc.interval_val = ctrl_tbc.interval_val * scal_factor
    ctrl_tbc.dz = ctrl_tbc.dz*_KM_2_M_ / sc.length
    ctrl_tbc.k = ctrl_tbc.k / sc.k
    ctrl_tbc.cp = ctrl_tbc.cp / sc.cp
    ctrl_tbc.rho = ctrl_tbc.rho / sc.rho
    ctrl_tbc.right_age = ctrl_tbc.right_age * scal_factor
    return ctrl_tbc
    
def scaling_control_parameters(ctrl:NumericalControls,sc:Scal)->NumericalControls:
    """_summary_

    Args:
        ctrl (NumericalControls): numerical control class
        scal (Scal): scaling class

    Returns:
        NumericalControls: adimensional numerical controls
    """

    ctrl.g = ctrl.g / (sc.length/sc.time**2)
    ctrl.time_max = ctrl.time_max * (sc.scale_myr2sec/sc.time)
    ctrl.dt = ctrl.dt * (sc.scale_myr2sec/sc.time)

    return ctrl
def scaling_mesh(mesh:Mesh,sc:Scal)->Mesh:
    """_summary_

    Args:
        mesh (Mesh): Mesh object
        sc (Scal): scaling object

    Returns:
        Mesh: updated mesh object
    """
    
    domain = tuple(['global_domain','crust_domain',
                    'subduction_plate_domain','wedge_domain'])
    
    for i in domain: 
        sub = getattr(mesh, i)
        sub.mesh.geometry.x[:] /= sc.length/_KM_2_M_
    
    mesh.g_input = dimensionless_ginput(mesh.g_input,sc)
    
    return mesh

def dimensionless_ginput(g_input:GeomInput,sc:Scal):
    """_summary_

    Args:
        g_input (GeomInput): geometrical input
        sc (Scal): scaling object

    Returns:
        g_input: scaled geometrical input
    """
    scale_lenght = sc.length/_KM_2_M_
    g_input.x /= scale_lenght # main grid coordinate
    g_input.y /= scale_lenght
    g_input.cr /= scale_lenght # crust 
    g_input.ocr /= scale_lenght # oceanic crust
    g_input.lit_mt /= scale_lenght # lithosperic mantle  
    g_input.wz_tk /= scale_lenght # weak zone 
    g_input.ns_depth /= scale_lenght # total lithosphere thickness
    g_input.decoupling /= scale_lenght # decoupling depth -> i.e. where the weak zone is prolonged 
    g_input.resolution_normal /= scale_lenght  # To Do
    g_input.resolution_refine /= scale_lenght  # To Do
    g_input.transition /= scale_lenght # the transition between coupled and uncoupled
    g_input.lab_d /= scale_lenght # Astenosphere-lithosphere 
    g_input.slab_tk /= scale_lenght

    return g_input

def scale_kinematic_bc(ctrl_ky:CtrlKy, sc: Scal)->CtrlKy:

    ctrl_ky.v_s *= sc.scale_vel * (sc.time/sc.length)
    ctrl_ky.interval_val *= sc.scale_myr2sec * 1/sc.time
    ctrl_ky.interval_val *= sc.scale_myr2sec * 1/sc.time
    return ctrl_ky