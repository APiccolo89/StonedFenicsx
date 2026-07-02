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
        """Derive all secondary scaling units from the four primary ones.

        Called once during configuration after the primary scales (length,
        temp, eta, stress) have been set.  Fills every `field(init=False)`
        attribute according to standard dimensional analysis:
          - time   = eta / stress           [s]
          - mass   = stress * L^2 * t^2 / L [kg]
          - rho    = mass / L^3             [kg/m^3]
          - k      = watt / (L * T)         [W/(m·K)]
          - cp     = energy / (T * mass)    [J/(kg·K)]
          etc.
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

@timing_function
def scaling_simulation_physical(
    ctrl_sim: SimulationControls, pdb: PhaseDataBase, mesh: Mesh
,sc:Scal) -> None:
    """Non-dimensionalise the entire simulation state in-place.

    Dispatches to the five individual scaling functions in the correct
    dependency order so that all SI values are converted to dimensionless
    form before the FEM problem objects are constructed.  Must be called
    exactly once, after `Scal.compute_the_derivative_scal()` and before
    `initialise_the_simulation`.

    Args:
        ctrl_sim (SimulationControls): All simulation controls; fields are
            overwritten with their dimensionless equivalents.
        pdb (PhaseDataBase): Material-property database; fields are
            overwritten with their dimensionless equivalents.
        mesh (Mesh): Mesh object; geometry coordinates and geometric input
            are overwritten with their dimensionless equivalents.
        sc (Scal): Fully initialised scaling object (primary + derived scales).
    """
    scale_kinematic_bc(ctrl_ky=ctrl_sim.ctrl_ky,sc=sc)
    scale_parameters(ctrl_tbc=ctrl_sim.ctrl_tbc,sc=sc)
    scaling_control_parameters(ctrl=ctrl_sim.ctrl,sc=sc)
    scaling_mesh(mesh=mesh,sc=sc)
    scaling_material_properties(pdb=pdb,sc=sc)


def scaling_material_properties(pdb:PhaseDataBase,sc:Scal)->PhaseDataBase:
    """Non-dimensionalise all material-property parameters in the phase database.

    Divides every parameter in `pdb` by its composite SI scaling factor
    derived from `sc`.  Covers: reference temperature and pressure, viscosity
    bounds, thermal conductivity coefficients (a, b, c, d, e series and
    pressure correction k_f), heat-capacity polynomial coefficients
    (c0–c5), thermal expansivity (alpha0–alpha2), density (rho0, kb),
    radiogenic heat production, and dislocation/diffusion creep prefactors
    (b_dis, b_dif, bdis_wz).

    Args:
        pdb (PhaseDataBase): Material-property database; all fields overwritten
            in-place with dimensionless values.
        sc (Scal): Fully initialised scaling object.

    Returns:
        PhaseDataBase: The same `pdb` object with all fields scaled.
    """
    # scal the references values   
    pdb.temp_ref /= sc.temp
    pdb.pres_ref /= sc.stress
    pdb.temp_scal = sc.temp
    pdb.pres_scal = sc.stress
    pdb.tau_min /= sc.stress
    
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

def scale_parameters(ctrl_tbc:CtrlTemperatureBC,sc:Scal)->None:
    """Non-dimensionalise all thermal boundary-condition parameters.

    Converts temperatures from Celsius to Kelvin then divides by `sc.temp`.
    Converts all time quantities (end_time, dt, slab_age, interval_time,
    interval_val, right_age) from Myr to seconds then to dimensionless time
    using the viscous time scale `sc.time`.  Converts the depth discretisation
    step `dz` from km to metres then to dimensionless length.  Scales
    conductivity, heat capacity, and density by their respective composite
    scales.

    Args:
        ctrl_tbc (CtrlTemperatureBC): Thermal boundary-condition controls;
            all fields overwritten in-place with dimensionless values.
        sc (Scal): Fully initialised scaling object.
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

def scaling_control_parameters(ctrl:NumericalControls,sc:Scal)->None:
    """Non-dimensionalise numerical solver controls.

    Scales gravity `g` by the characteristic acceleration (length/time^2),
    and converts `time_max` and `dt` from Myr to the dimensionless viscous
    time unit.

    Args:
        ctrl (NumericalControls): Numerical controls dataclass; `g`,
            `time_max`, and `dt` are overwritten in-place.
        sc (Scal): Fully initialised scaling object.
    """

    ctrl.g = ctrl.g / (sc.length/sc.time**2)
    ctrl.time_max = ctrl.time_max * (sc.scale_myr2sec/sc.time)
    ctrl.dt = ctrl.dt * (sc.scale_myr2sec/sc.time)

def scaling_mesh(mesh:Mesh,sc:Scal)->None:
    """Non-dimensionalise mesh geometry and geometric input parameters.

    Divides all mesh node coordinates (in metres) by `sc.length / 1000` for
    the four sub-domains (global, crust, subduction plate, wedge), converting
    them from kilometres to dimensionless units.  Then calls
    `dimensionless_ginput` to scale the scalar geometric parameters stored in
    `mesh.g_input`.

    Args:
        mesh (Mesh): Mesh object whose sub-domain geometry arrays and
            `g_input` are overwritten in-place.
        sc (Scal): Fully initialised scaling object.
    """
    
    domain = tuple(['global_domain','crust_domain',
                    'subduction_plate_domain','wedge_domain'])
    
    for i in domain: 
        sub = getattr(mesh, i)
        sub.mesh.geometry.x[:] /= sc.length/_KM_2_M_
    
    dimensionless_ginput(mesh.g_input,sc)

def dimensionless_ginput(g_input:GeomInput,sc:Scal)->None:
    """Non-dimensionalise all scalar geometric parameters in GeomInput.

    Divides every length parameter (grid extents, layer thicknesses, weak zone
    thickness, decoupling depth, mesh resolution targets, LAB depth, slab
    thickness) by `sc.length / 1000` to convert from kilometres to the
    dimensionless length unit.

    Args:
        g_input (GeomInput): Geometric input dataclass; all length fields
            are overwritten in-place.
        sc (Scal): Fully initialised scaling object.
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

def scale_kinematic_bc(ctrl_ky:CtrlKy, sc: Scal)->None:
    """Non-dimensionalise kinematic boundary-condition parameters.

    Converts the slab surface velocity `v_s` from cm/yr to m/s using
    `sc.scale_vel`, then to the dimensionless velocity unit (length/time).
    Converts the time-series interval array `interval_val` from Myr to the
    dimensionless time unit.

    Args:
        ctrl_ky (CtrlKy): Kinematic boundary-condition controls; `v_s` and
            `interval_val` are overwritten in-place.
        sc (Scal): Fully initialised scaling object.
    """
    ctrl_ky.v_s *= sc.scale_vel * (sc.time/sc.length)
    ctrl_ky.interval_val *= sc.scale_myr2sec * 1/sc.time