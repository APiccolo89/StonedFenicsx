# stonedFEnicsX, A. Piccolo: andreapiccolope@gmail.com for informations
from __future__ import annotations
from typing import Callable,Any
from dataclasses import dataclass, field, InitVar
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import Domain, GeomInput
from stonedfenicsx.config.phase_db import PhaseDataBase
from .solver import Solvers
import dolfinx 
import basix
import ufl 
import numpy as np

# --- 
@dataclass
class FEMFORM:
    pass

# ---------------------------------------------------------------------------------
@dataclass
class MATERIALS:
    pdb : InitVar[PhaseDataBase]
    phase : InitVar[dolfinx.fem.Function]

@dataclass
class THERMALCACHED(MATERIALS):
    """Function containing the fem.function per each of the parameter of material properties
    Member variables:
    k0: constant conductivity
    fr: radiogenic flag
    k_a, k_b, k_c, k_d, k_e, k_f: conductivity parameters
    C0, C1, C2, C3, C4, C5: heat capacity parameters
    rho0, alpha0, alpha1, alpha2, Kb: density parameters and alpha parameters
    option_rho: flag to choose the density formulation
    radio: radiogenic heating
    Tref, R, A, B, T_A, x_A, T_B, x_B: parameters for the radiative conductivity    
    """
    # Heat Conductivity properties
    k0 : dolfinx.fem.Function = None 
    fr : dolfinx.fem.Function = None
    k_a: dolfinx.fem.Function = None
    k_b: dolfinx.fem.Function = None
    k_c: dolfinx.fem.Function = None
    k_d: dolfinx.fem.Function = None
    k_e: dolfinx.fem.Function = None
    k_f: dolfinx.fem.Function = None
    # Heat Capacity properties
    c0 : dolfinx.fem.Function = None
    c1 : dolfinx.fem.Function = None
    c2 : dolfinx.fem.Function = None
    c3 : dolfinx.fem.Function = None
    c4 : dolfinx.fem.Function = None
    c5 : dolfinx.fem.Function = None
    # Density properties
    rho0    : dolfinx.fem.Function = None
    alpha0  : dolfinx.fem.Function = None
    alpha1  : dolfinx.fem.Function = None
    alpha2  : dolfinx.fem.Function = None
    kb      : dolfinx.fem.Function = None
    option_rho : dolfinx.fem.Function = None
    # Radiogenic heating properties
    radiogenic   : dolfinx.fem.Function = None
    # Other properties
    temp_ref : float = 0.0
    gas_constant : float = 8.3145
    a_rad : float = 0.0
    b_rad : float = 0.0
    temp_a : float = 0.0
    x_a : float = 0.0
    temp_b : float = 0.0
    x_b : float = 0.0
# ---------------------------------------------------------------------------------
    def __post_init__(self
                     ,pdb:PhaseDataBase
                     ,phase:dolfinx.fem.Function)->None:

        """    Initialize all thermal properties as FEniCS functions.
        Args:        
            self (Functions_material_properties_global): class of fem.function that will contain all the material properties as fem.function
            pdb (PhaseDataBase): class containing the material properties as numpy arrays, indexed by phase ID
            phase (fem.Function): function containing the phase ID for each cell, used to index the material properties from the PhaseDataBase
        Returns:
            self (Function_material_properties): update self with the material properties as fem.function

        Note: The field in this version of the code are static. They are not advected. It is necessary to call it 
        once during a preprocessing step, after the phase is defined, and before the solver routine is called. 

        """

        ph = np.int32(phase.x.array)
        ph_fs = phase.function_space

        # Heat Conductivity properties
        self.k0 = dolfinx.fem.Function(ph_fs)  
        self.fr = dolfinx.fem.Function(ph_fs)  
        self.k_a= dolfinx.fem.Function(ph_fs)  
        self.k_b= dolfinx.fem.Function(ph_fs)  
        self.k_c= dolfinx.fem.Function(ph_fs)  
        self.k_d= dolfinx.fem.Function(ph_fs)  
        self.k_e= dolfinx.fem.Function(ph_fs)  
        self.k_f= dolfinx.fem.Function(ph_fs)
        self.k0.x.array[:] =  pdb.k0[ph]
        self.fr.x.array[:] =  pdb.radio_flag[ph]
        self.k_a.x.array[:] = pdb.k_a[ph]
        self.k_b.x.array[:] = pdb.k_b[ph]
        self.k_c.x.array[:] = pdb.k_c[ph]
        self.k_d.x.array[:] = pdb.k_d[ph]
        self.k_e.x.array[:] = pdb.k_e[ph]
        self.k_f.x.array[:] = pdb.k_f[ph]
        # Heat Capacity properties
        self.c0 = dolfinx.fem.Function(ph_fs)
        self.c1 = dolfinx.fem.Function(ph_fs) 
        self.c2 = dolfinx.fem.Function(ph_fs) 
        self.c3 = dolfinx.fem.Function(ph_fs) 
        self.c4 = dolfinx.fem.Function(ph_fs) 
        self.c5 = dolfinx.fem.Function(ph_fs) 
        self.c0.x.array[:] = pdb.c0[ph]
        self.c1.x.array[:] = pdb.c1[ph]
        self.c2.x.array[:] = pdb.c2[ph]
        self.c3.x.array[:] = pdb.c3[ph]
        self.c4.x.array[:] = pdb.c4[ph]
        self.c5.x.array[:] = pdb.c5[ph]
        # Density properties
        self.rho0    =    dolfinx.fem.Function(ph_fs)   
        self.alpha0  =    dolfinx.fem.Function(ph_fs)   
        self.alpha1  =    dolfinx.fem.Function(ph_fs)   
        self.alpha2  =    dolfinx.fem.Function(ph_fs)   
        self.kb      =    dolfinx.fem.Function(ph_fs)   
        self.option_rho = dolfinx.fem.Function(ph_fs)
        self.rho0.x.array[:]    = pdb.rho0[ph]
        self.alpha0.x.array[:]  = pdb.alpha0[ph]
        self.alpha1.x.array[:]  = pdb.alpha1[ph]
        self.alpha2.x.array[:]  = pdb.alpha2[ph]
        self.kb.x.array[:]      = pdb.kb[ph]
        self.option_rho.x.array[:] = pdb.option_rho[ph]

        self.radiogenic   = dolfinx.fem.Function(ph_fs)
        self.radiogenic.x.array[:]     = pdb.radiogenic_heat[ph]
        self.radiogenic.x.scatter_forward()
        
        self.temp_ref    = pdb.temp_ref
        self.gas_constant       = pdb.gas_constant
        self.a_rad       = pdb.a_rad
        self.b_rad       = pdb.b_rad
        self.temp_a     = pdb.temp_a
        self.x_a     = pdb.x_a
        self.temp_b     = pdb.temp_b
        self.x_b     = pdb.x_b

   


@dataclass
class RHEOLOGYCACHED(MATERIALS):
    """Initialise the rheological properties 
    
    Note: Stokes equation can be evaluated in a sub-domain (wedge). The class
    is separated from the main material properties dataclass for this reason. 
    """
    b_dif    : dolfinx.fem.Function = None
    b_dis    : dolfinx.fem.Function= None
    n       : dolfinx.fem.Function = None
    e_dif   : dolfinx.fem.Function = None
    e_dis   : dolfinx.fem.Function = None
    v_dif   : dolfinx.fem.Function = None
    v_dis   : dolfinx.fem.Function = None
    eta     : dolfinx.fem.Function = None
    eta_def : dolfinx.fem.Function = None 
    option_eta : dolfinx.fem.Function = None
    eta_max : float = None
    gas_constant     : float = None

    def __post_init__(self,
                      pdb:PhaseDataBase
                      ,phase:dolfinx.fem.Function)->None:
        """Change the rheological properties as fem.function, given the phase distribution and the PhaseDataBase.

        Args:
            self (Functions_material_rheology): Rheological properties as fem.function
            pdb (PhaseDataBase): Phase Data Base containing the rheological properties as numpy arrays, indexed by phase ID
            phase (fem.Function): function containing the phase ID for each cell, used to index the material properties from the PhaseDataBase

        Returns:
            Functions_material_rheology: updated Functions_material_rheology with the rheological properties as fem.function
        """
        ph = np.int32(phase.x.array)
        ph_fs = phase.function_space

        self.b_dif    = dolfinx.fem.Function(ph_fs)  
        self.b_dis    = dolfinx.fem.Function(ph_fs)  
        self.n       = dolfinx.fem.Function(ph_fs)   
        self.e_dif    = dolfinx.fem.Function(ph_fs)  
        self.e_dis    = dolfinx.fem.Function(ph_fs)  
        self.v_dif    = dolfinx.fem.Function(ph_fs)  
        self.v_dis    = dolfinx.fem.Function(ph_fs)  
        self.eta     = dolfinx.fem.Function(ph_fs)   
        self.option_eta = dolfinx.fem.Function(ph_fs)
        self.b_dif.x.array[:]     = pdb.b_dif[ph]
        self.b_dis.x.array[:]     = pdb.b_dis[ph]
        self.n.x.array[:]        = pdb.n[ph]
        self.e_dif.x.array[:]     = pdb.e_dif[ph]
        self.e_dis.x.array[:]     = pdb.e_dis[ph]
        self.v_dif.x.array[:]     = pdb.v_dif[ph]
        self.v_dis.x.array[:]     = pdb.v_dis[ph]
        self.eta.x.array[:]      = pdb.eta[ph]
        self.option_eta.x.array[:] = pdb.option_eta[ph]
        self.eta_max = pdb.eta_max
        self.eta_def = pdb.eta_def
        self.gas_constant       = pdb.gas_constant
        self.eta.x.scatter_forward()


@dataclass
class Solutions:
    """_summary_
    """
    vel_el: InitVar[basix.ufl.element]
    pt_el: InitVar[basix.ufl.element]
    domain: InitVar[Domain]
    
    temp_new: dolfinx.fem.Function = field(init=False)
    temp_old: dolfinx.fem.Function = field(init=False)
    u: dolfinx.fem.Function = field(init=False)
    u_old: dolfinx.fem.Function = field(init=False)
    p_lit: dolfinx.fem.Function = field(init=False)
    p_dyn: dolfinx.fem.Function = field(init=False)
    
    def __post_init__(self,vel_el:basix.ufl.element,
                       pt_el:basix.ufl.element,
                       domain:Domain,
                       pd_el:bool=False):
        fs_vel = dolfinx.fem.FunctionSpace(domain.mesh,vel_el)
        self.u = dolfinx.fem.Function(fs_vel.sub(0))
        self.u_old = dolfinx.fem.Function(fs_vel.sub(0))
        fs_pt = dolfinx.fem.FunctionSpace(domain.mesh,pt_el)
        self.temp_new = dolfinx.fem.Function(fs_pt)
        self.temp_old = dolfinx.fem.Function(fs_pt)
        self.p_lit = dolfinx.fem.Function(fs_pt)
        if pd_el:
            fs_pd = dolfinx.fem.FunctionSpace(domain.mesh,pd_el)
            self.p_dyn = dolfinx.fem.Function(fs_pd)

@dataclass
class  ProblemOBJ:
    """
    Container and dispatcher for a simulation problem.

    This object keeps references to the domain, controls, solver, material
    database, function spaces, boundary conditions, and form-building
    callbacks required to solve one problem.
    """
    
    domain: Domain = field(init=False)
    ctrl_sim: SimulationControls 
    g_input: GeomInput
    solv: Solvers
    pdb : PhaseDataBase
    g : float | None = field(init=False)
    cached_mat: MATERIALS = field(init=False)
    f_make_exp: Callable[[ProblemOBJ,Solutions,int,float],Any] = field(init=False)
    f_create_solver:Callable[...,Any] = field(init=False)
    f_solve_lin: Callable[[ProblemOBJ,Solutions,int,float],Any] = field(init=False)
    f_solve_pic: Callable[[ProblemOBJ,Solutions,int,float],Any] = field(init=False)
    f_create_bc: Callable[[ProblemOBJ,Solutions,int,float],Any] = field(init=False) 
    type_problem:str = 'Linear'
    bc: list = field(default_factory=list)
    func_space: dolfinx.fem.FunctionSpace | None = field(init=False)
    func_space_aux: dolfinx.fem.FunctionSpace = field(init=False)
    dx: dolfinx.ufl.measure = field(init=False)
    ds: dolfinx.ufl.measure = field(init=False)
    ds_ext: dolfinx.ufl.measure = field(init=False)
    moving_wall_ref: dolfinx.fem.Function|None = field(init=False)
    moving_wall: dolfinx.fem.Function | None = field(init=False)
    shear_heating:dolfinx.fem.Function|None = field(init=False)
    elements:list = field(default_factory=list)
    
    def configure_problem(self,
                          domain:Domain
                          ,g_input:GeomInput
                          ,ctrl_sim: SimulationControls
                          ,material_property_problem:str
                          ,elements:tuple):
        """ Configure the problem 
    
        Args:
            domain (Domain): _description_
            g_input (GeomInput): _description_
            ctrl_sim (SimulationControls): _description_
            elements (tuple): _description_
        """
        self.domain = domain
        self.ctrl_sim = ctrl_sim
        self.g_input = g_input
        if len(elements)==1:
            self.func_space = dolfinx.fem.functionspace(domain.mesh, elements[0])
        else: 
            mixed_element = basix.ufl.mixed_element([elements[0], elements[1]])
            self.func_space = dolfinx.fem.functionspace(domain.mesh, mixed_element)
            self.func_space_aux = dolfinx.fem.functionspace(domain.mesh,elements[2])
            self.moving_wall
        
        self.dx = dolfinx.ufl.Measure("dx", domain=domain.mesh)
        self.ds = dolfinx.ufl.Measure("ds", domain=domain.mesh, subdomain_data=domain.facets) # Exterior -> for boundary external 
        self.ds_ext = dolfinx.ufl.Measure("dS", domain=domain.mesh, subdomain_data=domain.facets)
        self.cached_mat = self.configure_material_properties_cache(material_property_problem)

    def configure_material_properties_cache(self,material_property_problem:str)->MATERIALS:
        """_summary_

        Args:
            material_property_problem (str): _description_
        """
        if material_property_problem == 'Scalar':
            cached_mat = THERMALCACHED(pdb=self.pdb,phase=self.domain.phase)
        elif material_property_problem == 'Stokes':
            cached_mat = RHEOLOGYCACHED(pdb=self.pdb,phase=self.domain.phase)
        else:
            raise ValueError('wrong signature')
        
        return cached_mat
    
    def solve_the_problem(self,sol:Solutions,it:int,ts:int)->None:
        """Use the information stored in the problem 
        to modify the solution input object

        Args:
            sol (Solutions): _description_
        """
        form = self.f_make_exp(self,sol)
        self.f_create_bc(self,sol)
        self.solv = self.f_create_solver(self,obj)
        # update solver need to check how
        if self.type_problem == 'Linear':
            self.f_solve_lin(self,sol)
        else:
            self.f_solve_pic(self,sol)
