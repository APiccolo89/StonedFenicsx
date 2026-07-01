# --- libraries --- 
import dolfinx
import basix 
import ufl
import numpy as np 
import mpi4py.MPI as MPI
from numpy.typing import NDArray
from scipy.interpolate import griddata   
import petsc4py.PETSc as PETSc
# --- ufl 

# --- from config module 
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import Mesh, GeomInput, Domain
from stonedfenicsx.config.phase_db import PhaseDataBase
from stonedfenicsx.config.scal import Scal
# --- from solver module
from stonedfenicsx.solver_module.solver import Solvers,ScalarSolver,SolverStokes
from stonedfenicsx.solver_module.solver_utilities import (decoupling_function
                                                          ,update_solution
                                                          ,compute_residuum
                                                          ,min_max_array)
from stonedfenicsx.utils import compute_strain_rate
# --- from material properties 
from stonedfenicsx.material_property.compute_material_property import (compute_plastic_strain
                                                                       ,MATERIALS,RHEOLOGYCACHED,THERMALCACHED
                                                                       ,compute_radiogenic,
                                                                       density_FX,heat_capacity_FX,heat_conductivity_FX
                                                                       ,compute_viscosity_FX)
# --- from src 
from stonedfenicsx.utils import print_ph,timing_function
import time as timing 

def debug_boundary_condition(bc, name):
    comm = MPI.COMM_WORLD

    # bc.dof_indices() returns (dofs, first_ghost) in many dolfinx versions
    dofs, first_ghost = bc.dof_indices()

    # owned dofs are dofs[:first_ghost] (this ordering IS guaranteed for dof_indices)
    dofs_owned = dofs[:first_ghost]

    if dofs_owned.size == 0:
        local_min = np.inf
        local_max = -np.inf
    else:
        if hasattr(bc.g, "x"):  # Function
            vals = bc.g.x.array[dofs_owned]
            vals = np.linalg.norm(vals, axis=1)
        else:  # Constant / ndarray
            vals = np.asarray(bc.g.value)
        local_min = np.nanmin(vals)
        local_max = np.nanmax(vals)

    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print(f"{name}: global min {gmin:.6e} global max {gmax:.6e} "
              f"(rank{comm.rank} local min {local_min:.6e} local max {local_max:.6e}, n_owned {dofs_owned.size})")
        
# ---
class Problem:
    """
    Abstract problem super-class defining the common structure and metadata
    for all FEM problems in the framework.

    This class stores the function spaces, variational arguments, boundary
    conditions, integration measures, and solver interface required to
    assemble and solve a finite-element problem. Concrete problem classes
    (e.g., thermal, Stokes, slab, wedge) should inherit from this class and
    specialize the relevant fields.

    Attributes:
        name (list[str]):
            Identifiers of the problem and associated computational domains
            (e.g., ["global"], ["wedge"], ["slab"]).

        mixed (bool):
            Whether the problem is mixed (multiple coupled function spaces),
            such as the Stokes problem with velocity and pressure.

        FS (dolfinx.fem.FunctionSpace):
            Primary function space of the problem.

        F0 (dolfinx.fem.FunctionSpace | None):
            First subspace of a mixed formulation, if present.

        F1 (dolfinx.fem.FunctionSpace | None):
            Second subspace of a mixed formulation, if present.

        trial0 (ufl.Argument | None):
            Trial function associated with `F0` or the primary space.

        trial1 (ufl.Argument | None):
            Trial function associated with `F1` in mixed problems.

        test0 (ufl.Argument | None):
            Test function associated with `F0` or the primary space.

        test1 (ufl.Argument | None):
            Test function associated with `F1` in mixed problems.

        typology (str | None):
            Problem type descriptor (e.g., "linear", "nonlinear").

        dofs (np.ndarray | None):
            Degrees of freedom constrained by boundary conditions.

        bc (list):
            Collection of Dirichlet boundary conditions applied to the problem.

        ds (ufl.Measure):
            Boundary integration measure (surface/edge integrals).

        dx (ufl.Measure):
            Domain integration measure (volume/area integrals).

        solv (Solvers):
            Solver interface or container defining the numerical solution
            strategy for the problem.
    """

    name      : list                               # name of the problem, domain [global, domainA...]
    mixed     : bool                               # is a mixed problem (e.g. Stokes problem has two function spaces: velocity and pressure)
    domain    : Domain
    ctrl_sim  : SimulationControls
    g_input   : GeomInput
    pdb       : PhaseDataBase
    cached_mat : MATERIALS
    FS        : dolfinx.fem.FunctionSpace          # Function space of the problem 
    F0        : dolfinx.fem.FunctionSpace | None   # Function space of the subspace 
    F1        : dolfinx.fem.FunctionSpace | None   # Function space of the subspace
    trial0    : ufl.Argument | None                # Trial 
    trial1    : ufl.Argument | None                # Trial
    test0     : ufl.Argument | None                # Test
    test1     : ufl.Argument | None                # Test 
    typology  : str | None                         # Linear/Non Linear
    dofs      : np.ndarray | None                  # Boundary dofs
    bc        : list                               # Dirichlet BC list
    ds        : ufl.measure.Measure                # measure surface/length 
    dx        : ufl.measure.Measure
    solv      : Solvers
    # --
    def __init__(self
                 ,mesh: Mesh
                 ,pdb:PhaseDataBase
                 ,ctrl_sim:SimulationControls
                 ,elements: tuple
                 ,name: list):
        """Initialize the problem

        Args:
            M (Mesh): Mesh object storing the computational domains of the experiment
            elements (tuple): Finite elements that describe the main computational problem.
                             Note: Certain problems require storing additional element and function space.
            name (list): Identifiers of the problem and associated domains. 

        Raises:
            NameError: If the user changes the name. 
            
        """


        self.name = name
        if name[1] not in ("global_domain", "subduction_plate_domain", "crust_domain", "wedge_domain"):
            raise NameError("Wrong domain name, check the spelling, in my case was it")
        elif name[1] == "crust_domain":
            print("Are you sure? crust_domain is junk for this problem.")

        self.domain = getattr(mesh,name[1])
        self.ctrl_sim = ctrl_sim
        self.pdb = pdb
        self.g_input = mesh.g_input
        

        if len(elements) == 1:
            self.mixed    = False
            self.FS       = dolfinx.fem.functionspace(self.domain.mesh, elements[0])
            self.trial0   = ufl.TrialFunction(self.FS)
            self.test0    = ufl.TestFunction(self.FS)
            self.trial1   = None
            self.test1    = None
        else: 
            self.mixed    = True
            mixed_element = basix.ufl.mixed_element([elements[0], elements[1]])
            self.FS       = dolfinx.fem.functionspace(self.domain.mesh, mixed_element) # MA cristoiddio, perche' cazzo hanno messo FunctionSpace and functionspace come nomi, ma sono degli stronzi?
            # 
            self.F0,_       = self.FS.sub(0).collapse()
            self.F1,_       = self.FS.sub(1).collapse()
            # Define trial/test on mixed FS
            self.trial0   = ufl.TrialFunction(self.FS.sub(0).collapse()[0])
            self.trial1   = ufl.TrialFunction(self.FS.sub(1).collapse()[0])
            self.test0    = ufl.TestFunction(self.FS.sub(0).collapse()[0])
            self.test1    = ufl.TestFunction(self.FS.sub(1).collapse()[0])
        
        self.dx       = ufl.Measure("dx", domain=self.domain.mesh)
        self.ds       = ufl.Measure("ds", domain=self.domain.mesh, subdomain_data=self.domain.facets) # Exterior -> for boundary external 
        self.dS       = ufl.Measure("dS", domain=self.domain.mesh, subdomain_data=self.domain.facets) # Interior -> for boundary integral inside
        
    def create_cached_material(self,scalar:bool):
        """Create the cached material for each class -> slightly an overkill for the pressure problem, 
        I know. 

        Args:
            scalar (bool): flag rheological or thermal material property
        """
        if scalar:
            self.cached_mat = THERMALCACHED(pdb=self.pdb,phase=self.domain.phase)
        else:
            self.cached_mat = RHEOLOGYCACHED(pdb=self.pdb,phase=self.domain.phase)

#--------------------------------------------------------------------------------------------------------------
class Solution():
    def __init__(self):
        self.PL : dolfinx.fem.function.Function 
        self.T_i : dolfinx.fem.function.Function
        self.T_O : dolfinx.fem.function.Function 
        self.T_N : dolfinx.fem.function.Function 
        self.u_global : dolfinx.fem.function.Function
        self.u_wedge : dolfinx.fem.function.Function
        self.p_lwedge : dolfinx.fem.function.Function
        self.t_owedge : dolfinx.fem.function.Function
        self.p_lslab : dolfinx.fem.function.Function
        self.t_oslab : dolfinx.fem.function.Function
        self.u_slab : dolfinx.fem.function.Function
        self.p_global: dolfinx.fem.function.Function 
        self.p_wedge : dolfinx.fem.function.Function 
        self.p_slab : dolfinx.fem.function.Function
        self.Hs_wedge : dolfinx.fem.function.Function
        self.Hs_slab : dolfinx.fem.function.Function
        self.shear_heating : dolfinx.fem.function.Function
        self.T_ad : dolfinx.fem.function.Function
        self.outer_iteration : NDArray[:]
        self.mT : NDArray[:]
        self.MT : NDArray[:]     
        self.RMST : NDArray[:]
        self.mv : NDArray[:]
        self.Mv : NDArray[:]     
        self.RMSv : NDArray[:]
        self.ts : NDArray[:]
        
    def create_function(self
                        ,PG:Problem
                        ,PS:Problem
                        ,PW:Problem
                        ,elements:list)->None: 
        """Create the 'fem.Function' for storing the solution of each of the problem 

        Args:
            PG Problem(Global_thermal|Global_lithostatic): Global problem (either Thermal or lithostatic)
            PS Problem(Slab): Subducting plate problem
            PW Problem(Wedge): Wedge problem 
            elements list: Elements for each of the problem (i.e. -> vectorial or scalar)

        Returns:
            Solution: Updated solution class with cached function. 
            
        Notes: Creating a more general Solution class and create specific istance for any given problem

        """

        mixed_element = basix.ufl.mixed_element([elements[0], elements[1]])

        space_GL = dolfinx.fem.functionspace(PG.FS.mesh,mixed_element) # PORCO DIO 
        
        
        def gives_Function(space):
            Va = space.sub(0)
            Pa = space.sub(1)
            V,_  = Va.collapse()
            P,_  = Pa.collapse()
            a = dolfinx.fem.Function(V)
            b = dolfinx.fem.Function(P)
            return a,b
        
        self.PL       = dolfinx.fem.Function(PG.FS) # Thermal and Pressure problems share the same functional space -> Need to enforce this bullshit 
        self.T_O      = dolfinx.fem.Function(PG.FS) 
        self.T_N      = dolfinx.fem.Function(PG.FS)
        self.Hs_global = dolfinx.fem.Function(PG.FS)
        self.Hs_slab  = dolfinx.fem.Function(PS.FSPT)
        self.Hs_wedge = dolfinx.fem.Function(PW.FSPT)
        self.T_i      = dolfinx.fem.Function(PG.FS)
        self.p_lwedge = dolfinx.fem.Function(PW.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_owedge = dolfinx.fem.Function(PW.FSPT) # same stuff as before, again, this is a nightmare: why the fuck. 
        self.p_lslab = dolfinx.fem.Function(PS.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_oslab = dolfinx.fem.Function(PS.FSPT)
        self.u_global, self.p_global = gives_Function(space_GL)
        self.u_slab  , self.p_slab   = gives_Function(PS.FS)
        self.u_wedge , self.p_wedge  = gives_Function(PW.FS)
        self.T_ad                     = dolfinx.fem.Function(PG.FS)   
        self.mT    = np.zeros(1,dtype=float)
        self.MT    = np.zeros(1,dtype=float) 
        self.RMST    = np.zeros(1,dtype=float)
        self.Mv    = np.zeros(1,dtype=float) 
        self.mv   = np.zeros(1,dtype=float)
        self.RMSv    = np.zeros(1,dtype=float) 
        self.outer_iteration = np.zeros(1,dtype=float)
        self.ts             = np.zeros(1,dtype=int)

# --- 
 
class Global_thermal(Problem):
    def __init__(self,mesh:Mesh, elements:tuple, name:list,pdb:PhaseDataBase,ctrl_sim:SimulationControls):
        super().__init__(mesh=mesh,elements=elements,name=name,ctrl_sim=ctrl_sim,pdb=pdb)
                
        self.steady_state = ctrl_sim.ctrl.steady_state
        
        if np.all(self.pdb.option_rho<2) and np.all(self.pdb.option_k==0) and np.all(self.pdb.option_cp==0) and self.ctrl_sim.ctrl.model_shear == 0:
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'

        self.bc_left = None 
        self.bc_right_wed = None
        self.bc_bot_wed = None
        self.bc_right_lit = None
        self.bc_top = None
        self.energy_source = dolfinx.fem.Function(self.FS)
        self.shear_heating = dolfinx.fem.Function(self.FS)
        
        
    @staticmethod
    def interpolate_1d_vector_boundary(function_space, z, temp_vec, dofs_intp):
            buf_fct = dolfinx.fem.Function(function_space)
            buf_fct.x.array[:] = griddata(z, temp_vec, dofs_intp[:,1], method='nearest')
            buf_fct.x.scatter_forward()
            return buf_fct
    
    def create_bc_temp(self,u_global:dolfinx.fem.Function,it_outer:int,ts=0)->list:
        """Create the boundary condition

        Args:
            u_global (dolfinx.fem.Function): velocity field global
            it_outer (int): outer iteration
            ts (int, optional): _description_. timestep

        Returns:
            list of boundary conditions
        """
        
        # UnPack the needed variables 
        cd_dof = self.FS.tabulate_dof_coordinates()
        domain = self.domain
        ctrl_tbc = self.ctrl_sim.ctrl_tbc 
        temp_min = self.ctrl_sim.ctrl_tbc.temp_top
        # This part can be done only once -> bc dofs are constant 
        
        if ctrl_tbc.constant != 1 or (it_outer == 0 and ts ==0): 
            # if the boundary condition is not constant, or if the outer iteration is equal to 0.0 and ts as well. 
            # Extract dofs
            facets                 = domain.facets.find(domain.bc_dict['Left_inlet'])
            dofs_left              = dolfinx.fem.locate_dofs_topological(self.FS, domain.mesh.topology.dim-1, facets)
            # Interpolate
            temp_bc_left = self.interpolate_1d_vector_boundary(self.FS,ctrl_tbc.z,ctrl_tbc.temperature_1d,cd_dof)
            # Update dirichletbc
            self.bc_left = dolfinx.fem.dirichletbc(temp_bc_left, dofs_left)

        if ts == 0 and it_outer == 0:
            # Top
            facets                 = domain.facets.find(domain.bc_dict['Top'])    
            dofs_top               = dolfinx.fem.locate_dofs_topological(self.FS, domain.mesh.topology.dim-1, facets)
            # -> Probably I need to use some parallel shit here 
            self.bc_top            = dolfinx.fem.dirichletbc(temp_min, dofs_top, self.FS)
            # Right Lithosphere
            facets                 = domain.facets.find(domain.bc_dict['Right_lit']) 
            dofs_right_lit              = dolfinx.fem.locate_dofs_topological(self.FS, domain.mesh.topology.dim-1, facets)          
            temp_bc_right = self.interpolate_1d_vector_boundary(self.FS,ctrl_tbc.z_right,ctrl_tbc.temp_1d_right,cd_dof)
            self.bc_right_lit = dolfinx.fem.dirichletbc(temp_bc_right, dofs_right_lit)

        # Right wedge
        facets                 = domain.facets.find(domain.bc_dict['Right_wed'])                        
        dofs_right_wed        = dolfinx.fem.locate_dofs_topological(self.FS, domain.mesh.topology.dim-1, facets)
        h_vel  = u_global.sub(0) # index 1 = y-direction (2D)
        vel_T  = dolfinx.fem.Function(self.FS)
        vel_T.interpolate(h_vel)
        vel_T.x.scatter_forward()
        vel_bc = vel_T.x.array[dofs_right_wed]
        ind_z = np.where((vel_bc < 0.0) & (cd_dof[dofs_right_wed,1]<=-self.g_input.lab_d))
        dofs_vel = dofs_right_wed[ind_z[0]] 
        self.bc_right_wed = dolfinx.fem.dirichletbc(ctrl_tbc.temp_max, dofs_vel,self.FS)
        
        # Bottom wedge
        facets                 = domain.facets.find(domain.bc_dict['Bottom_wed'])                        
        dofs_bot_wed        = dolfinx.fem.locate_dofs_topological(self.FS, domain.mesh.topology.dim-1, facets)
        v_vel = u_global.sub(1) # index 1 = y-direction (2D)
        vel_T = dolfinx.fem.Function(self.FS)
        vel_T.interpolate(v_vel)
        vel_T.x.scatter_forward()
        vel_bc = vel_T.x.array[dofs_bot_wed]
        ind_z = np.where((vel_bc > 0.0))
        dofs_vel = dofs_bot_wed[ind_z[0]]   
                    
        
        self.bc_bot_wed = dolfinx.fem.dirichletbc(ctrl_tbc.temp_max, dofs_vel,self.FS)
        
        bc = [ self.bc_left,  self.bc_right_wed,self.bc_bot_wed, self.bc_right_lit,self.bc_top]
        
        return bc
        
    #------------------------------------------------------------------
    def compute_shear_heating(self
                              ,p:dolfinx.fem.Function
                              ,T_k:dolfinx.fem.Function)->dolfinx.fem.Expression:
        """
        Apperently the sociopath the devise this method, uses a delta function to describe 
        the interface frictional heating. 
        -> [A] => Shear heating becomes a ufl expression. So happy about it 
        
        """
        domain = self.domain
        mode_shear = self.ctrl_sim.ctrl.mode_shear
        expression = dolfinx.fem.Constant(domain.mesh,(0.0))
        if self.ctrl_sim.ctrl.decoupling == 1 and mode_shear:

            heat_source = dolfinx.fem.Function(self.FS)
            heat_source.x.array[:] = 0.0
            heat_source.x.scatter_forward()
        
            decoupling    = heat_source.copy()
            Z = self.FS.tabulate_dof_coordinates()[:,1]
            decoupling = decoupling_function(Z,decoupling,self.g_input)

            if mode_shear:
                # compute the plastic strain rate ratio and viscous shear heating strain rate 
                # Place holder function
                dS = ufl.Measure("dS", domain=domain.mesh, subdomain_data=domain.facets)
                tau_eff, _, _  = self.compute_friction_shear_expression(T_k,p)

                friction_heat = tau_eff * decoupling * self.ctrl_sim.ctrl_ky.v_s[0]
                
                expression = friction_heat('+') * self.test0('+') * (dS(domain.bc_dict['Subduction_top_lit']) + dS(domain.bc_dict['Subduction_top_wed']))
            return expression
        else:
            return 0.0

    def compute_friction_shear_expression(self
                                          ,T:dolfinx.fem.function.Function
                                          ,P:dolfinx.fem.function.Function):
        """_summary_

        Args:
            pdb (PhaseDataBase): _description_
            T (dolfinx.fem.function.Function): _description_
            P (dolfinx.fem.function.Function): _description_

        Returns:
            _type_: _description_
        """
        # Compute the effective strain rate of the weak zone: -> couette-poisuille flow -> scalar
        e_ii_fr = 0.5 * (self.ctrl_sim.ctrl_ky.v_s[0] * 1 /self.g_input.wz_tk)  # Second invariant strain rate

        # -> compute the plastic strain rate []

        tau, tau_vs, tau_lim = compute_plastic_strain(e_ii_fr,T,P,self.pdb)

        return tau, tau_vs, tau_lim
            
    #------------------------------------------------------------------
    def compute_energy_source(self):
        source = dolfinx.fem.Function(self.FS)
        source = compute_radiogenic(self.cached_mat, source)
        self.energy_source = source.copy()
        self.energy_source.x.scatter_forward()
        

    #------------------------------------------------------------------
    def compute_residual_SS(self
                            ,p :dolfinx.fem.function.Function = None
                            ,T :dolfinx.fem.function.Function = None
                            ,T_O :dolfinx.fem.function.Function = None
                            ,u_global :dolfinx.fem.function.Function = None
                            ,it_inner:int=0
                            ,dt:float = 0.0)->float:



       
        rho_k = density_FX(self.cached_mat, T, p)  # frozen
        
        Cp_k = heat_capacity_FX(self.cached_mat, T)  # frozen

        k_k = heat_conductivity_FX(self.cached_mat, T, p, Cp_k, rho_k)  # frozen

        f    = self.energy_source# source term
        
        dx  = self.dx
            
        diff = ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        
        adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx
        if it_inner > 0 and self.ctrl_sim.ctrl.mode_shear:
            L = ((f) * self.test0 * dx + self.shear_heating) 
        else: 
            L = ((f) * self.test0 * dx )
        R = fem.form(diff + adv - L)
           
        
        # Conservation residual -> [save in the solution as well, for visualising it]   
        RT = dolfinx.fem.petsc.assemble_vector(fem.form(R))
        RT.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if getattr(self, "bc", None):
            with RT.localForm() as lf:
                r = lf.getArray(readonly=False)
                for bc in self.bc:
                    dofs = bc.dof_indices()[0]   # local dof indices
                    r[dofs] = 0.0

        RTemp = RT.norm(PETSc.NormType.NORM_2)    
        
        return RTemp
    

    def compute_residual_TD(self
                            ,p :dolfinx.fem.function.Function = None
                            ,T :dolfinx.fem.function.Function = None
                            ,T_O :dolfinx.fem.function.Function = None
                            ,u_global :dolfinx.fem.function.Function = None
                            ,it_inner:int=0
                            ,dt:float = 0.0)->float:


        rho_k = density_FX(self.cached_mat, T, p)  # frozen
                
        Cp_k = heat_capacity_FX(self.cached_mat, T)  # frozen

        k_k = heat_conductivity_FX(self.cached_mat, T, p, Cp_k, rho_k)  # frozen


        rho_k0 = density_FX(self.cached_mat, T_O, p)  # frozen
                
        Cp_k0 = heat_capacity_FX(self.cached_mat, T_O)  # frozen
        
        k_k0 = heat_conductivity_FX(self.cached_mat, T_O, p, Cp_k0, rho_k0)  # frozen


                
        rhocp        =  (rho_k * Cp_k)

        rhocp_old    =  (rho_k0 * Cp_k0)
        
        dx  = self.dx
        if self.ctrl_sim.ctrl.model_shear>0:
            f    = (self.energy_source) * self.test0 * dx + self.shear_heating # source term {energy_source is radiogenic heating compute before hand, shear heating is frictional heating already a form}
        else: 
            f    = (self.energy_source) * self.test0 * dx 
        
        # a -> New temperature 
        diff_new = ( 1 / 2 ) * ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        
        adv_new  = (rhocp / 2 )* ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx
        
        mass_new = (rhocp / dt) * T * self.test0 * dx
        
        new = diff_new + adv_new + mass_new  
                        
   
            
        adv_old =  - (rhocp_old / 2 ) * ufl.dot(u_global, ufl.grad(T_O)) * self.test0 * dx

        diff_old =  - ( 1 / 2 ) * ufl.inner(k_k0 * ufl.grad(T_O), ufl.grad(self.test0)) * dx
        
        mass_old =  (rhocp_old / dt) * T_O * self.test0 * dx
        
        old = diff_old + adv_old + f + mass_old
        
        R = new + old

           
        RT = fem.petsc.assemble_vector(fem.form(R))
        
        RT.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        if getattr(self, "bc", None):
            with RT.localForm() as lf:
                r = lf.getArray(readonly=False)
                for bc in self.bc:
                    dofs = bc.dof_indices()[0]   # local dof indices
                    r[dofs] = 0.0

        RTemp = RT.norm(PETSc.NormType.NORM_2)    
        
        
        return RTemp
    

    
    def set_linear_picard_SS(self
                             ,p:dolfinx.fem.Function=None
                             ,T_k:dolfinx.fem.Function = None
                             ,T_O:dolfinx.fem.Function=None
                             ,u_global:dolfinx.fem.Function=None
                             ,it_outer :int=0
                             ,it_inner:int =0
                             ,ts:int = 0)->tuple[dolfinx.fem.Form,dolfinx.fem.Form]:
        """Set up fem form for Steady state solution. 
        Compute the coefficient from the current iteration temperature, form the bilinear form. 
        if iteration > 0 not form the Linear one. 

        Args:
            p (dolfinx.fem.Function, optional): Lithostatic pressure function . Defaults to None.
            T_k (dolfinx.fem.Function, optional): Current iteration/guess temperature. Defaults to None.
            T_O (dolfinx.fem.Function, optional): Old temperature. Defaults to None.
            u_global (dolfinx.fem.Function, optional): Global velocity. Defaults to None.
            D (Domain, optional): Domain . Defaults to None.
            FG (Functions_material_properties_global, optional):Cached material properties. Defaults to None.
            ctrl (NumericalControls, optional): Numerical controls. Defaults to None.
            dt (float, optional): dt . Defaults to None.
            it (int, optional): outer iteration. Defaults to 0.

        Returns:
            tuple[dolfinx.fem.Form,dolfinx.fem.Form]: _description_
        """
        
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(self.cached_mat, T_k, p)  # frozen
        
        Cp_k = heat_capacity_FX(self.cached_mat, T_k)  # frozen

        k_k = heat_conductivity_FX(self.cached_mat, T_k, p, Cp_k, rho_k)  # frozen


        f    = self.energy_source# source term
        
        dx  = self.dx            
        
        diff = ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
            
        adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
        
        # SUPG 
        
        # --- SUPG parameter tau ---
        h = ufl.CellDiameter(self.domain.mesh)
        
        u_norm = ufl.sqrt(ufl.dot(u_global, u_global) + 1.0e-8)
        # Simple tau based on advection
        tau = h / (2.0 * u_norm+1e-12)
        
        
        residual = (rho_k * Cp_k * ufl.dot(u_global, ufl.grad(self.trial0)))
        
        SUPG = tau * ufl.dot(u_global, ufl.grad(self.test0)) * residual * self.dx
        
        SUPG_L = tau * ufl.dot(u_global, ufl.grad(self.test0)) * f * self.dx

        a = dolfinx.fem.form(diff + adv+SUPG)
        
        
        # Linear operator with frozen coefficients
        if it_inner != 0 and self.ctrl_sim.ctrl.model_shear>0:
            L = dolfinx.fem.form((f) * self.test0 * dx + SUPG_L +self.shear_heating) 
        else:     
            L = dolfinx.fem.form((f) * self.test0 * dx +SUPG_L) 
                

        return a, L
    #------------------------------------------------------------------

    def set_linear_picard_TD(self
                             ,p:dolfinx.fem.Function=None
                             ,T_k:dolfinx.fem.Function = None
                             ,T_O:dolfinx.fem.Function=None
                             ,u_global:dolfinx.fem.Function=None
                             ,it_outer :int=0
                             ,it_inner:int =0
                             ,ts:int = 0)->tuple[dolfinx.fem.Form,dolfinx.fem.Form|None]:
        # Function that set linear form and linear picard for picard iteration
        # Crank Nicolson scheme 
        # a - > New temperature 
        # L - > Old temperature
        # -> Source term is assumed constant in time and do not vary between the timesteps 

        dt = self.ctrl_sim.ctrl.dt

        rho_k = density_FX(self.cached_mat, T_k, p)  # frozen
                
        Cp_k = heat_capacity_FX(self.cached_mat, T_k)  # frozen

        k_k = heat_conductivity_FX(self.cached_mat, T_k, p, Cp_k, rho_k)  # frozen


        
        rho_k0 = density_FX(self.cached_mat, T_O, p)  # frozen
                
        Cp_k0 = heat_capacity_FX(self.cached_mat, T_O)  # frozen
        
        k_k0 = heat_conductivity_FX(self.cached_mat, T_O, p, Cp_k0, rho_k0)  # frozen


                
        rhocp        =  (rho_k * Cp_k)

        rhocp_old    =  (rho_k0 * Cp_k0)
        
        dx  = self.dx
        
        if self.ctrl_sim.ctrl.mode_shear>0 and it_inner !=0:
        
            f    = (self.energy_source) * self.test0 * dx + self.shear_heating # source term {energy_source is radiogenic heating compute before hand, shear heating is frictional heating already a form}

        else: 
            f = self.energy_source * self.test0 * dx 
        
        # a -> New temperature 
        diff_new = ( 1 / 2 ) * ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
        
        adv_new  = (rhocp / 2 )* ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
        
        mass_new = (rhocp / dt) * self.trial0 * self.test0 * dx
        
        a = dolfinx.fem.form(diff_new + adv_new + mass_new)
                
        if it_inner == 0: 
            
            adv_old =  - (rhocp_old / 2 ) * ufl.dot(u_global, ufl.grad(T_O)) * self.test0 * dx

            diff_old =  - ( 1 / 2 ) * ufl.inner(k_k0 * ufl.grad(T_O), ufl.grad(self.test0)) * dx
            
            mass_old =  (rhocp_old / dt) * T_O * self.test0 * dx
            
            L = dolfinx.fem.form(diff_old + adv_old + f + mass_old)
            
            return a, L

        else: 
            return a, None
    #------------------------------------------------------------------
    def Solve_the_Problem(self
                          ,sol:Solution
                          ,it_outer:int=0
                          ,ts:int=0)->[]: 
        
        nl = 0 
        # choose the problemesh: 
        if self.ctrl_sim.ctrl.steady_state == 1: 
            self.set_linear = self.set_linear_picard_SS 
            self.compute_residual = self.compute_residual_SS
        else: 
            self.set_linear = self.set_linear_picard_TD
            self.compute_residual = self.compute_residual_TD
                    
        if it_outer == 0:         
            self.compute_energy_source()
        
        a,L = self.set_linear(p=sol.PL
                              ,T_k=sol.T_N
                              ,T_O=sol.T_O
                              ,u_global=sol.u_global)
        
        self.bc = self.create_bc_temp(u_global=sol.u_global,it_outer=it_outer,ts=ts)

        if it_outer == 0 and ts == 0: 
            self.solv = ScalarSolver(a,L,self.bc,self.domain.comm,self.ctrl_sim.ctrl.energy_solver_type)
            
        
        print_ph('              // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if self.typology == 'LinearProblem': 
            sol.T_N = self.solve_the_linear(sol
                                          ,a
                                          ,L
                                          ,sol.T_N) 
            sol.T_N.x.scatter_forward()
            

        else: 
            
            sol = self.solve_the_non_linear(sol,it_outer=it_outer,ts=ts)
        
        time_B = timing.time()
        
        f_viz = fem.Function(self.FS)

        if self.ctrl_sim.ctrl.model_shear>0: 

                    
            u_trial = ufl.TrialFunction(self.FS)
            v_test  = ufl.TestFunction(self.FS)
            dx = ufl.Measure("dx", domain=self.domain.mesh)

            a_mass = (u_trial * v_test * dx)
            problem = LinearProblem(a_mass, (self.shear_heating))
            f_viz = problem.solve()

            print("shear min/max:", f_viz.x.array.min(), f_viz.x.array.max())            
            
        sol.shear_heating = f_viz.copy()
        
        print_ph(f'              // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return sol 

    #------------------------------------------------------------------
    
    def solve_the_linear(self
                         ,sol:Solution
                         ,a:dolfinx.fem.Form
                         ,L:dolfinx.fem.Form
                         ,fen_function:dolfinx.fem.Function
                         ,isPicard:int=0
                         ,ts:int=0)->dolfinx.fem.Function:
        
        # Update the matrix
        self.solv.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc)
        # Assemble
        self.solv.A.assemble()
        # Update b solve [IMPORTANT BEFORE I WAS NOT DOING AS PARALLEL] 
        with self.solv.b.localForm() as loc:
            loc.set(0.0)              
        
        dolfinx.fem.petsc.assemble_vector(self.solv.b, dolfinx.fem.form(L))
        dolfinx.fem.petsc.apply_lifting(self.solv.b, [dolfinx.fem.form(a)], [self.bc])
        
        self.solv.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        
        dolfinx.fem.petsc.set_bc(self.solv.b, self.bc)
        self.solv.ksp.solve(self.solv.b, fen_function.x.petsc_vec)
        dolfinx.fem.petsc.set_bc(fen_function.x.petsc_vec,bcs=self.bc)
        
 
        fen_function.x.scatter_forward()
        
        return fen_function

    def solve_the_non_linear(self
                            ,sol: Solution
                            ,it_outer:int=0
                            ,ts:int=0):  
        
        """_summary_

        Returns:
            _type_: _description_
        """
        
        ctrl = self.ctrl_sim.ctrl        
        tol = 1.0 
        T_O = sol.T_O
        T_k = sol.T_N.copy() 
        T_k1 = sol.T_N.copy()

        it_inner = 0 
        time_A = timing.time()
        print_ph('              [//] Picard iterations for the non linear temperature problem')

        while (it_inner < ctrl.it_inner_max and tol > ctrl.tol_innerpic) or it_inner < 2:
            
            self.shear_heating = self.compute_shear_heating(T_k=T_k
                                                        ,p=sol.PL)

            time_ita = timing.time()
            
            if it_inner == 0: 
                A,L = self.set_linear(p=sol.PL
                                    ,T_k=T_k
                                    ,T_O=sol.T_O
                                    ,u_global=sol.u_global)
                          
            else: 
                if ctrl.steady_state==1: 
                    A,L = self.set_linear(p=sol.PL
                                    ,T_k=T_k
                                    ,T_O=sol.T_O
                                    ,u_global=sol.u_global
                                    ,it_inner = it_inner)
                else: 
                    A,_ = self.set_linear(p=sol.PL
                                    ,T_k=T_k
                                    ,T_O=sol.T_O
                                    ,u_global=sol.u_global
                                    ,it_inner = it_inner)
                    
            T_k1 = self.solve_the_linear(sol,A,L,T_k1,1,it_outer)
            T_k1.x.scatter_forward()
            # L2 norm 
            tol = compute_residuum(T_k1,T_k)
            
            rT = self.compute_residual(p = sol.PL
                                  ,T = sol.T_N
                                  ,T_O = sol.T_O
                                  ,u_global = sol.u_global
                                  ,it_inner=it_inner)            
            if it_inner == 0: 
                rT0 = rT 
            
            
            time_itb = timing.time()
            print_ph(f'              []Temperature L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            print_ph(f'                          [\\] Residual L2 norm is {rT:.3e}, L2_rel is {rT/rT0:.3e}')

            T_k = update_solution(T_k1,T_k,ctrl.relax)

            it_inner = it_inner + 1 
        
        sol.T_N.x.array[:] = T_k1.x.array[:]
        sol.T_N.x.scatter_forward()        
        return sol  
        
    @timing_function
    def initial_temperature_field(self)->dolfinx.fem.Function:
        from scipy.interpolate import griddata
        from ufl import conditional, Or, eq
        from functools import reduce
        """
        
        """    
        #- Create part of the thermal field: create function, extract dofs, 
        ctrl_tbc = self.ctrl_sim.ctrl_tbc
        
        
        X     = self.FS
        T_i_A = dolfinx.fem.Function(X)
        cd_dof = X.tabulate_dof_coordinates()
        T_i_A.x.array[:] = griddata(ctrl_tbc.z, ctrl_tbc.temperature_1d, cd_dof[:,1], method='nearest')
        T_i_A.x.scatter_forward() 
        #- 

        T_expr = dolfinx.fem.Function(X)
        ind_A = np.where(cd_dof[:,1] >= -self.g_input.lab_d)[0]
        ind_B = np.where(cd_dof[:,1] < -self.g_input.lab_d)[0]
        T_expr.x.array[ind_A] = griddata(ctrl_tbc.z_right, ctrl_tbc.temp_1d_right, cd_dof[ind_A,1], method='nearest')
        T_expr.x.array[ind_B] = ctrl_tbc.temp_max
        T_expr.x.scatter_forward()
        T_i = dolfinx.fem.Function(X)
        expr = conditional(
            reduce(Or,[eq(self.domain.phase, i) for i in [2, 3, 4, 5]]),
            T_expr,
            T_i_A
        )
        T_i.interpolate(dolfinx.fem.Expression(expr, X.element.interpolation_points()))
        T_i.x.array[ind_B] = ctrl_tbc.temp_max
        T_i.x.scatter_forward()
        return T_i 

# ---
class Global_pressure(Problem): 
    def __init__(self
                 ,mesh:Mesh
                 ,elements:tuple
                 ,name:list
                 ,pdb:PhaseDataBase
                 ,ctrl_sim:SimulationControls):
        super().__init__(mesh=mesh,elements=elements,name=name,ctrl_sim=ctrl_sim,pdb=pdb)
        
        if np.all(pdb.option_rho<2):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'

        self.bc = [self.set_problem_bc()]
        self.g = dolfinx.fem.Constant(self.domain.mesh, PETSc.ScalarType([0.0,-self.ctrl_sim.ctrl.g]))
        

    
    def set_problem_bc(self)->list[dolfinx.fem.DirichletBC]:
         
        top_facets   = self.domain.facets.find(self.domain.bc_dict['Top'])
        top_dofs    = dolfinx.fem.locate_dofs_topological(self.FS, 1, top_facets)
        bc = [dolfinx.fem.dirichletbc(0.0, top_dofs, self.FS)]
        return bc  
    
    
    
    def set_linear_picard(self
                          ,p_k:dolfinx.fem.Function
                          ,T:dolfinx.fem.Function
                          ,it:int=0):
        # Function that set linear form and linear picard for picard iteration
        
        
        rho_k = density_FX(self.cached_mat, T, p_k)  # frozen
        
        # Linear operator with frozen coefficients
        if it == 0: 
            a = ufl.inner(ufl.grad(self.trial0), ufl.grad(self.test0)) * self.dx
        else: 
            a = None
        
        L = ufl.inner(ufl.grad(self.test0), rho_k * self.g) * self.dx
        

        return a, L

    def Solve_the_Problem(self
                          ,sol:Solution
                          ,it_outer:int=0
                          ,ts:int=0)->Solution: 
        
        p_k = sol.PL.copy()  # Previous lithostatic pressure 
        T   = sol.T_N.copy() # -> will not eventually update 
        
        # If the problem is linear, p_k is not doing anything, it is there because I
        # design the density function to receive in anycase a pressure, potentially I
        # can use the multipledispach of python, which say ah density with pressure 
        # density without pressure is equal to fuck. But seems a bit lame, and I do 
        # not think that is a great improvement of the code. 
        
        a,L = self.set_linear_picard(p_k,T)
    

        if it_outer == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,self.bc,self.domain.comm,self.ctrl_sim.ctrl.energy_solver_type)
        
        print_ph('              // -- // --- LITHOSTATIC PROBLEM [GLOBAL] // -- // --- > ')

        time_A = timing.time()

        
        if self.typology=='LinearProblem': 
            sol.PL = self.solve_the_linear(a,L,sol.PL) 
        else: 
            sol.PL = self.solve_the_non_linear(sol)

        time_B = timing.time()

        print_ph(f'              || -- || --- Solution of Lithostatic pressure problem finished in {time_B-time_A:.2f} sec || -- || ---||')
        print_ph('')

        return sol
    
    def solve_the_linear(self
                         ,a:dolfinx.fem.Form
                         ,L:dolfinx.fem.Form
                         ,function_fen:dolfinx.fem.Function
                         ,isPicard:int=0
                         ,it:int=0
                         ,ts:int=0):
        
        if it == 0 or ts == 0:
            self.solv.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self.solv.A,dolfinx.fem.form(a),self.bc[0])
            self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        with self.solv.b.localForm() as loc:
            loc.set(0.0)        
        dolfinx.fem.petsc.assemble_vector(self.solv.b, dolfinx.fem.form(L))
        dolfinx.fem.petsc.apply_lifting(self.solv.b, [dolfinx.fem.form(a)], self.bc)
        self.solv.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)        
        dolfinx.fem.petsc.set_bc(self.solv.b, self.bc[0])
        self.solv.ksp.solve(self.solv.b, function_fen.x.petsc_vec)
        function_fen.x.scatter_forward()
        
        return function_fen
    
    def solve_the_non_linear(self
                             ,sol:Solution):  
  
        ctrl = self.ctrl_sim.ctrl
        p_k = sol.PL.copy() 
        p_k1 = sol.PL.copy()

        it_inner = 0 
        
        
        print_ph('              [||] Picard iterations for the non linear lithostatic pressure problem')

        res = 1.0
        while it_inner < ctrl.it_inner_max and res > ctrl.tol_innerpic:
            time_ita = timing.time()
            
            if it_inner == 0:
                A,L = self.set_linear_picard(p_k,sol.T_N)
            else: 
                _,L = self.set_linear_picard(p_k,sol.T_N,1)
            
            p_k1 = self.solve_the_linear(A,L,p_k1,1,it_inner,1) 

            # L2 norm 
            res = compute_residuum(p_k1,p_k)
            
            time_itb = timing.time()
            print_ph(f'              it:[{it_inner:d}]: L_2 norm is {res:.3e} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            p_k = update_solution(p_k1,p_k,ctrl.relax)
            p_k.x.scatter_forward()
            
            it_inner = it_inner + 1 
        
        
        sol.PL.x.array[:] = p_k.x.array[:]
        sol.PL.x.scatter_forward()
      
        
        
        return sol.PL

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Stokes_Problem(Problem):
    def __init__(self
                 ,mesh:Mesh
                 ,elements:list
                 ,name:list
                 ,ctrl_sim:SimulationControls
                 ,pdb:PhaseDataBase):
        super().__init__(mesh=mesh,elements=elements,name=name,ctrl_sim=ctrl_sim,pdb=pdb)
        self.FSPT = dolfinx.fem.functionspace(self.domain.mesh, elements[2])     
        # Create the moving wall functions: 
        # Allocate memory
        self.moving_wall_ref = dolfinx.fem.Function(self.F0)
        self.moving_wall = dolfinx.fem.Function(self.F0)
    
    def fem_stokes_form(self,a1,a2,a3,a_p):
        a   = [[a1, a2],[a3, None]]
        a_p0  = [[a1, a2],[a3, a_p]]
        return a,a_p0
    
    def compute_residuum_stokes(self
                                ,u_new:dolfinx.fem.function.Function
                                ,p_new:dolfinx.fem.function.Function
                                ,temp:dolfinx.fem.function.Function
                                ,pres_l:dolfinx.fem.function.Function):
        V = u_new.function_space
        Q = p_new.function_space
        v = ufl.TestFunction(V)
        q = ufl.TestFunction(Q)

        e = compute_strain_rate(u_new)
        eta_new = compute_viscosity_FX(e, temp, pres_l, self.pdb,self.cached_mat)
        f = dolfinx.fem.Constant(self.domain.mesh, PETSc.ScalarType((0.0,) * self.domain.mesh.geometry.dim))
        dx = ufl.dx

        Fmom = (ufl.inner(2*eta_new*ufl.sym(ufl.grad(u_new)), ufl.sym(ufl.grad(v))) * dx
                - ufl.inner(p_new, ufl.div(v)) * dx
                - ufl.inner(f, v) * dx)

        Fdiv = ufl.inner(q, ufl.div(u_new)) * dx

        Rm = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Fmom))
        Rm.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if getattr(self, "bc", None):
            with Rm.localForm() as lf:
                r = lf.getArray(readonly=False)
                for bc in self.bc:
                    dofs = bc.dof_indices()[0]   # local dof indices
                    r[dofs] = 0.0

        rmom = Rm.norm(PETSc.NormType.NORM_2) 


        Rd = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Fdiv))
        Rd.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        rdiv = Rd.norm(PETSc.NormType.NORM_2) 

        return rmom, rdiv


    def set_linear_picard(self
                          ,vel : dolfinx.fem.function.Function 
                          ,temp : dolfinx.fem.function.Function 
                          ,pres_l : dolfinx.fem.function.Function 
                          ,a_p = None
                          ,it : int = 0
                          ,ts : int = 0
                          ,slab = 1) -> tuple[dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form]:
        
        """Function that set linear form (both for picard iteration and linear problem solution)
        Args: 
            u : dolfinx.fem.function.Function -> Velocity field, used for computing the viscosity
            T : dolfinx.fem.function.Function -> Temperature field, used for computing the viscosity
            PL : dolfinx.fem.function.Function -> Lithostatic pressure field, used for computing the viscosity
            D : Domain -> Domain object, used for extracting the mesh and the boundary conditions
            FR : Functions_material_rheology -> Object containing the rheological functions, used for computing the viscosity
            ctrl : NumericalControls -> Object containing the numerical controls, used for controlling the decoupling of the boundary condition
            sc : Scal -> Object containing the scaling of the problem, used for computing the viscosity
            a_p : dolfinx.fem.Form -> Pressure mass form, used for preconditioning the pressure Schur complement.
            it : int -> Picard iteration number, used for controlling the decoupling of the boundary condition
            ts : int -> Time step number, used for controlling the decoupling of the boundary
        Returns: 
            a1 : dolfinx.fem.Form -> Linear form for the momentum equation
            a2 : dolfinx.fem.Form -> Linear form for the pressure equation (divergence of the test function)
            a3 : dolfinx.fem.Form -> Linear form for the continuity equation (divergence of the trial function)
            L : dolfinx.fem.Form -> Linear form for the right hand side of the momentum equation
            a_p0 : dolfinx.fem.Form -> Linear form for the pressure mass matrix, used for preconditioning the pressure Schur complement.
        """
        from stonedfenicsx.material_property.compute_material_property import cell_average_DG0
        
        u, p  = self.trial0, self.trial1
        v, q  = self.test0,  self.test1
        dx    = ufl.dx

        e = compute_strain_rate(vel)
        # If we are in the first iteration of the first timestep -> use the default viscosity for creating an initial guess. fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))   
        if it == 0 and ts == 0 and slab == 0:
            eta = dolfinx.fem.Constant(self.domain.mesh,PETSc.ScalarType(self.cached_mat.eta_def))
        else: 
            eta = compute_viscosity_FX(e,temp,pres_l,self.pdb,self.cached_mat)
        
        a1 = ufl.inner(2*eta*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
        a2 = - ufl.inner(ufl.div(v), p) * dx             # build once
        a3 = - ufl.inner(q, ufl.div(u)) * dx             # build once
        a_p0 =  -1/eta * ufl.inner( q, p) * dx                      # pressure mass (precond)

        f  = dolfinx.fem.Constant(self.domain.mesh, PETSc.ScalarType((0.0,)*self.domain.mesh.geometry.dim))
        f2 = dolfinx.fem.Constant(self.domain.mesh, PETSc.ScalarType(0.0))
        L  = dolfinx.fem.form([ufl.inner(f, v)*dx, ufl.inner(f2, q)*dx])    
    
        return a1, a2, a3 , L , a_p0       
    
    def compute_moving_wall(self
                        ,facet:str
                        )->None:
        """Compute the moving wall function for the kinematic boundary condition of the slab. 
        The function is computed once at the beginning of the simulation and cached for the entire simulation. 
        The function is computed by solving a simple linear problem with a projection of the velocity on the slab as a source term. 
        The velocity field of the moving wall is then used as a Dirichlet boundary condition for the velocity field on the slab domain.

        Args:
            D (Domain): Domain object, used for extracting the mesh and the boundary conditions
            ctrl (NumericalControls): NumericalControls object, used for controlling the decoupling of the boundary condition
            facet (str): the string that defines the facet on which the moving wall is applied.
        """

        u  = self.trial0
        v  = self.test0


        mesh = self.domain.mesh
        # exact facet normal in    weak   form
        n = ufl.FacetNormal(self.domain.mesh)       
        # slab velocity magnitude (Assuming that velocity of the slab is unit vector)        
        v_slab = float(1.0)  
        
        
        # slab velocity vector (Assuming that velocity of the slab is along x direction)
        v_const = ufl.as_vector((self.ctrl_sim.ctrl_ky.v_s[0], 0.0))
        # projector   onto  the  tangential plane
        proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)
        # tangential     velocity    vector on  slab
        t = ufl.dot(proj, v_const)   
        # tangential   versor            
        t_hat = t / ufl.sqrt(ufl.inner(t, t))    
        # projected tangential velocity vector on   slab
        v_project = v_slab * t_hat  
        # Creating the function space that will host the unit vector of the velocity field along the slab
        # Extract the trial and test function for the subspace of the slab domain

        # Build the linear problem to compute the velocity field of the moving wall. The problem is a simple mass matrix with a projection of the velocity on the slab as a source term.
        a = ufl.inner(u, v) * self.ds(self.domain.bc_dict[facet])        #  boundary     mass    matrix (vector)
        L = ufl.inner(v_project, v) * self.ds(self.domain.bc_dict[facet])
        # Solve the linar problem to compute the velocity field of the moving wall and cache it for the entire simulation 
        problem = dolfinx.fem.petsc.LinearProblem(
            a, L,
            u = self.moving_wall_ref, # Forcing the solution to using the same function space
            petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-20,
            }
        )  # ut_h \in V
        problem.solve()
        self.moving_wall_ref.x.scatter_forward()
        
        return self.moving_wall_ref
    def solve_linear_picard(self
                            ,a:dolfinx.fem.Form
                            ,a_p0:dolfinx.fem.Form
                            ,L:dolfinx.fem.Form
                            ,u:dolfinx.fem.Function
                            ,p:dolfinx.fem.Function
                            ,it:int=0
                            ,ts:int=0
                            ,slab:int = 0):
            
        if it == 0 and ts == 0: 
            self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,self.ctrl_sim.ctrl ,J = None, r = None,it = it, ts = ts, slab=slab)
        else: 
            self.solv.update_block_operator(a,a_p0,self.bc,L,self.F0,self.F1)
        
        u_solved = u.copy()
        p_solved = p.copy()
        x = self.solv.A.createVecLeft()
        self.solv.ksp.solve(self.solv.b, x)

        u_solved.x.array[:self.solv.offset] = x.array[:self.solv.offset]
        p_solved.x.array[: (len(x.array_r) - self.solv.offset)] = x.array[self.solv.offset:]
        u_solved.x.scatter_forward()
        p_solved.x.scatter_forward()
    
        
        if self.ctrl_sim.ctrl.stokes_solver_type == 0: 
            reason = self.solv.ksp.getConvergedReason()
            its    = self.solv.ksp.getIterationNumber()
            rnorm  = self.solv.ksp.getResidualNorm()
            PETSc.Sys.Print(f"                       KSP reason/its/rnormesh: {reason} {its} {rnorm:.3e}")

        minMaxU = min_max_array(u_solved,vel=True)
        print_ph(f'                       min vel = {minMaxU[0]:.5e}, max vel = {minMaxU[1]:.5e}, RMS = {minMaxU[2]:.5e}')
                
        return u_solved,p_solved 
# --- 
class Wedge(Stokes_Problem): 
    def __init__(self
                 ,mesh:Mesh
                 ,elements:list
                 ,name:list
                 ,ctrl_sim:SimulationControls
                 ,pdb:PhaseDataBase):
        """Wedge problem constructor.

        The Wedge class inherits from ``Stokes_Problem`` and therefore
        has all methods and attributes of the parent class.

        This represents a specific Stokes problem defined on a wedge-shaped domain.

        The wedge problem can be:
        - linear
        - non-linear
        - non-linearT

        depending on the rheology of the materials.

        Linear:
            The problem is solved only once for the entire time-dependent solution
            if the velocity is constant in time.

        Non-linear:
            The problem is solved iteratively at each time step, and the viscosity
            is updated at every outer iteration. This activates the internal
            non-linear iterative scheme.

        Non-linearT:
            The problem is solved once per outer iteration to account for the
            feedback from temperature evolution.

        Args:
            M (Mesh): Mesh object containing the mesh and related utilities.
            elements (tuple): Tuple containing the elements of the function spaces.
            name (list): Name of the problem, used to extract the domain from the mesh.
            pdb (PhaseDataBase): Phase database used to determine the rheology type
                and therefore whether the problem is linear or non-linear.
        """

        
        super().__init__(mesh=mesh,elements=elements,name=name,ctrl_sim=ctrl_sim,pdb=pdb)
        

        comm = MPI.COMM_WORLD

        # Example: each rank has some local IDs
        local_ids = np.int32(self.domain.phase.x.array[:])

        # Gather arrays from all processes
        all_ids = comm.allgather(local_ids)

        # Root (or all, since allgather) concatenates
        all_ids = np.concatenate(all_ids)

        # Compute global unique IDs
        unique_ids = np.unique(all_ids)

        non_linear_v = False 
        non_linear_T = False 

        
        for i in range(np.shape(unique_ids)[0]): 
            if pdb.option_eta[unique_ids[i]] > 1: 
                non_linear_v = True
            elif pdb.option_eta[unique_ids[i]] > 0:
                non_linear_T = True
    
            
        if non_linear_v:
            self.typology = 'NonlinearProblem'
        elif non_linear_T and not non_linear_v: 
            self.typology = 'NonlinearProblemT'
        else:
            self.typology = 'LinearProblem'
        
        # Cached boundary condition.     
        self.bc_overriding = None
                
    def setdirichlecht(self
                       ,V : dolfinx.fem.FunctionSpace
                       ,it : int = 0
                       ,ts : int = 0)-> list:
        
        # Extract the mesh from the function
        mesh = V.mesh 
        tdim = mesh.topology.dim
        fdim = tdim - 1
        
        
        if it == 0 and ts == 0:
            # facet ids
            # Extract the dofs from the overriding plate and cache it.  
            noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
            dofs_over = dolfinx.fem.locate_dofs_topological(V, fdim, self.domain.facets.find(self.domain.bc_dict['overriding']))

            self.bc_overriding = dolfinx.fem.dirichletbc(noslip, dofs_over, V)

            #------------------------------------------------------------------------
            # compute the unit-vector components of the moving wall
            self.moving_wall_ref = self.compute_moving_wall('slab') 

            # scale the vector with the decoupling
            if self.ctrl_sim.ctrl.decoupling_ctrl == 1:
                scaling = dolfinx.fem.Function(self.FSPT)
                scaling = decoupling_function(self.FSPT.tabulate_dof_coordinates()[:,1],scaling,self.g_input)  
                scaling.x.array[:] = 1.0 - scaling.x.array[:] 
                scaling.x.scatter_forward()
                # from :https://fenicsproject.discourse.group/t/scale-vector-function-by-scalar-function/10638
                temp_buf = self.moving_wall_ref.copy()
                temp_buf.interpolate(dolfinx.fem.Expression(self.moving_wall_ref*scaling
                                                    ,self.moving_wall_ref.function_space.element.interpolation_points()))
                self.moving_wall_ref = temp_buf.copy()
                
        # update the moving wall normalised field with the actual velocity of the slab.        
        self.moving_wall.x.array[:] = self.moving_wall_ref.x.array[:]*self.ctrl_sim.ctrl_ky.v_s[0]
        print_ph(f'                      Slab velocity is {self.ctrl_sim.ctrl_ky.v_s[0]:.3e} [n.d.]')
        
        self.moving_wall.x.scatter_forward()
        # Set the the boundary condition    
        dofs_s_x = dolfinx.fem.locate_dofs_topological(self.F0.sub(0), fdim, self.domain.facets.find(self.domain.bc_dict['slab']))
        dofs_s_y = dolfinx.fem.locate_dofs_topological(self.F0.sub(1), fdim, self.domain.facets.find(self.domain.bc_dict['slab']))
        # create the dirichlet bc for the slab boundary using the computed velocity field of the moving wall.
        bcx = dolfinx.fem.dirichletbc(self.moving_wall.sub(0), dofs_s_x)
        bcy = dolfinx.fem.dirichletbc(self.moving_wall.sub(1), dofs_s_y)

        return [bcx,bcy, self.bc_overriding]
                
    def solve_the_non_linear(self
                            ,sol: Solution
                            ,it_outer:int=0
                            ,ts:int=0):
        
        time_A = timing.time()
        print_ph('              [//] Picard iterations for the non linear lithostatic pressure problem')    
        u_k = sol.u_wedge.copy()
        u_k1 = sol.u_wedge.copy()
        u_k1.x.array[:]=0.0
        p_k = sol.p_wedge.copy()
        p_k1 = sol.p_wedge.copy() 

        res  = 1.0 
        it_inner   = 0 
        while (res > self.ctrl_sim.ctrl.tol_innerpic) and it_inner < self.ctrl_sim.ctrl.it_inner_max: 
            time_ita = timing.time()
            if it_inner==0: 
                a1,a2,a3,L,a_p = self.set_linear_picard(u_k,
                                                     sol.t_owedge,
                                                     sol.p_lwedge
                                                     ,it = it_outer
                                                     ,ts = ts
                                                     ,slab =0)
            else: 
                a1,_,_,_,a_p = self.set_linear_picard(u_k,
                                                     sol.t_owedge,
                                                     sol.p_lwedge
                                                     ,it = it_outer
                                                     ,ts = ts
                                                     ,slab =0)           
            
            a, a_p0 = self.fem_stokes_form(a1,a2,a3,a_p)

            u_k1, p_k1  = self.solve_linear_picard(dolfinx.fem.form(a)
                                                        ,dolfinx.fem.form(a_p0)
                                                        ,dolfinx.fem.form(L)
                                                        ,u_k
                                                        ,p_k
                                                        ,it_outer
                                                        ,ts)
            
            tol_u = compute_residuum(u_k1,u_k)  
            tol_p = compute_residuum(p_k1,p_k)

            rmom, rdiv = self.compute_residuum_stokes(u_new=u_k1
                                                              ,p_new=p_k1
                                                              ,temp=sol.t_owedge
                                                              ,pres_l=sol.p_lwedge
                                                         )

            if it_inner == 0:
                rmom_0 = rmom
                rdiv_0 = rdiv   


            res   = np.sqrt(tol_u**2+tol_p**2)

            u_k = update_solution(u_k1,u_k,self.ctrl_sim.ctrl.relax)
            p_k =  update_solution(p_k1,p_k,self.ctrl_sim.ctrl.relax)


            u_k.x.scatter_forward()
            p_k.x.scatter_forward()
            
            time_itb = timing.time()    
            print_ph(f'              it[{it_inner}]Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            print_ph(f'                         [x] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
            print_ph(f'                         [x] |F^mom|           {rmom:.3e},         |F^div| {rdiv:.3e}')
            it_inner = it_inner+1 
            print_ph(f'              it[{it_inner}]Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
        print_ph(f'                         [?] |F^mom|L2/|F^mom_0|L2 {rmom/rmom_0:.3e}, |F^div|L2/|F^div_0|L2 {rdiv/rdiv_0:.3e}')
        print_ph(f'                         [?] |F^mom|L2           {rmom:.3e}, abs div residuum |F^div|L2 {rdiv:.3e}')
        print_ph('              []Converged ')  
        sol.u_wedge = u_k.copy()
        sol.p_wedge = p_k.copy() 
        time_B = timing.time()
        print_ph(f'              || -- || --- Solution of Wedge in {time_B-time_A:.2f} sec || -- || --- ||')

        return sol.u_wedge, sol.p_wedge
       
       
   
   
    def Solve_the_Problem(self
                          ,sol : Solution
                          ,it : int =0
                          ,ts : int=0)->Solution:

        """Function that solve the stokes problem for the wedge domain.
            Args:
                S : Solution -> Object containing the solution of the problem, used for storing the solution of the stokes problem
                ctrl : NumericalControls -> Object containing the numerical controls, used for controlling the decoupling of the boundary condition and the type of problem to solve
                FGW : Functions_material_rheology -> Object containing the rheological functions, used for computing the viscosity
                D : Domain -> Domain object, used for extracting the mesh and the boundary conditions
                g : dolfinx.fem.function.Function -> Gravity vector, used for computing the right hand side of the momentum equation
                sc : Scal -> Object containing the scaling of the problem, used for computing the viscosity
                g_input : Geom_input -> Object containing the geometric input, used for computing the decoupling function for the boundary condition
                it : int -> Outer iteration number
                ts : int -> Time step number
            Returns:
                S : Solution -> Object containing the solution of the problem, used for storing the solution of
        
        """     
        if (ts == 0) and (it == 0):
            V_subs0 = self.FS.sub(0)
            p_subs0 = self.FS.sub(1)
            self.V_subs, _ = V_subs0.collapse()
            self.p_subs, _ = p_subs0.collapse()
    
            self.trial0 = ufl.TrialFunction(self.V_subs)
            self.test0 = ufl.TestFunction(self.V_subs)
            self.trial1 = ufl.TrialFunction(self.p_subs)
            self.test1 = ufl.TestFunction(self.p_subs)
            
            # Better to recreate 
            self.moving_wall = dolfinx.fem.Function(self.V_subs)
            self.moving_wall_ref = dolfinx.fem.Function(self.V_subs)
            

            
    

        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(vel=sol.u_wedge
                                                  ,temp=sol.t_owedge
                                                  ,pres_l=sol.p_lwedge
                                                  ,it = it
                                                  ,ts = ts 
                                                  ,slab = 0)
        # Iteration outer 0 -> Initial guess -> start linear
        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(self.V_subs,ts=ts,it=it) 
        
        # Form the system 
        a, a_p0 = self.fem_stokes_form(a1,a2,a3,a_p)

        print_ph('              || -- || Solving the Stokes problem for the wedge domain || -- ||')

        

        
        if self.typology == 'LinearProblem' or self.typology == 'NonlinearProblemT': 
            time_A = timing.time()
            sol.u_wedge,sol.p_wedge = self.solve_linear_picard(dolfinx.fem.form(a),dolfinx.fem.form(a_p0),dolfinx.fem.form(L), sol.u_wedge,sol.p_wedge,it=it,ts=ts)
            time_B = timing.time()
            print_ph(f'              || -- || --- Solution of Wedge in {time_B-time_A:.2f} sec || -- || --- ||')

        else: 
            sol.u_wedge,sol.p_wedge= self.solve_the_non_linear(sol,it_outer=it,ts=ts)

        return sol
    


#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Slab(Stokes_Problem): 
    """Slab problem class. 

    Args:
        Stokes_Problemesh: type of solver (i.e., scalar problem, stokes problem)
    
    Class that inherits from the Stokes problem class, with the specificities of the slab problem. 
    
    The slab domain is solved only once in the following cases: 
    - Steady state thermal solution.
    - Age of the incoming slab is changed.
    If the velocity of the slab is changing with time, slab is solved each timestep. 
 
    The rheology of the slab is linear. The rheology is linear because the internal wall boundary condition
    over-constrains the velocity field.
    
    The class solves the velocity field, in the subducting_plate sub-domain. 
    - The top surface of the slab is an internal wall boundary boundary condition imposed as a Dirichlet boundary condition.
    - The bottom surface of the slab is a free slip boundary condition imposed via Nitsche method.
    - Inflow-Outflow left and bottom boundaries feature a do-nothing bouundary conditions.

    """
    def __init__(self
                 ,mesh:Mesh
                 ,elements:list
                 ,name:list
                 ,ctrl_sim:SimulationControls
                 ,pdb:PhaseDataBase):
        super().__init__(mesh=mesh,elements=elements,name=name,ctrl_sim=ctrl_sim,pdb=pdb)
    
    def setdirichlecht(self
                       ,it : int = 0
                       ,ts:int = 0)-> list:
        """Set Dirichlet boundary condition (Subducting plate domain)

        Args:
            ctrl (NumericalControls): Numerical control structure containing the main information of the simulation
            D (Domain): Subdomain meshes and boundary information
            it (int, optional): iteration of the outer loop. Defaults to 0.
            ts (int, optional): timestep. Defaults to 0.

        Returns:
            list of DirichletBC: List of Dirichlet boundary conditions to be applied on the slab domain. 
            
        [Explanation]: During the first timestep and first outer iteration the component of the unit vector of the velocity along the slab is computed
        redundantly in the entire function space. Then if the velocity of the slab is changing over time, the dirchlecht boundary condition
        is updated by scaling the pre-computed unit vector of the velocity along the slab with the current velocity of the slab.
        """
        mesh = self.F0.mesh 
        
        tdim = mesh.topology.dim
        
        fdim = tdim - 1
        
        # facet ids
        
        
        if (it == 0 or ts == 0 ):
        
            self.moving_wall_ref = self.compute_moving_wall('top_subduction')

        
        # Update the velocity field of the moving wall according to the current velocity of the slab.
        self.moving_wall.x.array[:] = self.moving_wall_ref.x.array[:] * self.ctrl_sim.ctrl_ky.v_s[0] 
        self.moving_wall.x.scatter_forward()
        # locate the dofs on the slab boundary and create dirichlet bc for them.
        dofs_s_x = dolfinx.fem.locate_dofs_topological(self.F0.sub(0), fdim, self.domain.facets.find(self.domain.bc_dict['top_subduction']))
        dofs_s_y = dolfinx.fem.locate_dofs_topological(self.F0.sub(1), fdim, self.domain.facets.find(self.domain.bc_dict['top_subduction']))
        # create the dirichlet bc for the slab boundary using the computed velocity field of the moving wall.
        bcx = dolfinx.fem.dirichletbc(self.moving_wall.sub(0), dofs_s_x)
        bcy = dolfinx.fem.dirichletbc(self.moving_wall.sub(1), dofs_s_y)
        
        return [bcx,bcy]
                
    #-------------------------------------------------------------------
    def compute_nitsche_FS(self
                           ,sol:Solution
                           ,dS:ufl.measure.Measure  
                           ,a1:ufl.form.Form
                           ,a2:ufl.form.Form
                           ,a3:ufl.form.Form
                           ,gamma:float
                           ,it:int = 0)->tuple([ufl.form.Form,ufl.form.Form,ufl.form.Form]):

        """Update the fem form to integrate the weak formulation of free slip boundary condition
        Args:
            D (Domain): Domain object containing the mesh and the boundary information.
            S (Solution): Solution object containing the current solution fields.
            dS (ufl.measure.Measure): Measure for integrating over the boundary facets.
            a1, a2, a3 (ufl.expression): Current forms of the linear system to be updated with the Nitsche terms.
            FGS (Functions_material_properties_global): Object containing the global material properties functions.
            gamma (float): Penalty parameter for the Nitsche method.
            sc (Scaling): Scaling object for non-dimensionalization.
            it (int, optional): Current iteration number for Picard iteration. Defaults to 0.
        Returns:
            tuple: Updated forms (a1, a2, a3) with the Nitsche free slip boundary condition integrated.
        """

        # Compute the shear stress tensor for a given viscosity and velocity field. 
        def tau(eta, u):
            return 2 * eta * ufl.sym(ufl.grad(u))
        
        
        # Linear 
        e   = compute_strain_rate(sol.u_slab)   
        # Viscosity computation
        eta = compute_viscosity_FX(e,sol.t_oslab,sol.p_lslab,self.pdb,self.cached_mat)
        # Extract the facet normal and the cell diameter for the mesh to compute the Nitsche terms.
        n = ufl.FacetNormal(self.domain.mesh)
        h = ufl.CellDiameter(self.domain.mesh)
        # Update the forms with the Nitsche terms.
        a1 += (
            - ufl.inner(tau(eta, self.trial0), ufl.outer(ufl.dot(self.test0, n) * n, n)) * self.ds(dS)
            - ufl.inner(ufl.outer(ufl.dot(self.trial0, n) * n, n), tau(eta, self.test0)) * self.ds(dS)
            + (2 * eta * gamma / h)
            * ufl.inner(ufl.outer(ufl.dot(self.trial0, n) * n, n), ufl.outer(self.test0, n))
            * self.ds(dS)
        )
        if it == 0:
            a2 += ufl.inner(self.trial1, ufl.dot(self.test0, n)) * self.ds(dS)
            a3 += ufl.inner(self.test1, ufl.dot(self.trial0, n)) * self.ds(dS)
        else:
            a2 += 0 
            a3 += 0 
        return a1, a2, a3 
    #-------------------------------------------------------------------
    def Solve_the_Problem(self
                          ,sol:Solution
                          ,it_outer=0
                          ,ts=0):

        
        # Recreate the test and trial function on the subspace of the slab
        # Note: Initially, the problem was not solved because I was trying to reuse
        # The test and trial function created at the beginning of the simulation. 
        # However the composite spaces (V/P blocks) are not allowing to reuse these functions. 
        if ts == 0: 
            V_subs0 = self.FS.sub(0)
            p_subs0 = self.FS.sub(1)
            self.V_subs, _ = V_subs0.collapse()
            self.p_subs, _ = p_subs0.collapse()

            self.trial0 = ufl.TrialFunction(self.V_subs)
            self.test0 = ufl.TestFunction(self.V_subs)
            self.trial1 = ufl.TrialFunction(self.p_subs)
            self.test1 = ufl.TestFunction(self.p_subs)
                        # Better to recreate 
            self.moving_wall = dolfinx.fem.Function(self.V_subs)
            self.moving_wall_ref = dolfinx.fem.Function(self.V_subs)
            
        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(vel = sol.u_slab
                                                  ,temp = sol.t_oslab
                                                  ,pres_l = sol.p_lslab
                                                  ,it = it_outer
                                                  ,ts = ts
                                                  ,slab = 1)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht() 
        # Set Nietsche FS boundary condition 
        dS_bot = self.domain.bc_dict["bot_subduction"]
        # 1 Extract ds 
        a1,a2,a3 = self.compute_nitsche_FS(sol=sol
                                           ,dS=dS_bot
                                           ,a1=a1
                                           ,a2=a2
                                           ,a3=a3
                                           ,gamma=50.0
                                           ,it=it_outer)
                
        a, a_p0 = self.fem_stokes_form(a1,a2,a3,a_p)
        time_A = timing.time()
        sol.u_slab,sol.p_slab =  self.solve_linear_picard(a=dolfinx.fem.form(a)
                                                         ,a_p0=dolfinx.fem.form(a_p0)
                                                         ,L=dolfinx.fem.form(L)
                                                         ,u=sol.u_slab
                                                         ,p=sol.p_slab
                                                         ,it=it_outer
                                                         ,ts = ts
                                                         ,slab=1)

        
        time_B = timing.time()
        print_ph(f'              || -- || --- Solution of Stokes problem in {time_B-time_A:.2f} sec || -- || ---||')

        return sol
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#        