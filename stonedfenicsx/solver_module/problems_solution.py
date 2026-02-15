from stonedfenicsx.package_import import *

from stonedfenicsx.utils import timing_function, print_ph
from stonedfenicsx.material_property.compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX, compute_radiogenic 
from stonedfenicsx.create_mesh.aux_create_mesh import Mesh, Domain, Geom_input
from stonedfenicsx.material_property.phase_db import PhaseDataBase
from stonedfenicsx.numerical_control import NumericalControls, ctrl_LHS, IOControls
from stonedfenicsx.utils import interpolate_from_sub_to_main
from stonedfenicsx.scal import Scal
from stonedfenicsx.output import OUTPUT
from stonedfenicsx.utils import compute_strain_rate
from stonedfenicsx.material_property.compute_material_property import Functions_material_properties_global, Functions_material_rheology
#from stonedfenicsx.solver_module.solver import Solvers, ScalarSolver, SolverStokes
from stonedfenicsx.scal import Scal
from stonedfenicsx.solver_module.solution_routine import *
from stonedfenicsx.solver_module.solver_utilities import *
from stonedfenicsx.solver_module.solver import ScalarSolver, SolverStokes, Solvers


def debug_boundary_condition(bc,name):
    
    if isinstance(bc.g ,dolfinx.fem.Function) or isinstance(bc.g ,dolfinx.cpp.fem.Function_float64):
        dofs = bc.dof_indices()
        dofs_array = dofs[:][0]
        dofs_ghost = dofs[:][1]
        if len(dofs_array) != 0: 
            min_vl = np.nanmin(bc.g.x.array[dofs_array[:dofs_ghost]])
            max_vl = np.nanmax(bc.g.x.array[dofs_array[:dofs_ghost]])
        else: 
            min_vl = -5.01 
            max_vl = -5.01  
    else: 
        min_vl = np.nanmin(bc.g.value)
        max_vl = np.nanmax(bc.g.value)
    
    print_ph(f'{name}')
    if MPI.COMM_WORLD.rank == 0: 
        print(f'Pr0: minV {min_vl:.2f} maxV {max_vl:.2f}')
    if MPI.COMM_WORLD.rank == 1:
        print(f'Pr1: minV {min_vl:.2f} maxV {max_vl:.2f}')
        
    
    

#----------------------------------------------------------------------------     
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
                 ,M: Mesh
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
        if name[1] not in ("domainA", "domainB", "domainC", "domainG"):
            raise NameError("Wrong domain name, check the spelling, in my case was it")
        elif name[1] == "domainC":
            print("Are you sure? DomainC is junk for this problem.")

        M = getattr(M, name[1])

        if len(elements) == 1: 
            self.mixed    = False
            self.FS       = dolfinx.fem.functionspace(M.mesh, elements[0]) 
            self.trial0   = ufl.TrialFunction(self.FS)
            self.test0    = ufl.TestFunction(self.FS)
            self.trial1   = None
            self.test1    = None
        else: 
            self.mixed    = True
            mixed_element = basix.ufl.mixed_element([elements[0], elements[1]])
            self.FS       = dolfinx.fem.functionspace(M.mesh, mixed_element) # MA cristoiddio, perche' cazzo hanno messo FunctionSpace and functionspace come nomi, ma sono degli stronzi?
            # 
            self.F0,_       = self.FS.sub(0).collapse()
            self.F1,_       = self.FS.sub(1).collapse()
            # Define trial/test on mixed FS
            self.trial0   = ufl.TrialFunction(self.FS.sub(0).collapse()[0])
            self.trial1   = ufl.TrialFunction(self.FS.sub(1).collapse()[0])
            self.test0    = ufl.TestFunction(self.FS.sub(0).collapse()[0])
            self.test1    = ufl.TestFunction(self.FS.sub(1).collapse()[0])
        
        self.dx       = ufl.Measure("dx", domain=M.mesh)
        self.ds       = ufl.Measure("ds", domain=M.mesh, subdomain_data=M.facets) # Exterior -> for boundary external 
        self.dS       = ufl.Measure("dS", domain=M.mesh, subdomain_data=M.facets) # Interior -> for boundary integral inside

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
        self.Hs_global : dolfinx.fem.function.Function
        self.T_ad : dolfinx.fem.function.Function
        self.outer_iteration : NDArray[:]
        self.mT : NDArray[:]
        self.MT : NDArray[:]     
        self.ts : NDArray[:]   
        
    def create_function(self
                        ,PG:Problem 
                        ,PS:Problem
                        ,PW:Problem
                        ,elements:list): 
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

        space_GL = fem.functionspace(PG.FS.mesh,mixed_element) # PORCO DIO 
        
        
        def gives_Function(space):
            Va = space.sub(0)
            Pa = space.sub(1)
            V,_  = Va.collapse()
            P,_  = Pa.collapse()
            a = fem.Function(V)
            b = fem.Function(P)
            return a,b 
        
        self.PL       = fem.Function(PG.FS) # Thermal and Pressure problems share the same functional space -> Need to enforce this bullshit 
        self.T_O      = fem.Function(PG.FS) 
        self.T_N      = fem.Function(PG.FS)
        self.Hs_global = fem.Function(PG.FS)
        self.Hs_slab  = fem.Function(PS.FSPT)
        self.Hs_wedge = fem.Function(PW.FSPT)
        self.T_i      = fem.Function(PG.FS)
        self.p_lwedge = fem.Function(PW.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_owedge = fem.Function(PW.FSPT) # same stuff as before, again, this is a nightmare: why the fuck. 
        self.p_lslab = fem.Function(PS.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_oslab = fem.Function(PS.FSPT)
        self.t_nslab = fem.Function(PS.FSPT)
        self.u_global, self.p_global = gives_Function(space_GL)
        self.u_slab  , self.p_slab   = gives_Function(PS.FS)
        self.u_wedge , self.p_wedge  = gives_Function(PW.FS)
        self.T_ad                     = fem.Function(PG.FS)   
        self.mT    = np.zeros(1,dtype=float)
        self.MT    = np.zeros(1,dtype=float) 
        self.outer_iteration = np.zeros(1,dtype=float)
        self.ts             = np.zeros(1,dtype=int)

        return self 


        
#------------------------------------------------------------------- 
#-------------------------------------------------------------------
class Global_thermal(Problem):
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase,ctrl:NumericalControls):
        super().__init__(M,elements,name)
                
        self.steady_state = ctrl.steady_state
        
        if np.all(pdb.option_rho<2) and np.all(pdb.option_k==0) and np.all(pdb.option_Cp==0):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'
        
    def create_bc_temp(self,M:Mesh,ctrl:NumericalControls,geom,lhs,u_global,T_i,it,ts=0):
        from scipy.interpolate import griddata   
        cd_dof = self.FS.tabulate_dof_coordinates()
        # This part can be done only once -> bc dofs are constant 
        if ts == 0 and it == 0:
            facets                 = M.facets.find(M.bc_dict['Top'])    
            dofs_top               = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
            # -> Probably I need to use some parallel shit here 
            self.bc_top            = fem.dirichletbc(ctrl.Ttop, dofs_top, self.FS) 
            facets                 = M.facets.find(M.bc_dict['Left_inlet'])    
            dofs_left              = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
            
            # Create suitable function space for the problem
            T_bc_L = fem.Function(self.FS)
            # Extract z and lhs 
            z   = lhs.z
            LHS = lhs.LHS 
            # Extract coordinate dofs
            # Interpolate temperature field: 
            T_bc_L.x.array[:] = griddata(z, LHS, cd_dof[:,1], method='nearest')
            T_bc_L.x.scatter_forward()

            self.bc_left = fem.dirichletbc(T_bc_L, dofs_left)
            

            facets                 = M.facets.find(M.bc_dict['Right_lit'])                        
            dofs_right_lit        = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
    
            T_gr = (-geom.lab_d-0)/(ctrl.Tmax-ctrl.Ttop)
            T_gr = T_gr**(-1) 
        
            bc_fun = fem.Function(self.FS)
            bc_fun.x.array[dofs_right_lit] = ctrl.Ttop + T_gr * cd_dof[dofs_right_lit,1]
            bc_fun.x.scatter_forward()
        
            self.bc_right_lit = fem.dirichletbc(bc_fun, dofs_right_lit)

        facets                 = M.facets.find(M.bc_dict['Right_wed'])                        
        dofs_right_wed        = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
        h_vel  = u_global.sub(0) # index 1 = y-direction (2D)
        vel_T  = fem.Function(self.FS)
        vel_T.interpolate(h_vel)
        vel_T.x.scatter_forward()
        vel_bc = vel_T.x.array[dofs_right_wed]
        ind_z = np.where((vel_bc < 0.0) & (cd_dof[dofs_right_wed,1]<=-geom.lab_d))
        dofs_vel = dofs_right_wed[ind_z[0]]        
        if ctrl.adiabatic_heating==0:
            self.bc_right_wed = fem.dirichletbc(ctrl.Tmax, dofs_vel,self.FS)
        else: 
            function_bc = fem.Function(self.FS)
            function_bc.interpolate(T_i)
            self.bc_right_wed  = fem.dirichletbc(function_bc,dofs_vel)
            
        facets                 = M.facets.find(M.bc_dict['Bottom_wed'])                        
        dofs_bot_wed        = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
        h_vel = u_global.sub(0) # index 1 = y-direction (2D)   
        v_vel = u_global.sub(1) # index 1 = y-direction (2D)
        vel_T = fem.Function(self.FS)
        vel_T.interpolate(v_vel)
        vel_T.x.scatter_forward()
        vel_bc = vel_T.x.array[dofs_bot_wed]
        ind_z = np.where((vel_bc > 0.0))
        dofs_vel = dofs_bot_wed[ind_z[0]]    
                    
        
        self.bc_bot_wed = fem.dirichletbc(ctrl.Tmax, dofs_vel,self.FS)
        
        bc = [ self.bc_left,  self.bc_right_wed,self.bc_bot_wed, self.bc_right_lit,self.bc_top]

        
        DEBUG = 0
        if DEBUG == 1: 
            print_ph('DEBUGGING...')
            debug_boundary_condition(self.bc_top,'top')
            MPI.COMM_WORLD.barrier()
            debug_boundary_condition(self.bc_left,'left')
            MPI.COMM_WORLD.barrier()
            debug_boundary_condition(self.bc_bot_wed,'bot_wedge')
            MPI.COMM_WORLD.barrier()

            debug_boundary_condition(self.bc_right_lit,'right_lit')
        
        
        
        return bc 
        
    #------------------------------------------------------------------
    def compute_shear_heating(self,ctrl,pdb,S,D,g_input,sc):
        """
        Apperently the sociopath the devise this method, uses a delta function to describe 
        the interface frictional heating. 
        -> [A] => Shear heating becomes a ufl expression. So happy about it 
        
        """

        if ctrl.decoupling == 1 and ctrl.model_shear >0: 
            facets1                = D.facets.find(D.bc_dict['Subduction_top_lit'])
            facets2                = D.facets.find(D.bc_dict['Subduction_top_wed'])

            facet_seismogenic = np.unique(np.concatenate((facets1,facets2)))

            dofs              = fem.locate_dofs_topological(self.FS, D.mesh.topology.dim-1, facet_seismogenic)

            heat_source = fem.Function(self.FS)
            heat_source.x.array[:] = 0.0   
            heat_source.x.scatter_forward()
        
            decoupling    = heat_source.copy()
            Z = self.FS.tabulate_dof_coordinates()[:,1]
            decoupling = decoupling_function(Z,decoupling,g_input)

            if ctrl.model_shear==2:
                # compute the plastic strain rate ratio and viscous shear heating strain rate 
                # Place holder function
                expression = self.compute_friction_shear_expression(pdb,ctrl,D,S.T_O,S.PL,ctrl.v_s[0],decoupling,sc,dofs) * ufl.avg(self.test0) * (self.dS(D.bc_dict['Subduction_top_lit']) +self.dS(D.bc_dict['Subduction_top_wed']))

            else:  
                phi = np.tan(pdb.friction_angle)
                expression = decoupling * ufl.avg(S.PL) * ctrl.v_s[0] * phi * ufl.avg(self.test0) * (self.dS(D.bc_dict['Subduction_top_lit']) +self.dS(D.bc_dict['Subduction_top_wed']))

            return expression

        else:
            return 0.0 
        
        
    def compute_friction_shear_expression(self,pdb,ctrl,D,T,P,vs,decoupling,sc,dofs):

        from .compute_material_property import compute_plastic_strain

        e_II_fr = (vs * decoupling * 1 /ctrl.wz_tk)/2  # Second invariant strain rate

        # -> compute the plastic strain rate

        e_pl, tau = compute_plastic_strain(e_II_fr,T,P,pdb,D.phase,ctrl.phase_wz-1,sc)

        e_vs = 1 - e_pl 

        phi = ufl.tan(pdb.friction_angle)

        friction = (e_pl * vs * decoupling * phi * P + e_vs * e_II_fr * tau * ctrl.wz_tk) 

        return friction 
            
    
    def set_newton_SS(self,p,D,T,u_global,pdb):
        trial0 = self.trial0                # δp (TrialFunction) for Jacobian


        rho_k = density_FX(pdb, T, p, D.phase, D.mesh)  # frozen
        
        k_k = density_FX(pdb, T, p, D.phase, D.mesh)  # frozen
        
        Cp_k = heat_capacity_FX(pdb, T,  D.phase, D.mesh)  # frozen


        f    = fem.Constant(D.mesh, 0.0)  # source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
            
        diff = ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        adv  = rho_k *Cp_k *ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx

        a_lin = diff + adv
        L     = f * self.test0 * dx   
        # Nonlinear residual: F(p; v) = ∫ ∇p·∇v dx - ∫ ∇v·(ρ(T, p) g) dx
        F = a_lin - L
        # Jacobian dF/dp in direction δp (trial0)
        J = ufl.derivative(F, T, trial0)
        
        return F,J 
    
    #------------------------------------------------------------------
    def compute_energy_source(self,D,FG):
        source = fem.Function(self.FS)
        source = compute_radiogenic(FG, source)
        self.energy_source = source.copy()

    #------------------------------------------------------------------
    def compute_adiabatic_heating(self,D,FG,u,T,p,ctrl):
        from .compute_material_property import alpha_FX
        
        if ctrl.adiabatic_heating != 0: 
            
        
            alpha = alpha_FX(FG,T,p)
            adiabatic_heating = alpha * T * ufl.inner(ufl.grad(p), u) 
        else: 
            adiabatic_heating = (0.0)
        
        
        self.adiabatic_heating = adiabatic_heating
        

    #------------------------------------------------------------------
    def compute_residual_SS(self,p_k,T,u_global,D,FG, ctrl):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(FG, T, p_k)  # frozen
        
        Cp_k = heat_capacity_FX(FG, T)  # frozen

        k_k = heat_conductivity_FX(FG, T, p_k, Cp_k, rho_k)  # frozen

        self.compute_adiabatic_heating(D,FG,u_global,T,p_k,ctrl)

        f    = self.energy_source# source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
            
        diff = ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        
        adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx
            
        L = fem.form((f + self.adiabatic_heating) * self.test0 * dx + self.shear_heating )      
        
        R = fem.form(diff + adv - L)
                

        return R
    
    
    def set_linear_picard_SS(self,
                             p_k:dolfinx.fem.Function=None,
                             T:dolfinx.fem.Function = None,
                             T_O:dolfinx.fem.Function=None,
                             u_global:dolfinx.fem.Function=None,
                             D:Domain =None,
                             FG:Functions_material_properties_global=None,
                             ctrl:NumericalControls=None,
                             dt:float = None,
                             it:int=0)->tuple[dolfinx.fem.Form,dolfinx.fem.Form]:
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(FG, T, p_k)  # frozen
        
        Cp_k = heat_capacity_FX(FG, T)  # frozen

        k_k = heat_conductivity_FX(FG, T, p_k, Cp_k, rho_k)  # frozen


        f    = self.energy_source# source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
        if it == 0: 
            
            diff = ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
            
            adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
            
            a = fem.form(diff + adv)
            
            L = fem.form((f) * self.test0 * dx + self.shear_heating )      
        
        else: 
        
            a = fem.Form(None)
                

        return a, L
    #------------------------------------------------------------------

    def set_linear_picard_TD(self,
                             p_k:dolfinx.fem.Function=None,
                             T:dolfinx.fem.Function = None,
                             T_O:dolfinx.fem.Function=None,
                             u_global:dolfinx.fem.Function=None,
                             D:Domain =None,
                             FG:Functions_material_properties_global=None,
                             ctrl:NumericalControls=None,
                             dt:float = None,
                             it:int=0):
        # Function that set linear form and linear picard for picard iteration
        # Crank Nicolson scheme 
        # a - > New temperature 
        # L - > Old temperature
        # -> Source term is assumed constant in time and do not vary between the timesteps 
        
        rho_k = density_FX(FG, T, p_k)  # frozen
                
        Cp_k = heat_capacity_FX(FG, T)  # frozen

        k_k = heat_conductivity_FX(FG, T, p_k, Cp_k, rho_k)  # frozen


        
        rho_k0 = density_FX(FG, T_O, p_k)  # frozen
                
        Cp_k0 = heat_capacity_FX(FG, T_O)  # frozen
        
        k_k0 = heat_conductivity_FX(FG, T_O, p_k, Cp_k, rho_k)  # frozen


                
        rhocp        =  (rho_k * Cp_k)

        rhocp_old    =  (rho_k0 * Cp_k0)
        
        dx  = self.dx
        
        f    = (self.energy_source) * self.test0 * dx + self.shear_heating # source term {energy_source is radiogenic heating compute before hand, shear heating is frictional heating already a form}

        
        # a -> New temperature 
        diff_new = ( 1 / 2 ) * ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
        
        adv_new  = (rhocp / 2 )* ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
        
        mass_new = (rhocp / dt) * self.trial0 * self.test0 * dx
        
        a = fem.form(diff_new + adv_new + mass_new)
                
        if it == 0: 
            
            adv_old =  - (rhocp_old / 2 ) * ufl.dot(u_global, ufl.grad(T_O)) * self.test0 * dx

            diff_old =  - ( 1 / 2 ) * ufl.inner(k_k0 * ufl.grad(T_O), ufl.grad(self.test0)) * dx
            
            mass_old =  (rhocp_old / dt) * T_O * self.test0 * dx
            
            L = fem.form(diff_old + adv_old + f + mass_old)
            
            return a, L

        else: 
            return a, fem.Form(None)
    #------------------------------------------------------------------
    def Solve_the_Problem(self,S,ctrl,FG,M,lhs,geom,sc,it=0,ts=0): 
        
        nl = 0 
        dt = None
        # choose the problem: 
        if ctrl.steady_state == 1: 
            self.set_linear = self.set_linear_picard_SS 
        else: 
            self.set_linear = self.set_linear_picard_TD
            dt = ctrl.dt
        
        
        p_k = S.PL.copy()  # Previous lithostatic pressure 
        T   = S.T_N.copy() # -> will not eventually update 
        
            
        if it == 0:         
            self.shear_heating = self.compute_shear_heating(ctrl,FG, S,getattr(M,'domainG'),geom,sc)
            self.compute_energy_source(getattr(M,'domainG'),FG)
        
        a,L = self.set_linear(p_k
                              ,T
                              ,S.T_O
                              ,S.u_global
                              ,getattr(M,'domainG')
                              ,FG
                              ,ctrl
                              ,dt)
        
        self.bc = self.create_bc_temp(getattr(M,'domainG'),ctrl,geom,lhs,S.u_global,S.T_i,it)

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl)
        
        print_ph('              // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if nl == 0: 
            S.T_N = self.solve_the_linear(S,a,L,S.T_N) 
            S.T_N.x.scatter_forward()

        else: 
            S = self.solve_the_non_linear(M,S,ctrl,pdb)
        
        time_B = timing.time()
        
        print_ph(f'              // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return S 

    #------------------------------------------------------------------
    
    def solve_the_linear(self,S,a,L,fen_function,isPicard=0,it=0,ts=0):
        

        self.solv.A.zeroEntries()
        fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc)
        self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        self.solv.b.set(0.0)
        fem.petsc.assemble_vector(self.solv.b, fem.form(L))
        fem.petsc.apply_lifting(self.solv.b, [fem.form(a)], [self.bc])
        
        self.solv.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        
        fem.petsc.set_bc(self.solv.b, self.bc)
        self.solv.ksp.solve(self.solv.b, fen_function.x.petsc_vec)
        fem.petsc.set_bc(fen_function.x.petsc_vec,bcs=self.bc)
        
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            fen_function.x.scatter_forward()

            return fen_function 
        else:
            fen_function.x.scatter_forward()
            return fen_function


    def solve_the_non_linear(self
                            ,M
                            ,S
                            ,Hs
                            ,ctrl
                            ,FGT
                            ,it=0):  
        
        isPicard = 1 # Flag for the linear solver. 
        tol = 1.0 
        T_O = S.T_O
        T_k = S.T_N.copy() 
        T_k1 = S.T_N.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        time_A = timing.time()
        print_ph('              [//] Picard iterations for the non linear temperature problem')
        omega = ctrl.relax
        while it_inner < max_it and tol > ctrl.tol:
            time_ita = timing.time()
            
            if it_inner == 0: 
                A,L = self.set_linear_picard(S.PL
                                            ,T_k
                                            ,S.u_global
                                            ,Hs,getattr(M,'domainG')
                                            ,pdb
                                            ,ctrl)
            else: 
                if ctrl.steady_state==1: 
                    _,L = self.set_linear_picard(S.PL
                                                ,T_k
                                                ,S.u_global
                                                ,Hs,getattr(M,'domainG')
                                                ,pdb
                                                ,ctrl
                                                ,it=it_inner)
                else: 
                    A,_ = self.set_linear_picard(S.PL
                             ,T_k
                             ,S.u_global
                             ,Hs,getattr(M,'domainG')
                             ,pdb
                             ,ctrl
                             ,it_inner)
                    
            T_k1 = self.solve_the_linear(S,A,L,T_k1,1,it,1)
            T_k1.x.scatter_forward()
            # L2 norm 
            tol = compute_residuum(T_k1,T_k)

            
            time_itb = timing.time()
            print_ph(f'              []Temperature L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            x  = T_k.x.petsc_vec
            x1 = T_k1.x.petsc_vec

            # x = (1-omega)*x + omega*x1
            x.scale(1.0 - omega)
            x.axpy(omega, x1)

            T_k.x.scatter_forward()
            #update solution
            #T_k.x.array[:] = T_k1.x.array[:]*0.7 + T_k.x.array[:]*(1-0.7)
            #T_k.x.scatter_forward()
            it_inner = it_inner + 1 
        S.T_N.x.array[:] = T_k1.x.array[:]
        S.T_N.scatter_forward()
        print_ph(f'')
        
        return S  
        
    @timing_function
    def initial_temperature_field(self,M, ctrl, lhs, g_input):
        from scipy.interpolate import griddata
        from ufl import conditional, Or, eq
        from functools import reduce
        """
        X    -:- Functionspace (i.e., an abstract stuff that represents all the possible solution for the given mesh and element type)
        M    -:- Mesh object (i.e., a random container of utils related to the mesh)
        ctrl -:- Control structure containing the information of the simulations 
        lhs  -:- left side boundary condition controls. Separated from the control structure for avoiding clutter in the main ctrl  
        ---- 
        Function: Create a function out of the function space (T_i). From the function extract dofs, interpolate (initial) lhs all over. 
        Then select the crustal+lithospheric marker, and overwrite the T_i with a linear geotherm. Simple. 
        ----
        output : T_i the initial temperature field.  
            T_gr = (-M.g_input.ns_depth-0)/(ctrl.Tmax-ctrl.Ttop)
            T_gr = T_gr**(-1) 

            bc_fun = fem.Function(X)
            bc_fun.x.array[dofs_dirichlet] = ctrl.Ttop + T_gr * cd_dof[dofs_dirichlet,1]
            bc_fun.x.scatter_forward()
        """    
        #- Create part of the thermal field: create function, extract dofs, 
        X     = self.FS
        T_i_A = fem.Function(X)
        cd_dof = X.tabulate_dof_coordinates()
        T_i_A.x.array[:] = griddata(lhs.z, lhs.LHS, cd_dof[:,1], method='nearest')
        T_i_A.x.scatter_forward() 
        #- 
        T_gr = (-g_input.lab_d-0)/(ctrl.Tmax-ctrl.Ttop)
        T_gr = T_gr**(-1) 
        T_expr = fem.Function(X)
        ind_A = np.where(cd_dof[:,1] >= -g_input.lab_d)[0]
        ind_B = np.where(cd_dof[:,1] < -g_input.lab_d)[0]
        T_expr.x.array[ind_A] = ctrl.Ttop + T_gr * cd_dof[ind_A,1]
        T_expr.x.array[ind_B] = ctrl.Tmax
        T_expr.x.scatter_forward()
        T_i = fem.Function(X)
        expr = conditional(
            reduce(Or,[eq(M.phase, i) for i in [2, 3, 4, 5]]),
            T_expr,
            T_i_A
        )
        T_i.interpolate(fem.Expression(expr, X.element.interpolation_points()))
        T_i.x.array[ind_B] = ctrl.Tmax
        T_i.x.scatter_forward()
        return T_i 

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Global_pressure(Problem): 
    def __init__(self
                 ,M:Mesh
                 ,elements:tuple
                 ,name:list
                 ,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        
        if np.all(pdb.option_rho<2):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'

        self.bc = [self.set_problem_bc(getattr(M,'domainG'))]
    
    def set_problem_bc(self
                       ,D:Domain)->list[fem.DirichletBC]:
         
        top_facets   = D.facets.find(D.bc_dict['Top'])
        top_dofs    = fem.locate_dofs_topological(self.FS, 1, top_facets)
        bc = [fem.dirichletbc(0.0, top_dofs, self.FS)]
        
        return bc  
    
    def set_newton(self,p,D,T,g,pdb):
        test   = self.test0                 # v
        trial0 = self.trial0                # δp (TrialFunction) for Jacobian

        a_lin = ufl.inner(ufl.grad(p), ufl.grad(test)) * self.dx
        L     = ufl.inner(ufl.grad(test),
                              density_FX(pdb, T, p) * g) * self.dx
        # Nonlinear residual: F(p; v) = ∫ ∇p·∇v dx - ∫ ∇v·(ρ(T, p) g) dx
        F = a_lin - L
        # Jacobian dF/dp in direction δp (trial0)
        J = ufl.derivative(F, p, trial0)
        
        return F,J 
    
    
    def set_linear_picard(self,p_k,T,D,FG,g, it=0):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(FG, T, p_k)  # frozen
        
        # Linear operator with frozen coefficients
        if it == 0: 
            a = ufl.inner(ufl.grad(self.trial0), ufl.grad(self.test0)) * self.dx
        else: 
            a = None
        
        L = ufl.inner(ufl.grad(self.test0), rho_k * g) * self.dx
        

        return a, L

    def Solve_the_Problem(self,S,ctrl,pdb,M,g,it_outer=0,ts=0): 
        
        nl = 0 
        p_k = S.PL.copy()  # Previous lithostatic pressure 
        T   = S.T_N.copy() # -> will not eventually update 
        
        # If the problem is linear, p_k is not doing anything, it is there because I
        # design the density function to receive in anycase a pressure, potentially I
        # can use the multipledispach of python, which say ah density with pressure 
        # density without pressure is equal to fuck. But seems a bit lame, and I do 
        # not think that is a great improvement of the code. 
        
        a,L = self.set_linear_picard(p_k,T,getattr(M,'domainG'),pdb,g)
        
        if self.typology == 'NonlinearProblem':
            F,J = self.set_newton(p_k,getattr(M,'domainG'),T,g,pdb)
            nl = 1 
        else: 
            F = None;J=None 

        if it_outer == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        
        print_ph('              // -- // --- LITHOSTATIC PROBLEM [GLOBAL] // -- // --- > ')

        time_A = timing.time()

        
        if nl == 0: 
            S = self.solve_the_linear(S,a,L,S.PL) 
        else: 
            S = self.solve_the_non_linear(M,S,ctrl,pdb,g)

        time_B = timing.time()

        print_ph('              // -- // --- Solution of Lithostatic pressure problem finished in {time_B-time_A:.2f} sec // -- // --->')
        print_ph('')

        return S 
    
    def solve_the_linear(self,S,a,L,function_fen,isPicard=0,it=0,ts=0):
        
        if it == 0 or ts == 0:
            self.solv.A.zeroEntries()
            fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc[0])
            self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        self.solv.b.set(0.0)
        fem.petsc.assemble_vector(self.solv.b, fem.form(L))
        fem.petsc.apply_lifting(self.solv.b, [fem.form(a)], self.bc)
        self.solv.b.ghostUpdate()
        fem.petsc.set_bc(self.solv.b, self.bc[0])
        self.solv.ksp.solve(self.solv.b, function_fen.x.petsc_vec)
        function_fen.x.scatter_forward()
        
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            S.PL = function_fen 
            return S
        else:
            return function_fen
    
    def solve_the_non_linear(self,M,S,ctrl,FG,g):  
        
        tol = 1e-3  # Tolerance Picard  
        tol = 1.0 
        p_k = S.PL.copy() 
        p_k1 = S.PL.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        
        
        print_ph('              [//] Picard iterations for the non linear lithostatic pressure problem')

        omega = ctrl.relax
        
        while it_inner < ctrl.it_max and tol > ctrl.tol_innerPic:
            time_ita = timing.time()
            
            if it_inner == 0:
                A,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),FG,g)
            else: 
                _,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),FG,g,1)
            
            p_k1 = self.solve_the_linear(S,A,L,p_k1,1,it_inner,1) 

            # L2 norm 
            tol = compute_residuum(p_k1,p_k)
            
            time_itb = timing.time()
            print_ph(f'              []L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            p_k.x.array[:] = p_k1.x.array[:]*omega + p_k.x.array[:]*(1-omega)
            p_k.x.scatter_forward()
            
            it_inner = it_inner + 1 
        
        print_ph('              [//] Newton iterations for the non linear lithostatic pressure problem')

        # --- Newton =>         
        
        S.PL.x.array[:] = p_k.x.array[:]
      
        
        
        return S  

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Stokes_Problem(Problem):
    def __init__(self,M,elements,name):
        super().__init__(M,elements,name)       
        M = getattr(M,name[1])
        self.FSPT = dolfinx.fem.functionspace(M.mesh, elements[2]) 
    
    def compute_shear_heating(self,ctrl,FR,S,D,sc,wedge=1):
        from .compute_material_property import compute_viscosity_FX
        from .utils import evaluate_material_property
        if wedge ==1: 
            V = S.u_wedge.function_space
            PT = self.FSPT
            e = compute_strain_rate(S.u_wedge)
            eta_new = compute_viscosity_FX(e, S.t_owedge, S.p_lwedge, FR, sc)
        else: 
            V = S.u_slab.function_space
            PT = self.FSPT
            e = compute_strain_rate(S.u_slab)
            eta_new = compute_viscosity_FX(e, S.t_oslab, S.p_lslab, FR, sc)

        shear_heating = ufl.inner(2*eta_new*e, e)

        if wedge ==1: 
            S.Hs_wedge = evaluate_material_property(shear_heating,PT)
        else: 
            S.Hs_slab = evaluate_material_property(shear_heating,PT)

        return S

    def fem_stokes_form(self,a1,a2,a3,a_p):
        a   = fem.form([[a1, a2],[a3, None]])
        a_p0  = fem.form([[a1, a2],[a3, a_p]])
        return a,a_p0
    
    def compute_residuum_stokes(self, u_new, p_new, D, T, PL, FR, sc):
        V = u_new.function_space
        Q = p_new.function_space
        v = ufl.TestFunction(V)
        q = ufl.TestFunction(Q)

        e = compute_strain_rate(u_new)
        eta_new = compute_viscosity_FX(e, T, PL, FR,sc)
        f = fem.Constant(D.mesh, PETSc.ScalarType((0.0,) * D.mesh.geometry.dim))
        dx = ufl.dx

        Fmom = (ufl.inner(2*eta_new*ufl.sym(ufl.grad(u_new)), ufl.sym(ufl.grad(v))) * dx
                - ufl.inner(p_new, ufl.div(v)) * dx
                - ufl.inner(f, v) * dx)

        Fdiv = ufl.inner(q, ufl.div(u_new)) * dx

        Rm = fem.petsc.assemble_vector(fem.form(Fmom))
        Rm.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if getattr(self, "bc", None):
            with Rm.localForm() as lf:
                r = lf.getArray(readonly=False)
                for bc in self.bc:
                    dofs = bc.dof_indices()[0]   # local dof indices
                    r[dofs] = 0.0

        rmom = Rm.norm()

        # divergence residual: I'd rely on divuL2 instead
        divuL2 = (fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_new), ufl.div(u_new))*dx)))**0.5

        # optional: keep your Q-vector residual if you really want it
        Rd = fem.petsc.assemble_vector(fem.form(Fdiv))
        Rd.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        rdiv = Rd.norm()

        return rmom, rdiv, divuL2


    def set_linear_picard(self
                          ,u_s : dolfinx.fem.function.Function 
                          ,T : dolfinx.fem.function.Function 
                          ,PL : dolfinx.fem.function.Function 
                          ,D : Domain
                          ,FR : Functions_material_rheology
                          ,ctrl : NumericalControls
                          ,sc : Scal
                          ,a_p = None
                          ,it : int = 0
                          ,ts : int = 0) -> tuple[dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form]:
        
        """Function that set linear form (both for picard iteration and linear problem solution)
        Args: 
            u_slab : dolfinx.fem.function.Function -> Velocity field of the slab, used for computing the viscosity
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
        
        u, p  = self.trial0, self.trial1
        v, q  = self.test0,  self.test1
        dx    = ufl.dx

        e = compute_strain_rate(u_s)

        eta = compute_viscosity_FX(e,T,PL,FR,sc)

        a1 = ufl.inner(2*eta*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
        a2 = - ufl.inner(ufl.div(v), p) * dx             # build once
        a3 = - ufl.inner(q, ufl.div(u)) * dx             # build once
        a_p0 = - 1/eta * ufl.inner( q, p) * dx                      # pressure mass (precond)

        f  = fem.Constant(D.mesh, PETSc.ScalarType((0.0,)*D.mesh.geometry.dim))
        f2 = fem.Constant(D.mesh, PETSc.ScalarType(0.0))
        L  = fem.form([ufl.inner(f, v)*dx, ufl.inner(f2, q)*dx])    
    
        return a1, a2, a3 , L , a_p0       
    
    def compute_moving_wall(self
                        ,D:Domain
                        ,ctrl:NumericalControls
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

        u, p  = self.trial0, self.trial1
        v, q  = self.test0,  self.test1


        mesh = D.mesh
        # exact facet normal in    weak   form
        n = ufl.FacetNormal(D.mesh)       
        # slab velocity magnitude (Assuming that velocity of the slab is unit vector)        
        v_slab = float(1.0)  
        # slab velocity vector (Assuming that velocity of the slab is along x direction)
        v_const = ufl.as_vector((ctrl.v_s[0], 0.0))
        # projector   onto  the  tangential plane
        proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)
        # tangential     velocity    vector on  slab
        t = ufl.dot(proj, v_const)   
        # tangential   versor            
        t_hat = t / ufl.sqrt(ufl.inner(t, t))    
        # projected tangential velocity vector on   slab
        v_project = v_slab * t_hat  
        # Creating the function space that will host the unit vector of the velocity field along the slab
        self.moving_wall = fem.Function(self.FS)
        # Extract the trial and test function for the subspace of the slab domain
        w = u
        v = v
        # Build the linear problem to compute the velocity field of the moving wall. The problem is a simple mass matrix with a projection of the velocity on the slab as a source term.
        a = ufl.inner(w, v) * self.ds(D.bc_dict[facet])        #  boundary     mass    matrix (vector)
        L = ufl.inner(v_project, v) * self.ds(D.bc_dict[facet])
        # Solve the linar problem to compute the velocity field of the moving wall and cache it for the entire simulation 
        self.moving_wall = fem.petsc.LinearProblem(
            a, L,
            petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-20,
            }
        ).solve()  # ut_h \in V
        self.moving_wall.x.scatter_forward()
        
        return self 
    def solve_linear_picard(self
                            ,a
                            ,a_p0
                            ,L
                            ,ctrl
                            ,u
                            ,p
                            ,it=0
                            ,ts=0):
            
        if it == 0 and ts == 0: 
            self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,ctrl ,J = None, r = None,it = it, ts = ts)
        else: 
            self.solv.update_block_operator(a,a_p0,self.bc,L,self.F0,self.F1)
        
        
        x = self.solv.A.createVecLeft()
        self.solv.ksp.solve(self.solv.b, x)

        u.x.array[:self.solv.offset] = x.array[:self.solv.offset]
        p.x.array[: (len(x.array_r) - self.solv.offset)] = x.array[self.solv.offset:]
        u.x.scatter_forward()
        p.x.scatter_forward()
        
        self.r = self.solv.b.duplicate()
        self.solv.A.mult(x, self.r)          # r = A x
        self.r.scale(-1.0)         # r = -A x
        self.r.axpy(1.0, self.solv.b)        # r = b - A x
        self.r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        abs_res = self.r.norm()
        
        return u,p,abs_res 


#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Wedge(Stokes_Problem): 
    def __init__(self
                 ,M:Domain
                 ,elements:tuple
                 ,name:list
                 ,pdb:PhaseDataBase)->None:
        
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

        
        super().__init__(M,elements,name)
        

        comm = MPI.COMM_WORLD

        # Example: each rank has some local IDs
        local_ids = np.int32(M.domainB.phase.x.array[:])

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
    
            
        if non_linear_v == True:
            self.typology = 'NonlinearProblem'
        elif non_linear_T == True and non_linear_v==False: 
            self.typology = 'NonlinearProblemT'
        else:
            self.typology = 'LinearProblem'
        
        # Cached boundary condition.     
        self.bc_overriding = None
        
        self.bc_moving_wall = dolfinx.fem.function.Function(self.FS)
        
    #------------------------------------------------------------------
    def setdirichlecht(self
                       ,ctrl : NumericalControls
                       ,D : Domain 
                       ,V : dolfinx.fem.FunctionSpace
                       ,g_input : Geom_input
                       ,it : int = 0
                       ,ts : int = 0)-> list:
        
        # Extract the mesh from the function
        mesh = V.mesh 
        tdim = mesh.topology.dim
        fdim = tdim - 1
        
        if it == 0 and ts == 0:
            # facet ids
            slab_facets      = D.facets.find(D.bc_dict['slab'])
            # Extract the dofs from the overriding plate and cache it.  
            noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
            dofs_over = fem.locate_dofs_topological(V, fdim, D.facets.find(D.bc_dict['overriding']))

            self.bc_overriding = fem.dirichletbc(noslip, dofs_over, V)

            #------------------------------------------------------------------------
            # compute the velocity field of the moving wall and cache it for the entire simulation
            self.compute_moving_wall(D,ctrl,'slab') 

            # Correct the moving wall for the decoupling if needed.
            if ctrl.decoupling == 1:
                scaling = fem.Function(self.FSPT)
                scaling = decoupling_function(self.FSPT.tabulate_dof_coordinates()[:,1],scaling,g_input)  
                scaling.x.array[:] = 1.0 - scaling.x.array[:] 
                # from :https://fenicsproject.discourse.group/t/scale-vector-function-by-scalar-function/10638
                temp_buf = self.moving_wall.copy()
                temp_buf.interpolate(fem.Expression(self.moving_wall*scaling
                                                    ,self.moving_wall.function_space.element.interpolation_points()))
                self.moving_wall = temp_buf.copy()
                
        # update the moving wall normalised field with the actual velocity of the slab.        
        self.moving_wall.x.array[:] = self.moving_wall.x.array[:]*ctrl.v_s[0]
        self.moving_wall.x.scatter_forward()
            
        dofs_s_x = fem.locate_dofs_topological(V.sub(0), fdim,slab_facets)
        dofs_s_y = fem.locate_dofs_topological(V.sub(1), fdim,slab_facets)

        bcx = fem.dirichletbc(self.moving_wall.sub(0), dofs_s_x)
        bcy = fem.dirichletbc(self.moving_wall.sub(1), dofs_s_y)
        

        return [bcx, bcy, self.bc_overriding]
                
   
    def Solve_the_Problem(self
                          ,S : Solution
                          ,ctrl : NumericalControls
                          ,FGW : Functions_material_rheology
                          ,D : Domain 
                          ,g : dolfinx.fem.function.Function
                          ,sc : Scal
                          ,g_input : Geom_input
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
    

        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_wedge
                                                  ,S.t_owedge
                                                  ,S.p_lwedge
                                                  ,D
                                                  ,FGW
                                                  ,ctrl
                                                  ,sc)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl,D,self.V_subs,g_input) 
        
        # Form the system 
        a, a_p0 = self.fem_stokes_form(a1,a2,a3,a_p)

        print_ph('              // -- // --- STOKES PROBLEM [WEDGE] // -- // --- > ')

        
        time_A = timing.time()

        
        if self.typology == 'LinearProblem' or self.typology == 'NonlinearProblemT': 
            S.u_wedge,S.p_wedge,r_al = self.solve_linear_picard(fem.form(a),fem.form(a_p0),fem.form(L),ctrl, S.u_wedge,S.p_wedge,it=it,ts=ts)

        else: 
            print_ph('              [//] Picard iterations for the non linear lithostatic pressure problem')

            u_k = S.u_wedge.copy()
            u_k1 = S.u_wedge.copy()
            p_k = S.p_wedge.copy()
            p_k1 = S.p_wedge.copy() 
            
            res  = 1.0 
            it_inner   = 0 
            while (res > ctrl.tol_innerPic) and it_inner < ctrl.it_max: 
                time_ita = timing.time()
                if it_inner>0: 
                    a1,_,_, _,_ = self.set_linear_picard(u_k,S.t_owedge,S.p_lwedge,D,FGW,ctrl,sc)
                    a[0][0] = a1 
                    a       = fem.form(a)
                    a_p0[0][0] = a1 
                    a_p0       = fem.form(a_p0)
                
                u_k1, p_k1 ,r_al = self.solve_linear_picard(fem.form(a),fem.form(a_p0),fem.form(L),ctrl,u_k,p_k,it,ts)
                
                tol_u = compute_residuum(u_k1,u_k)

                tol_p = compute_residuum(p_k1,p_k)
                rmom, rdiv, divuL2 = self.compute_residuum_stokes(u_k1,p_k1,D,S.t_owedge,S.p_lwedge,FGW,sc)
                
                if it_inner == 0:
                    rmom_0 = rmom
                    rdiv_0 = rdiv
                    r_al0 = r_al
                
                                
                res   = np.sqrt(tol_u**2+tol_p**2)
                
                u_k.x.array[:] = ctrl.relax * u_k1.x.array[:] + (1-ctrl.relax) * u_k.x.array[:]
                p_k.x.array[:] = ctrl.relax * p_k1.x.array[:] + (1-ctrl.relax) * p_k.x.array[:]

                time_itb = timing.time()

                print_ph(f'              []Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
                print_ph(f'                         [x] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
                print_ph(f'                         [x] |F^mom|           {rmom:.3e},         |F^div| {rdiv:.3e}')
                print_ph(f'                         [x] |r_al|/|r_al0|    {r_al/r_al0:.3e},         |r_al|  {r_al:.3e}')

                it_inner = it_inner+1 
        
            print_ph(f'              []Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            print_ph(f'                         [?] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
            print_ph(f'                         [?] |F^mom|           {rmom:.3e}, abs div residuum |F^div| {rdiv:.3e}')
            print_ph('              []Converged ')

            S.u_wedge = u_k1.copy()
            S.p_wedge = p_k1.copy() 
        
        time_B = timing.time()
        print_ph(f'              // -- // --- Solution of Wedge in {time_B-time_A:.2f} sec // -- // --- >')
        print_ph('')

        return S 
    


#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Slab(Stokes_Problem): 
    """Slab problem class. 

    Args:
        Stokes_Problem: type of solver (i.e., scalar problem, stokes problem)
    
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
                 ,M:Mesh 
                 ,elements:tuple
                 ,name:list):
        super().__init__(M,elements,name)
        self.moving_wall : dolfinx.fem.function.Function = None
    
    def setdirichlecht(self
                       ,ctrl : NumericalControls
                       ,D : Domain
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
        
            self.compute_moving_wall(D,ctrl,'top_subduction')

        
        # Update the velocity field of the moving wall according to the current velocity of the slab.
        self.moving_wall.x.array[:] = self.moving_wall.x.array[:] * ctrl.v_s[0] 
        self.moving_wall.x.scatter_forward()
        # locate the dofs on the slab boundary and create dirichlet bc for them.
        dofs_s_x = fem.locate_dofs_topological(self.F0.sub(0), fdim, D.facets.find(D.bc_dict['top_subduction']))
        dofs_s_y = fem.locate_dofs_topological(self.F0.sub(1), fdim, D.facets.find(D.bc_dict['top_subduction']))
        # create the dirichlet bc for the slab boundary using the computed velocity field of the moving wall.
        bcx = fem.dirichletbc(self.moving_wall.sub(0), dofs_s_x)
        bcy = fem.dirichletbc(self.moving_wall.sub(1), dofs_s_y)
        
        return [bcx,bcy]
                
    #-------------------------------------------------------------------
    def compute_nitsche_FS(self
                           ,D:Domain
                           ,S:Solution
                           ,dS:ufl.measure.Measure  
                           ,a1:ufl.form.Form
                           ,a2:ufl.form.Form
                           ,a3:ufl.form.Form
                           ,FGS:Functions_material_properties_global
                           ,gamma:float
                           ,sc:Scal
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
        e   = compute_strain_rate(S.u_slab)   
        # Viscosity computation
        eta = compute_viscosity_FX(e,S.t_oslab,S.p_lslab,FGS,sc)
        # Extract the facet normal and the cell diameter for the mesh to compute the Nitsche terms.
        n = ufl.FacetNormal(D.mesh)
        h = ufl.CellDiameter(D.mesh)
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
                          ,S:Solution
                          ,ctrl:NumericalControls
                          ,FGS:Functions_material_properties_global
                          ,D:Domain
                          ,g:fem.Function
                          ,sc:Scal
                          ,it=0
                          ,ts=0):

        
        # Recreate the test and trial function on the subspace of the slab
        # Note: Initially, the problem was not solved because I was trying to reuse
        # The test and trial function created at the beginning of the simulation. 
        # However the composite spaces (V/P blocks) are not allowing to reuse these functions. 
        
        V_subs0 = self.FS.sub(0)
        p_subs0 = self.FS.sub(1)
        V_subs, _ = V_subs0.collapse()
        p_subs, _ = p_subs0.collapse()
    
        self.trial0 = ufl.TrialFunction(V_subs)
        self.test0 = ufl.TestFunction(V_subs)
        self.trial1 = ufl.TrialFunction(p_subs)
        self.test1 = ufl.TestFunction(p_subs)
        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_slab
                                                  ,S.t_oslab
                                                  ,S.p_lslab
                                                  ,D
                                                  ,FGS
                                                  ,ctrl
                                                  ,sc)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl
                                        ,D) 
        # Set Nietsche FS boundary condition 
        dS_bot = D.bc_dict["bot_subduction"]
        # 1 Extract ds 
        a1,a2,a3 = self.compute_nitsche_FS(D
                                           ,S
                                           ,dS_bot
                                           ,a1
                                           ,a2
                                           ,a3
                                           ,FGS
                                           ,50.0
                                           ,sc)
                
        a, a_p0 = self.fem_stokes_form(a1,a2,a3,a_p)
        time_A = timing.time()
        S.u_slab,S.p_slab, _ =  self.solve_linear_picard(fem.form(a)
                                                         ,fem.form(a_p0)
                                                         ,fem.form(L)
                                                         ,ctrl
                                                         ,S.u_slab
                                                         ,S.p_slab
                                                         ,it
                                                         ,ts)

        
        time_B = timing.time()
        print_ph(f'              // -- // --- Solution of Stokes problem in {time_B-time_A:.2f} sec // -- // --->')
        print_ph(f'')

        return S 
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#             x = self.solv.A.createVecRight()
#            self.solv.ksp.solve(self.solv.b, x)
#            if self.solv.direct_solver == 0: 
#                xu = x.getSubVector(self.solv.is_u)
#                xp = x.getSubVector(self.solv.is_p)
#            
#                # Copy OWNED entries only into the function PETSc Vecs
#            
#                with S.u_slab.x.petsc_vec.localForm() as uloc:
#                    uloc.array[:] = xu.array_r
#                with S.p_slab.x.petsc_vec.localForm() as ploc:
#                    ploc.array[:] = xp.array_r
#
#                # Fill ghosts
#                S.u_slab.x.scatter_forward()
#                S.p_slab.x.scatter_forward()
#
#                x.restoreSubVector(self.solv.is_u, xu)
#                x.restoreSubVector(self.solv.is_p, xp)
#        