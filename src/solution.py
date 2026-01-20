import numpy as np
import gmsh 

import ufl
import meshio
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx 
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile, gmshio
import gmsh 
from ufl import exp, conditional, eq, as_ufl
import basix.ufl
from .utils import timing_function, print_ph,time_the_time
from .compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX, compute_radiogenic 

from .create_mesh import Mesh 
from .phase_db import PhaseDataBase
import time                          as timing
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from .numerical_control import NumericalControls, ctrl_LHS, IOControls
from .utils import interpolate_from_sub_to_main
from .scal import Scal
from .output import OUTPUT,OUTPUT_WEDGE
from .utils import compute_eII,compute_strain_rate


"""
Lame explanation: So, my idea is to conceive everything in a set of problems, without creating ad hoc classes for the wedge, subduction and so on
yes by the end I do, but at least I create problem class that solves stokes/scalar problems. Most of the surrounding stuff (controls/phase) are set, 
[Problem]=> uses Solver for solving a piece of the domain or the whole domain -> to increase the modularity I can introduce a class for the BC, but, 
to lazy to do it, but could be something that I can do after I have a full working code. 

=> Global Variables 
print yes|no 
-> import a python code with global variable is it possible? 

-> Apperently, following the Stokes tutorial, I was porting the generation of null space: it appears, that this trick, is not working together with null space
-> Nitsche Boundary condition constraints the pressure, and the null space dislike it, apperently because the simmetry of the formulation is wrong, but this is black
magic. 
ta  = fem.Function(P0, name="ta")        # target Function
expr = fem.Expression(rho, P0.element.interpolation_points())

ta.interpolate(expr)    


"""

fig2 = plt.figure()
fig1 = plt.figure()
direct_solver = 1
DEBUG = 1

def mesh_of(obj):
    if hasattr(obj, "function_space"):    # Function
        return obj.function_space.mesh
    if hasattr(obj, "mesh"):              # FunctionSpace
        return obj.mesh
    if hasattr(obj, "ufl_domain"):        # Constant
        dom = obj.ufl_domain()
        return dom.ufl_cargo() if dom else None
    if hasattr(obj, "ufl_domains"):       # UFL expr
        ds = obj.ufl_domains()
        if len(ds) == 1:
            return list(ds)[0].ufl_cargo()
    return None

def check_single_domain(expr):
    """
    Inspect a UFL expression or Form and report all meshes/domains used.
    Returns True if all coefficients live on the same mesh, False otherwise.
    """
    from ufl.algorithms import extract_coefficients
    # Collect domains
    if hasattr(expr, "integrals"):      # Form
        doms = {itg.ufl_domain() for itg in expr.integrals()}
        kind = "Form"
    else:                               # Expression
        doms = set(expr.ufl_domains())
        kind = "Expr"

    print(f"[{kind}] domains referenced ({len(doms)}):")
    for d in doms:
        print("  -", d, "cargo:", d.ufl_cargo() if d else None)

    # Collect coefficients
    coeffs = list(extract_coefficients(expr))
    print(f"Coefficients used ({len(coeffs)}):")
    for c in coeffs:
        nm = getattr(c, "name", None) or repr(c)
        dom = getattr(c, "ufl_domain", lambda: None)()
        print("  -", nm, "on", dom.ufl_cargo() if dom else None)

    if len(doms) == 1:
        print(" :P Expression is single-domain.")
        return True
    else:
        print(" :( Expression involves multiple domains — rebuild coefficients on one mesh.")
        return False

#---------------------------------------------------------------------------

    
    
    


#--------------------------------------------------------------------------------------------------------------
class Solution():
    def __init__(self):
        self.PL       : dolfinx.fem.function.Function 
        self.T_i      : dolfinx.fem.function.Function
        self.T_O      : dolfinx.fem.function.Function 
        self.T_N      : dolfinx.fem.function.Function 
        self.u_global : dolfinx.fem.function.Function
        self.u_wedge  : dolfinx.fem.function.Function
        self.p_lwedge : dolfinx.fem.function.Function
        self.t_owedge : dolfinx.fem.function.Function
        self.p_lslab  : dolfinx.fem.function.Function
        self.t_oslab  : dolfinx.fem.function.Function
        self.u_slab   : dolfinx.fem.function.Function
        self.p_global : dolfinx.fem.function.Function 
        self.p_wedge  : dolfinx.fem.function.Function 
        self.p_slab   : dolfinx.fem.function.Function
        self.Hs_wedge : dolfinx.fem.function.Function
        self.Hs_slab  : dolfinx.fem.function.Function
        self.Hs_global : dolfinx.fem.function.Function
        
        
    def create_function(self,PG,PS,PW,elements): 
        """
        ======
        I am pretty pissed off about this problem: it is impossible to have a generalised approach that can be flexible 
        whoever had this diabolical idea, should be granted with a damnatio memoriae. So, inevitable, this class must 
        be modified as a function of the problem at hand.
        
        Argument: 
        PG : Global Problem 
        PS : Slab Problem
        PW : Wedge Problem
        
        -> update the solution functions that will be used later on. 
        
        =======
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
        self.u_global, self.p_global = gives_Function(space_GL)
        self.u_slab  , self.p_slab   = gives_Function(PS.FS)
        self.u_wedge , self.p_wedge  = gives_Function(PW.FS)

        return self 
#---------------------------------------------------------------------------
class Solvers():
    pass


class  ScalarSolver(Solvers):
    """
    class that store all the information for scalar like problems. Temperature and lithostatic pressure (potentially darcy like) are similar problem 
    they diffuse and advect a scalar. 
    --- 
    So -> Solver are more or less the same, I can store a few things and update as a function of the needs. 
    --- 
    Solve function require the problem P -> and a decision between linear and non linear -> form are handled by p class, so I do not give a fuck in this class 
    for now all the parameter will be default. 
    """
    def __init__(self,A ,b ,COMM, nl, J = None, r = None):
        self.A = fem.petsc.create_matrix(fem.form(A)) # Store the sparsisity 
        self.b = fem.petsc.create_vector(fem.form(b)) # Store the vector
        self.ksp = PETSc.KSP().create(COMM)           # Create the ksp object 
        self.ksp.setOperators(self.A)                # Set Operator
        self.ksp.setType("gmres")
        self.ksp.setTolerances(rtol=1e-4, atol=1e-3)
        self.pc = self.ksp.getPC()
        self.pc.setType("lu")
        self.pc.setFactorSolverType("mumps")

        self.J = None
        self.r = None 
    
class SolverStokes(): 
    
    def __init__(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0):
        if direct_solver == 1: 
            self.set_direct_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0)
            self.offset = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        elif direct_solver ==0: 
            self.set_iterative_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0)    
    
    

    def set_direct_solver(self,
                          a,
                          a_p,
                          L,
                          COMM, 
                          nl,
                          bcs,
                          F0,
                          F1, 
                          ctrl,
                          J = None, 
                          r = None,
                          it = 0, 
                          ts = 0):
        
        if it == 0 or ts == 0:
            self.set_block_operator(a, a_p, bcs, L, F0, F1)

            # Create a solver
            self.ksp = PETSc.KSP().create(COMM)
            self.ksp.setOperators(self.A)
            self.ksp.setType("preonly")

            # Set the solver type to MUMPS (LU solver) and configure MUMPS to
            # handle pressure nullspace
            self.pc = self.ksp.getPC()
            self.pc.setType("lu")
            use_superlu = PETSc.IntType == np.int64
            self.pc.setFactorSolverType("mumps")
            self.pc.setFactorSetUpSolverType()
            self.pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
            self.pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)


        
    
    
    def set_iterative_solver(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0): 
        #Return block operators and block RHS vector for the Stokes problem'
        # nullspace vector [0_u; 1_p] locally
        if it == 0 or ts == 0:
            self.set_block_operator(a,a_p,bcs,L,F0,F1)
            
            
            offset_u = 0
            offset_p = self.nloc_u
            # local starts in the *assembled block vector*

            self.is_u = PETSc.IS().createStride(self.nloc_u, offset_u, 1, comm=PETSc.COMM_SELF)
            self.is_p = PETSc.IS().createStride(self.nloc_p, offset_p, 1, comm=PETSc.COMM_SELF)
            V_map = F0.dofmap.index_map
            Q_map = F1.dofmap.index_map
            bs_u  = F0.dofmap.index_map_bs  # = mesh.dim for vector CG
            nloc_u = V_map.size_local * bs_u
            nloc_p = Q_map.size_local

            # local starts in the *assembled block vector*



            self.is_u = PETSc.IS().createStride(nloc_u, offset_u, 1, comm=PETSc.COMM_SELF)
            self.is_p = PETSc.IS().createStride(nloc_p, offset_p, 1, comm=PETSc.COMM_SELF)
            
            
            self.ksp = PETSc.KSP().create(COMM)
            self.ksp.setOperators(self.A, self.P)
            self.ksp.setTolerances(rtol=1e-9)
            self.ksp.setType("minres")
            self.ksp.getPC().setType("fieldsplit")
            self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            self.ksp.getPC().setFieldSplitIS(("u", self.is_u), ("p", self.is_p))

            # Configure velocity and pressure sub-solvers
            self.ksp_u, self.ksp_p = self.ksp.getPC().getFieldSplitSubKSP()
            self.ksp_u.setType("preonly")
            self.ksp_u.getPC().setType("gamg")
            self.ksp_p.setType("preonly")
            self.ksp_p.getPC().setType("jacobi")

            # The matrix A combined the vector velocity and scalar pressure
            # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
            # cannot infer the correct near-nullspace from the matrix block
            # size. Therefore, we set block size on the top-left block of the
            # preconditioner so that GAMG can infer the appropriate near
            # nullspace.
            self.ksp.getPC().setUp()
            self.Pu, _ = self.ksp_u.getPC().getOperators()
            self.Pu.setBlockSize(2)

    def set_block_operator(self,a,a_p,bcs,L,F0,F1):

        self.A = assemble_matrix_block(a, bcs=bcs)   ; self.A.assemble()
        self.P = assemble_matrix_block(a_p, bcs=bcs) ; self.P.assemble()
        self.b = assemble_vector_block(L, a, bcs=bcs); self.b.assemble()
        self.null_vec = self.A.createVecLeft()
        self.nloc_u = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        self.nloc_p = F1.dofmap.index_map.size_local
        self.null_vec.array[:self.nloc_u]  = 0.0
        self.null_vec.array[self.nloc_u:self.nloc_u+self.nloc_p] = 1.0
        self.null_vec.normalize()
        self.nsp = PETSc.NullSpace().create(vectors=[self.null_vec])
        self.A.setNullSpace(self.nsp)

        
        
#-------------------------------------------------------------------------       
def decoupling_function(z,fun,g_input):
    """
    Function explanation: 
    [a]: z depth coordinate 
    [b]: fun a function 
    [c]: ctrl => deprecated 
    [d]: D => deprecated 
    [e]: g_input -> still here, geometric parameters
    
    
    
    """
    
    dc = g_input.decoupling
    lit = g_input.lt_d
    dc = dc/g_input.decoupling
    lit = lit/g_input.decoupling
    z2 = np.abs(z)/g_input.decoupling
    trans = g_input.trans/g_input.decoupling
    

    fun.x.array[:] = 1-0.5 * ((1)+(1)*np.tanh((z2-dc)/(trans/4)))
    
    
    return fun
        

#----------------------------------------------------------------------------     
class Problem:
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
    def __init__(self, M: Mesh, elements: tuple, name: list):
        """
        Arguments: 
            M       : the mesh object 
            elements: tuple containing the elements.
                      If >1 -> assumes mixed problem
            name    : list ['problem_name', 'domain']
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
            trial         = ufl.TrialFunction(self.FS)
            test          = ufl.TestFunction(self.FS)
            self.trial0   = ufl.TrialFunction(self.FS.sub(0).collapse()[0])
            self.trial1   = ufl.TrialFunction(self.FS.sub(1).collapse()[0])
            self.test0    = ufl.TestFunction(self.FS.sub(0).collapse()[0])
            self.test1    = ufl.TestFunction(self.FS.sub(1).collapse()[0])
        
        self.dx       = ufl.Measure("dx", domain=M.mesh)
        self.ds       = ufl.Measure("ds", domain=M.mesh, subdomain_data=M.facets) # Exterior -> for boundary external 
        self.dS       = ufl.Measure("dS", domain=M.mesh, subdomain_data=M.facets) # Interior -> for boundary integral inside
 
        

        
#------------------------------------------------------------------
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
        fdim     = M.mesh.topology.dim - 1    
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

            cd_dof_b = cd_dof[dofs_right_lit]
    
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
        h_vel       = u_global.sub(0) # index 1 = y-direction (2D)   
        v_vel       = u_global.sub(1) # index 1 = y-direction (2D)
        vel_T  = fem.Function(self.FS)
        vel_T.interpolate(v_vel)
        vel_bc = vel_T.x.array[dofs_bot_wed]
        ind_z = np.where((vel_bc > 0.0))
        dofs_vel = dofs_bot_wed[ind_z[0]]                
        
        self.bc_bot_wed = fem.dirichletbc(ctrl.Tmax, dofs_vel,self.FS)
        
        bc = [self.bc_top, self.bc_left,  self.bc_right_wed,self.bc_bot_wed, self.bc_right_lit]

        
        
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
            efficiency    = heat_source.copy()
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
        test   = self.test0                 # v
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
    def compute_energy_source(self,D,pdb):
        source = fem.Function(self.FS)
        source = compute_radiogenic(pdb, source, D.phase, D.mesh)
        self.energy_source = source.copy()

    #------------------------------------------------------------------
    def compute_adiabatic_heating(self,D,pdb,u,T,p,ctrl):
        from .compute_material_property import alpha_FX
        
        if ctrl.adiabatic_heating != 0: 
            
        
            alpha = alpha_FX(pdb,T,p,D.phase,D)
            adiabatic_heating = alpha * T * ufl.inner(ufl.grad(p), u) 
        else: 
            adiabatic_heating = (0.0)
        
        
        self.adiabatic_heating = adiabatic_heating
        

    #------------------------------------------------------------------
    def compute_residual_SS(self,p_k,T,u_global,D,pdb, ctrl):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(pdb, T, p_k, D.phase, D.mesh)  # frozen
        
        Cp_k = heat_capacity_FX(pdb, T, D.phase, D.mesh)  # frozen

        k_k = heat_conductivity_FX(pdb, T, p_k, D.phase, D.mesh, Cp_k, rho_k)  # frozen

        self.compute_adiabatic_heating(D,pdb,u_global,T,p_k,ctrl)

        f    = self.energy_source# source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
            
        diff = ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        
        adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx
            
        L = fem.form((f + self.adiabatic_heating) * self.test0 * dx + self.shear_heating )      
        
        R = fem.form(diff + adv - L)
                

        return R
    
    
    def set_linear_picard_SS(self,p_k,T,u_global,Hs,D,pdb, ctrl, it=0):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(pdb, T, p_k, D.phase, D.mesh)  # frozen
        
        Cp_k = heat_capacity_FX(pdb, T, D.phase, D.mesh)  # frozen

        k_k = heat_conductivity_FX(pdb, T, p_k, D.phase, D.mesh, Cp_k, rho_k)  # frozen

        if ctrl.adiabatic_heating != 0:
            self.compute_adiabatic_heating(D,pdb,u_global,T,p_k,ctrl)
        else:
            self.adiabatic_heating = 0.0

        f    = self.energy_source# source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
        if it == 0: 
            
            diff = ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
            
            adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
            
            a = fem.form(diff + adv)
            
            L = fem.form((f + Hs + self.adiabatic_heating) * self.test0 * dx + self.shear_heating )      
        
        else: 
        
            a = None
                

        return a, L
    #------------------------------------------------------------------

    def set_linear_picard_TD(self,
                             p_k,     # lithostatic pressure current time step 
                             T_N,     # Temperature at new time step {new temperature guess}
                             T_O,     # Temperature at old time step
                             u_global,# velocity field 
                             D,       # domain
                             pdb,     # phase database
                             dt,      # time step
                             it=0):   # picard iteration 
        # Function that set linear form and linear picard for picard iteration
        # Crank Nicolson scheme 
        # a - > New temperature 
        # L - > Old temperature
        # -> Source term is assumed constant in time and do not vary between the timesteps 
        
        rho_k = density_FX(pdb, T_N, p_k, D.phase, D.mesh)  # frozen
                
        Cp_k = heat_capacity_FX(pdb, T_N, D.phase, D.mesh)  # frozen

        k_k = heat_conductivity_FX(pdb, T_N, p_k, D.phase, D.mesh, Cp_k, rho_k)  # frozen


        
        rho_k0 = density_FX(pdb, T_O, p_k, D.phase, D.mesh)  # frozen
                
        Cp_k0 = heat_capacity_FX(pdb, T_O, D.phase, D.mesh)  # frozen
        
        k_k0 = heat_conductivity_FX(pdb, T_O, p_k, D.phase, D.mesh, Cp_k, rho_k)  # frozen

                
        rhocp        =  (rho_k * Cp_k)

        rhocp_old    =  (rho_k0 * Cp_k0)
        
        dx  = self.dx
        
        f    = (self.energy_source+self.adiabatic_heating) * self.test0 * dx + self.shear_heating # source term {energy_source is radiogenic heating compute before hand, shear heating is frictional heating already a form}

        # Adiabatic term [Ex]

        # Linear operator with frozen coefficients

        
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
            return a, None
    #------------------------------------------------------------------

    def Solve_the_Problem_SS(self,S,ctrl,pdb,M,lhs,geom,sc,it=0,ts=0): 
        
        nl = 0 
        p_k = S.PL.copy()  # Previous lithostatic pressure 
        T   = S.T_O # -> will not eventually update 
        if ctrl.adiabatic_heating ==2:
            Hs = S.Hs_global # Shear heating
        else:
            Hs = 0.0
            
        if it == 0:         
            self.shear_heating = self.compute_shear_heating(ctrl,pdb, S,getattr(M,'domainG'),geom,sc)
            self.compute_energy_source(getattr(M,'domainG'),pdb)
        
        a,L = self.set_linear_picard_SS(p_k,T,S.u_global,Hs,getattr(M,'domainG'),pdb,ctrl)
        
        self.bc = self.create_bc_temp(getattr(M,'domainG'),ctrl,geom,lhs,S.u_global,S.T_i,it)
        if self.typology == 'NonlinearProblem':
            F,J = self.set_newton_SS(p_k,getattr(M,'domainG'),T,S.u_global,pdb)
            nl = 1 
        else: 
            F = None;J=None 

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        
        print_ph(f'              // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if nl == 0: 
            S = self.solve_the_linear(S,a,L,S.T_O) 
        else: 
            S = self.solve_the_non_linear_SS(M,S,Hs,ctrl,pdb)
        
        time_B = timing.time()
        
        print_ph(f'              // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return S 
    #------------------------------------------------------------------
    
    def Solve_the_Problem_TD(self,S,ctrl,pdb,M,lhs,geom,sc,it=0,ts=0): 

        nl = 0 
        p_k = S.PL.copy()  # Previous lithostatic pressure 



        if it == 0:
         
            self.shear_heating = self.compute_shear_heating(ctrl,pdb, S,getattr(M,'domainG'),geom,sc)
            self.compute_energy_source(getattr(M,'domainG'),pdb)
            self.compute_adiabatic_heating(getattr(M,'domainG'),pdb,S.u_global,T,p_k,ctrl)


        a,L = self.set_linear_picard_TD(p_k,S.T_N,S.T_O,S.u_global,getattr(M,'domainG'),pdb,ctrl.dt)

        self.bc = self.create_bc_temp(getattr(M,'domainG'),ctrl,geom,lhs,S.u_global,it)
        if self.typology == 'NonlinearProblem':
            nl = 1 
        F = None;J=None 

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        

        print_ph(f'. // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if nl == 0: 
            S.T_N = self.solve_the_linear(S,a,L,S.T_O) 
        else: 
            S = self.solve_the_non_linear_TD(M,S,ctrl,pdb)
        
        time_B = timing.time()
        
        print_ph(f'. // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return S 
    
    def solve_the_linear(self,S,a,L,fen_function,isPicard=0,it=0,ts=0):
        
        buf = fem.Function(self.FS)
        x   = self.solv.b.copy()
        self.solv.A.zeroEntries()
        fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc)
        self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        self.solv.b.set(0.0)
        fem.petsc.assemble_vector(self.solv.b, fem.form(L))
        fem.petsc.apply_lifting(self.solv.b, [fem.form(a)], [self.bc])
        self.solv.b.ghostUpdate()
        fem.petsc.set_bc(self.solv.b, self.bc)
        self.solv.ksp.solve(self.solv.b, fen_function.x.petsc_vec)
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            fen_function.x.scatter_forward()
            return fen_function 
        else:
            return fen_function

    def solve_the_non_linear_SS(self,M,S,Hs,ctrl,pdb,it=0):  
        
        tol = 1e-3  # Tolerance Picard  
        max_it = 20  # Max iteration before Newton
        isPicard = 1 # Flag for the linear solver. 
        tol = 1.0 
        T_k = S.T_O.copy() 
        T_k1 = S.T_O.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        time_A = timing.time()
        print_ph(f'              [//] Picard iterations for the non linear temperature problem')

        while it_inner < max_it and tol > 1e-5:
            time_ita = timing.time()
            
            A,L = self.set_linear_picard_SS(S.PL,T_k,S.u_global,Hs,getattr(M,'domainG'),pdb,ctrl)
            
            T_k1 = self.solve_the_linear(S,A,L,T_k1,1,it,1)
            T_k1.x.scatter_forward()
            # L2 norm 
            du.x.array[:]  = T_k1.x.array[:] - T_k.x.array[:];du.x.scatter_forward()
            du2.x.array[:] = T_k1.x.array[:] + T_k.x.array[:];du2.x.scatter_forward()
            tol= L2_norm_calculation(du)/L2_norm_calculation(du2)
            
            time_itb = timing.time()
            print_ph(f'              []Temperature L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            T_k.x.array[:] = T_k1.x.array[:]*0.7 + T_k.x.array[:]*(1-0.7)
            
            it_inner = it_inner + 1 
            
        # --- Newton =>         
        F,J = self.set_newton_SS(S.PL,getattr(M,'domainG'),T_k1,S.u_global,pdb)
        
        problem = fem.petsc.NonlinearProblem(F, T_k1, bcs=self.bc, J=J)

        # Newton solver
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "residual"
        solver.rtol = 1e-6
        solver.report = True
        
        #n, converged = solver.solve(T_k1)
        
        S.T_O.x.array[:] = T_k1.x.array[:]



        print_ph(f'')

        
        
        return S  

    def solve_the_non_linear_TD(self,M,S,ctrl,pdb,it=0):  
        
        tol = 1e-3  # Tolerance Picard  
        max_it = 20  # Max iteration before Newton
        isPicard = 1 # Flag for the linear solver. 
        tol = 1.0 
        T_O = S.T_O.copy()
        T_k = S.T_O.copy() 
        T_k1 = S.T_O.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        time_A = timing.time()
        print_ph(f'              [//] Picard iterations for the non linear temperature problem')

        while it_inner < max_it and tol > 1e-2:
            time_ita = timing.time()
            
            if it_inner ==0:

                A,L = self.set_linear_picard_TD(S.PL,T_k,S.T_O,S.u_global,getattr(M,'domainG'),pdb,ctrl.dt)
            else:
                A,_ = self.set_linear_picard_TD(S.PL,T_k,T_prev,S.u_global,getattr(M,'domainG'),pdb,ctrl.dt,it_inner)
            
            T_k1 = self.solve_the_linear(S,A,L,T_k1,1,it,1)
            T_k1.x.scatter_forward()
            # L2 norm 
            du.x.array[:]  = T_k1.x.array[:] - T_k.x.array[:];du.x.scatter_forward()
            du2.x.array[:] = T_k1.x.array[:] + T_k.x.array[:];du2.x.scatter_forward()
            tol= L2_norm_calculation(du)/L2_norm_calculation(du2)
            
            time_itb = timing.time()
            print_ph(f'              []Temperature L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            T_k.x.array[:] = T_k1.x.array[:]*0.7 + T_k.x.array[:]*(1-0.7)
            
            it_inner = it_inner + 1 
            
        # --- Newton =>         
        
        S.T_N.x.array[:] = T_k1.x.array[:]
        S.T_N.x.scatter_forward()



        print_ph(f'')

        
        
        return S

    #------------------------------------------------------------------
        
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
            T_gr = (-M.g_input.lt_d-0)/(ctrl.Tmax-ctrl.Ttop)
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


               
    
        return T_i 
#-----------------------------------------------------------------
class Global_pressure(Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        if np.all(pdb.option_rho<2):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'

        self.bc = [self.set_problem_bc(getattr(M,'domainG'))]
    def set_problem_bc(self,M):
         
        fdim = 1    
        top_facets   = M.facets.find(M.bc_dict['Top'])
        top_dofs    = fem.locate_dofs_topological(self.FS, 1, top_facets)
        bc = [fem.dirichletbc(0.0, top_dofs, self.FS)]
        return bc  
    
    def set_newton(self,p,D,T,g,pdb):
        test   = self.test0                 # v
        trial0 = self.trial0                # δp (TrialFunction) for Jacobian

        a_lin = ufl.inner(ufl.grad(p), ufl.grad(test)) * self.dx
        L     = ufl.inner(ufl.grad(test),
                              density_FX(pdb, T, p, D.phase, D.mesh) * g) * self.dx
        # Nonlinear residual: F(p; v) = ∫ ∇p·∇v dx - ∫ ∇v·(ρ(T, p) g) dx
        F = a_lin - L
        # Jacobian dF/dp in direction δp (trial0)
        J = ufl.derivative(F, p, trial0)
        
        return F,J 
    
    
    def set_linear_picard(self,p_k,T,D,pdb,g, it=0):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(pdb, T, p_k, D.phase, D.mesh)  # frozen
        
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
        T   = S.T_O # -> will not eventually update 
        
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
        
        print_ph(f'              // -- // --- LITHOSTATIC PROBLEM [GLOBAL] // -- // --- > ')

        time_A = timing.time()

        
        if nl == 0: 
            S = self.solve_the_linear(S,a,L,S.PL) 
        else: 
            S = self.solve_the_non_linear(M,S,ctrl,pdb,g)

        time_B = timing.time()

        print_ph(f'              // -- // --- Solution of Lithostatic pressure problem finished in {time_B-time_A:.2f} sec // -- // --->')
        print_ph(f'')

        return S 
    
    def solve_the_linear(self,S,a,L,function_fen,isPicard=0,it=0,ts=0):
        
        buf = fem.Function(self.FS)
        x   = self.solv.b.copy()
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
    
    def solve_the_non_linear(self,M,S,ctrl,pdb,g):  
        
        tol = 1e-3  # Tolerance Picard  
        isPicard = 1 # Flag for the linear solver. 
        tol = 1.0 
        p_k = S.PL.copy() 
        p_k1 = S.PL.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        
        
        print_ph(f'              [//] Picard iterations for the non linear lithostatic pressure problem')

        
        while it_inner < ctrl.it_max and tol > ctrl.tol_innerPic:
            time_ita = timing.time()
            
            if it_inner == 0:
                A,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),pdb,g)
            else: 
                _,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),pdb,g,1)
            
            p_k1 = self.solve_the_linear(S,A,L,p_k1,1,it_inner,1) 

            # L2 norm 
            du.x.array[:]  = p_k1.x.array[:] - p_k.x.array[:];du.x.scatter_forward()
            du2.x.array[:] = p_k1.x.array[:] + p_k.x.array[:];du2.x.scatter_forward()
            tol= L2_norm_calculation(du)/L2_norm_calculation(du2)
            
            time_itb = timing.time()
            print_ph(f'              []L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            p_k.x.array[:] = p_k1.x.array[:]*0.7 + p_k.x.array[:]*(1-0.7)
            
            it_inner = it_inner + 1 
        
        print_ph(f'              [//] Newton iterations for the non linear lithostatic pressure problem')

        # --- Newton =>         
        F,J = self.set_newton(p_k,getattr(M,'domainG'),S.T_O,g,pdb)
        
        problem = fem.petsc.NonlinearProblem(F, p_k, bcs=self.bc[0], J=J)

        # Newton solver
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "residual"
        #solver.rtol = ctrl.tol_innerNew
        #solver.report = True
        
        #n, converged = solver.solve(p_k)
        #print(f"              []Newton iterations: {n}, converged = {converged}")   
        
        S.PL.x.array[:] = p_k.x.array[:]
      
        
        
        return S  
        
        
#-----------------------------------------------------------------
class Stokes_Problem(Problem):
    def __init__(self,M,elements,name):
        super().__init__(M,elements,name)       
        M = getattr(M,name[1])
        self.FSPT = dolfinx.fem.functionspace(M.mesh, elements[2]) 
        
    def set_newton():
        # Place holder -> Not finding useful tutorial for newton 
     
        pass
    
    def compute_shear_heating(self,ctrl,pdb,S,D,sc,wedge=1):
        from .compute_material_property import compute_viscosity_FX
        from .utils import evaluate_material_property
        if wedge ==1: 
            V = S.u_wedge.function_space
            PT = self.FSPT
            v = ufl.TestFunction(V)
            e = compute_strain_rate(S.u_wedge)
            vel_e = ufl.sym(ufl.grad(S.u_wedge))
            eta_new = compute_viscosity_FX(e, S.t_owedge, S.p_lwedge, pdb, D.phase, D, sc)
        else: 
            V = S.u_slab.function_space
            PT = self.FSPT
            e = compute_strain_rate(S.u_slab)
            vel_e = ufl.sym(ufl.grad(S.u_slab))
            eta_new = compute_viscosity_FX(e, S.t_oslab, S.p_lslab, pdb, D.phase, D, sc)

        shear_heating = ufl.inner(2*eta_new*vel_e, vel_e)

        if wedge ==1: 
            S.Hs_wedge = evaluate_material_property(shear_heating,PT)
        else: 
            S.Hs_slab = evaluate_material_property(shear_heating,PT)

        return S
    
    def compute_residuum_stokes(self, u_new, p_new, D, T, PL, pdb, sc):
        V = u_new.function_space
        Q = p_new.function_space
        v = ufl.TestFunction(V)
        q = ufl.TestFunction(Q)

        e = compute_strain_rate(u_new)
        eta_new = compute_viscosity_FX(e, T, PL, pdb, D.phase, D, sc)
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

    
    
    def set_linear_picard(self,u_slab,T,PL,D,pdb,ctrl,sc, a_p = None,it=0, ts = 0):
        
        """
        The problem is always LINEAR when depends on T/P -> Becomes fairly 
        non linear when the temperature is involved. 
        """
        
        u, p  = self.trial0, self.trial1
        v, q  = self.test0,  self.test1
        dx    = ufl.dx

        e = compute_strain_rate(u_slab)

        eta = compute_viscosity_FX(e,T,PL,pdb,D.phase,D,sc)

        a1 = ufl.inner(2*eta*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
        a2 = - ufl.inner(ufl.div(v), p) * dx             # build once
        a3 = - ufl.inner(q, ufl.div(u)) * dx             # build once
        a_p0 = ufl.inner(q, p) * dx                      # pressure mass (precond)

        f  = fem.Constant(D.mesh, PETSc.ScalarType((0.0,)*D.mesh.geometry.dim))
        f2 = fem.Constant(D.mesh, PETSc.ScalarType(0.0))
        L  = fem.form([ufl.inner(f, v)*dx, ufl.inner(f2, q)*dx])    
    
        return a1, a2, a3 , L , a_p0       

        

        
def L2_norm_calculation(f):
    return fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx))**0.5       
        
        
#------------------------------------------------------------------
class Slab(Stokes_Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
    
    def setdirichlecht(self,ctrl,D,theta,it=0, ts = 0):
        mesh = self.F0.mesh 
        tdim = mesh.topology.dim
        fdim = tdim - 1
        
        # facet ids
        inflow_facets  = D.facets.find(D.bc_dict['inflow'])
        outflow_facets = D.facets.find(D.bc_dict['outflow'])
        slab_facets    = D.facets.find(D.bc_dict['top_subduction'])
        
        if (it == 0 or ts == 0 ):
            if ctrl.slab_bc == 1: # moving wall slab bc 
                # Compute this crap only once @ the beginning of the simulation 
                if it == 0 and ts == 0:
                        dofs        = fem.locate_dofs_topological(self.F0, fdim, slab_facets)

                        n = ufl.FacetNormal(mesh)                    # exact facet normal in    weak   form
                        v_slab = float(1.0)  # slab velocity magnitude

                        v_const = ufl.as_vector((ctrl.v_s[0], 0.0))

                        proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)  # projector   onto  the  tangential plane
                        t = ufl.dot(proj, v_const)                     # tangential     velocity    vector on  slab
                        t_hat = t / ufl.sqrt(ufl.inner(t, t))    
                        v_project = v_slab * t_hat  # projected tangential velocity vector on   slab
                        self.moving_wall = fem.Function(self.F0)
                        w = self.trial0
                        v = self.test0
                        a = ufl.inner(w, v) * self.ds(D.bc_dict['top_subduction'])        #  boundary     mass    matrix (vector)
                        L = ufl.inner(v_project, v) * self.ds(D.bc_dict['top_subduction'])

                        self.moving_wall = fem.petsc.LinearProblem(
                            a, L,
                            petsc_options={
                            "ksp_type": "cg",
                            "pc_type": "jacobi",
                            "ksp_rtol": 1e-20,
                            }
                        ).solve()  # ut_h \in V
                self.moving_wall.x.array[:] = self.moving_wall.x.array[:]*ctrl.v_s[0] 

                dofs_s_x = fem.locate_dofs_topological(self.F0.sub(0), fdim, slab_facets)
                dofs_s_y = fem.locate_dofs_topological(self.F0.sub(1), fdim, slab_facets)

                bcx = fem.dirichletbc(self.moving_wall.sub(0), dofs_s_x)
                bcy = fem.dirichletbc(self.moving_wall.sub(1), dofs_s_y)
                return [bcx,bcy]
            else:
                # Pipe like boundary condition  {dirichlect bottom top and free slip in the boundaries applied with nietsche method.}
                dofs_in_x  = fem.locate_dofs_topological(self.F0.sub(0), fdim, inflow_facets)
                dofs_in_y  = fem.locate_dofs_topological(self.F0.sub(1), fdim, inflow_facets)
                dofs_out_x = fem.locate_dofs_topological(self.F0.sub(0), fdim, outflow_facets)
                dofs_out_y = fem.locate_dofs_topological(self.F0.sub(1), fdim, outflow_facets)
                
                # scalar BCs on subspaces
                bc_left_x   = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]), dofs_in_x,  self.F0.sub(0))
                bc_left_y   = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[1]), dofs_in_y,    self.F0.sub(1))
                bc_bottom_x = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]*np.cos(theta)),          dofs_out_x, self.F0.sub(0))
                bc_bottom_y = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]*np.sin(theta)),          dofs_out_y, self.F0.sub(1))
                
                return [bc_left_x,bc_left_y,bc_bottom_x,bc_bottom_y]
                
    
    def compute_nitsche_FS(self,D, S, dS, a1,a2,a3,pdb,gamma,sc,it = 0):
        """
        Compute the Nitsche free slip boundary condition for the slab problem.
        This is a placeholder function and should be implemented with the actual Nitsche    method.
        """
        def tau(eta, u):
            return 2 * eta * ufl.sym(ufl.grad(u))
        
        
        # Linear 
        e   = compute_strain_rate(S.u_slab)   
        eta = compute_viscosity_FX(e,S.t_oslab,S.p_lslab,pdb,D.phase,D,sc)
        
        n = ufl.FacetNormal(D.mesh)
        h = ufl.CellDiameter(D.mesh)

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
    
    def Solve_the_Problem(self,S,ctrl,pdb,M,g,sc,it=0,ts=0):
        theta = M.g_input.theta_out_slab
        M = getattr(M,'domainA')
        
        
        """
        According to chatgpt, and I have already verified this notion before asking, 
        I cannot create test and trial function from a previous instance. In case of mixed space
        I need always to recreate on the spot, which is very annoying despite they are pretty much cheap 
        in terms of computation. 
        
    
        """
        V_subs0 = self.FS.sub(0)
        p_subs0 = self.FS.sub(1)
        V_subs, _ = V_subs0.collapse()
        p_subs, _ = p_subs0.collapse()
    
        self.trial0 = ufl.TrialFunction(V_subs)
        self.test0 = ufl.TestFunction(V_subs)
        self.trial1 = ufl.TrialFunction(p_subs)
        self.test1 = ufl.TestFunction(p_subs)
        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_slab,S.t_oslab,S.p_lslab,M,pdb,ctrl,sc)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl,M,theta) 
        # Set Nietsche FS boundary condition 
        dS_top = M.bc_dict["top_subduction"]
        dS_bot = M.bc_dict["bot_subduction"]
        # 1 Extract ds 
        a1,a2,a3 = self.compute_nitsche_FS(M, S, dS_bot, a1, a2 ,a3,pdb ,50.0,sc)
        
        if ctrl.slab_bc == 0: 
            a1,a2,a3 = self.compute_nitsche_FS(M, S, dS_top, a1, a2 ,a3,pdb ,50.0,sc)
        
        # Slab problem is ALWAYS LINEAR
        
        a   = fem.form([[a1, a2],[a3, None]])
        a_p0  = fem.form([[a1, None],[None, a_p]])


        self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,ctrl ,J = None, r = None,it = 0, ts = 0)
        
        print_ph(f'              // -- // --- SLAB STOKES PROBLEM // -- // --->')    
        time_A = timing.time()    
        if direct_solver==1:
            x = self.solv.A.createVecLeft()
        else:
            x = self.solv.A.createVecRight()
        self.solv.ksp.solve(self.solv.b, x)
        if direct_solver == 0: 
            xu = x.getSubVector(self.solv.is_u)
            xp = x.getSubVector(self.solv.is_p)
            u, p = fem.Function(self.F0), fem.Function(self.F1)
    
            u.x.array[:] = xu.array_r
            p.x.array[:] = xp.array_r
            p.x.scatter_forward()
            u.x.scatter_forward()
            S.u_slab.x.array[:] = u.x.array[:]
            S.p_slab.x.array[:] = p.x.array[:]
            S.u_slab.x.scatter_forward()
            S.p_slab.x.scatter_forward()
        else: 
            S.u_slab.x.array[:self.solv.offset] = x.array_r[:self.solv.offset]
            S.p_slab.x.array[: (len(x.array_r) - self.solv.offset)] = x.array_r[self.solv.offset:]
            S.u_slab.x.scatter_forward()
            S.p_slab.x.scatter_forward()
        

        time_B = timing.time()
        print_ph(f'              // -- // --- Solution of Stokes problem in {time_B-time_A:.2f} sec // -- // --->')
        print_ph(f'')
        if ctrl.adiabatic_heating==2:
            S = self.compute_shear_heating(ctrl,pdb,S,M,sc,wedge=0)
        return S 
    
#---------------------------------------------------------------------------------------------------       
class Wedge(Stokes_Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        
        M = getattr(M,'domainB')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Example: each rank has some local IDs
        local_ids = np.int32(M.phase.x.array[:])

        # Gather arrays from all processes
        all_ids = comm.allgather(local_ids)

        # Root (or all, since allgather) concatenates
        all_ids = np.concatenate(all_ids)

        # Compute global unique IDs
        unique_ids = np.unique(all_ids)

        non_linear = False 
        
        for i in range(np.shape(unique_ids)[0]): 
            if pdb.option_eta[unique_ids[i]] > 0: 
                non_linear = True
                break
            
        if non_linear == True:
            self.typology = 'NonlinearProblem'
        else:
            self.typology = 'LinearProblem'
        
        
            
    
    def setdirichlecht(self,ctrl,D,theta,V,g_input,it=0, ts = 0):
        mesh = V.mesh 
        tdim = mesh.topology.dim
        fdim = tdim - 1
        
    
        # facet ids
        overriding_facet = D.facets.find(D.bc_dict['overriding'])
        slab_facets      = D.facets.find(D.bc_dict['slab'])
        
            
        # Compute this crap only once @ the beginning of the simulation 


        noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
        dofs_over = fem.locate_dofs_topological(V, fdim, D.facets.find(D.bc_dict['overriding']))

        bc_overriding = fem.dirichletbc(noslip, dofs_over, V)
        
        
        #------------------------------------------------------------------------
        dofs        = fem.locate_dofs_topological(V, fdim, slab_facets)

        n = ufl.FacetNormal(mesh)                    # exact facet normal in    weak   form
        v_slab = float(1.0)  # slab velocity magnitude

        v_const = ufl.as_vector((ctrl.v_s[0], 0.0))

        proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)  # projector   onto  the  tangential plane
        t = ufl.dot(proj, v_const)                     # tangential     velocity    vector on  slab
        t_hat = t / ufl.sqrt(ufl.inner(t, t))    
        v_project = v_slab * t_hat  # projected tangential velocity vector on   slab
        moving_wall_wedge = fem.Function(V)
        w = self.trial0
        v = self.test0
        

              
        a = ufl.inner(w, v) * self.ds(D.bc_dict['slab'])        #  boundary     mass    matrix (vector)
        
        if ctrl.decoupling == 1:
            scaling = fem.Function(self.FSPT)
            scaling = decoupling_function(self.FSPT.tabulate_dof_coordinates()[:,1],scaling,g_input)  
            scaling.x.array[:] = 1-scaling.x.array[:]
            scaling.x.scatter_forward()
            L = ufl.inner(v_project * scaling, v) * self.ds(D.bc_dict['slab'])
        else: 
            L = ufl.inner(v_project, v) * self.ds(D.bc_dict['slab'])

        moving_wall_wedge = fem.petsc.LinearProblem(
            a, L,
            petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-20,
            }
        ).solve()
            
        self.moving_wall_wedge = moving_wall_wedge.copy()
        self.moving_wall_wedge.x.array[:] = moving_wall_wedge.x.array[:]*ctrl.v_s[0] 

        dofs_s_x = fem.locate_dofs_topological(V.sub(0), fdim,slab_facets)
        dofs_s_y = fem.locate_dofs_topological(V.sub(1), fdim,slab_facets)

        bcx = fem.dirichletbc(self.moving_wall_wedge.sub(0), dofs_s_x)
        bcy = fem.dirichletbc(self.moving_wall_wedge.sub(1), dofs_s_y)
        
    
        return [bcx, bcy, bc_overriding]
                
    
    def compute_nitsche_traction(self,D, S, dS, a1,a2,a3,pdb ,gamma,it = 0):
        # Place holder for eventual traction boundary condition 
        """
        In case I want to introduce the feedback between thermal evolution and density 
        I will introduce this boundary condition. The material properties will be computed 
        as a function of the lithostatic pressure 
        
        
        
        """


        return a1, a2, a3 
    
    def Solve_the_Problem(self,S,ctrl,pdb,M,g,sc,g_input,it=0,ts=0):
        theta = M.g_input.theta_out_slab
        M = getattr(M,'domainB')

        
        """
        According to chatgpt, and I have already verified this notion before asking, 
        I cannot create test and trial function from a previous instance. In case of mixed space
        I need always to recreate on the spot, which is very annoying despite they are pretty much cheap 
        in terms of computation. 
        
    
        """
        V_subs0 = self.FS.sub(0)
        p_subs0 = self.FS.sub(1)
        V_subs, _ = V_subs0.collapse()
        p_subs, _ = p_subs0.collapse()
    
        self.trial0 = ufl.TrialFunction(V_subs)
        self.test0 = ufl.TestFunction(V_subs)
        self.trial1 = ufl.TrialFunction(p_subs)
        self.test1 = ufl.TestFunction(p_subs)
    

        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_wedge,S.t_owedge,S.p_lwedge,M,pdb,ctrl,sc)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl,M,theta,V_subs,g_input) 
        
        # Form the system 
        a   = [[a1, a2],[a3, None]]
        a_p0  = [[a1, None],[None, a_p]]

        print_ph(f'              // -- // --- STOKES PROBLEM [WEDGE] // -- // --- > ')

        
        time_A = timing.time()

        
        if self.typology == 'LinearProblem': 
            S,r_al = self.solve_linear_picard(fem.form(a),fem.form(a_p0),fem.form(L),ctrl, S)

        else: 
            print_ph(f'              [//] Picard iterations for the non linear lithostatic pressure problem')

            u_k   = S.u_wedge.copy()
            p_k   = S.p_wedge.copy()
            du0   = S.u_wedge.copy()
            dp0   = S.p_wedge.copy()
            du1   = S.u_wedge.copy()
            dp1   = S.p_wedge.copy()
            
            res  = 1.0 
            it_inner   = 0 
            while (res > ctrl.tol_innerPic) and it_inner < ctrl.it_max: 
                time_ita = timing.time()
                if it_inner>0: 
                    a1,_,_, _,_ = self.set_linear_picard(u_k,S.t_owedge,S.p_lwedge,M,pdb,ctrl,sc)
                    a[0][0] = a1 
                    a       = fem.form(a)
                    a_p0[0][0] = a1 
                    a_p0       = fem.form(a_p0)
                
                S,r_al = self.solve_linear_picard(fem.form(a),fem.form(a_p0),fem.form(L),ctrl, S,it)
                
                
                du0.x.array[:]  = S.u_wedge.x.array[:] - u_k.x.array[:];du0.x.scatter_forward()
                
                du1.x.array[:] = S.u_wedge.x.array[:] + u_k.x.array[:];du1.x.scatter_forward()
                
                dp0.x.array[:]  = S.p_wedge.x.array[:] - p_k.x.array[:];dp0.x.scatter_forward()
                
                dp1.x.array[:] = S.p_wedge.x.array[:] + p_k.x.array[:];dp1.x.scatter_forward()
        
                rmom, rdiv, divuL2 = self.compute_residuum_stokes(S.u_wedge,S.p_wedge,M,S.t_owedge,S.p_lwedge,pdb,sc)
                
                if it_inner == 0:
                    rmom_0 = rmom
                    rdiv_0 = rdiv
                    r_al0 = r_al
                
        
                tol_u = L2_norm_calculation(du0)/L2_norm_calculation(du1)
                
                tol_p = L2_norm_calculation(dp0)/L2_norm_calculation(dp1)
                                
                res   = np.sqrt(tol_u**2+tol_p**2)
                
                u_k.x.array[:] = ctrl.relax*S.u_wedge.x.array[:] + (1-ctrl.relax)*u_k.x.array[:]
                p_k.x.array[:] = ctrl.relax*S.p_wedge.x.array[:] + (1-ctrl.relax)*p_k.x.array[:]

                time_itb = timing.time()

                print_ph(f'              []Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
                print_ph(f'                         [x] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
                print_ph(f'                         [x] |F^mom|           {rmom:.3e},         |F^div| {rdiv:.3e}')
                print_ph(f'                         [x] |r_al|/|r_al0|    {r_al/r_al0:.3e},         |r_al|  {r_al:.3e}')

                it_inner = it_inner+1 
        
            print_ph(f'              []Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            print_ph(f'                         [?] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
            print_ph(f'                         [?] |F^mom|           {rmom:.3e}, abs div residuum |F^div| {rdiv:.3e}')
            print_ph(f'              []Converged ')

        time_B = timing.time()
        print_ph(f'              // -- // --- Solution of Wedge in {time_B-time_A:.2f} sec // -- // --- >')
        print_ph(f'')



        if ctrl.adiabatic_heating==2:
            S = self.compute_shear_heating(ctrl,pdb,S,M,sc,wedge=1)
        


        return S 
    
    def solve_linear_picard(self,a,a_p0,L,ctrl, S,it=0,ts=0):
            

        self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,ctrl ,J = None, r = None,it = 0, ts = 0)
        
        
        if direct_solver==1:
            x = self.solv.A.createVecLeft()
        else:
            x = self.solv.A.createVecRight()
        self.solv.ksp.solve(self.solv.b, x)
        if direct_solver == 0: 
            xu = x.getSubVector(self.solv.is_u)
            xp = x.getSubVector(self.solv.is_p)
    

            S.u_wedge.x.array[:] = xu.array_r[:]
            S.p_wedge.x.array[:] = xp.array_r[:]
            S.u_wedge.x.scatter_forward()
            S.p_wedge.x.scatter_forward()
            

            
        else: 
            S.u_wedge.x.array[:self.solv.offset] = x.array[:self.solv.offset]
            S.p_wedge.x.array[: (len(x.array_r) - self.solv.offset)] = x.array[self.solv.offset:]
            S.u_wedge.x.scatter_forward()
            S.p_wedge.x.scatter_forward()
        
        r = self.solv.b.copy()
        self.solv.A.mult(x, r)          # r = A x
        r.scale(-1.0)         # r = -A x
        r.axpy(1.0, self.solv.b)        # r = b - A x
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        abs_res = r.norm()



        return S,abs_res 

def compute_adiabatic_initial_adiabatic_contribution(M,T,Tgue,p,pdb,vankeken): 
    
    from .compute_material_property import alpha_FX 
    
    
    FS = T.function_space 
    Tg = fem.Function(FS)
    Tg = T.copy()
    v  = ufl.TestFunction(FS)


    if vankeken==0:
        expr = (alpha_FX(pdb,Tg,p,M.phase,M) * p)/(heat_capacity_FX(pdb,Tg,M.phase,M) * density_FX(pdb,Tg,p,M.phase,M))
        F = (Tg-T * ufl.exp(expr)) * v * ufl.dx 
    
    
        bcs = []

        problem = NonlinearProblem(F, Tg, bcs, J)
        solver = NewtonSolver(M.mesh.comm, problem)

        solver.rtol = 1e-4
        solver.atol = 1e-4



        n_iter, converged = solver.solve(Tg)
        Tg.x.scatter_forward()
    else: 
        from utils import evaluate_material_property
        expr = (alpha_FX(pdb,Tg,p,M.phase,M) * p)/(heat_capacity_FX(pdb,Tg,M.phase,M) * density_FX(pdb,Tg,p,M.phase,M))
        F = T * ufl.exp(expr)
        Tg = evaluate_material_property(F,FS)

    
    return Tg
    
    

def initial_adiabatic_lithostatic_thermal_gradient(sol,lps,pdb,M,g,it_outer,ctrl):
    res = 1 
    it = 0
    T_0 = sol.T_O.copy()
    T_Oa = sol.T_O.copy()
    while res > 1e-3: 
        P_old = sol.PL.copy()
        sol = lps.Solve_the_Problem(sol,ctrl,pdb,M,g,it_outer,ts=0)
        T_O = compute_adiabatic_initial_adiabatic_contribution(M.domainG,T_0,T_Oa,sol.PL,pdb,ctrl.van_keken)
        resp = compute_residuum(sol.PL,P_old)
        resT = compute_residuum(T_O, T_Oa)
        res = np.sqrt(resp**2 + resT**2)
        if it !=0: 
            sol.PL.x.array[:] = 0.8 * sol.PL.x.array[:] + (1-0.8) * P_old.x.array[:]
            sol.PL.x.scatter_forward()
            T_Oa.x.array[:]= T_O.x.array[:] * 0.8 + (1-0.8) * T_O.x.array[:]
            sol.T_O.x.scatter_forward()
        it = it + 1 
        sol.T_O = T_Oa.copy()
        print_ph('Adiabatic res is %.3e'%res)
    
    sol.T_i = T_O.copy()
    return sol 

    
     

#------------------------------------------------------------------------------------------------------------
@timing_function
def steady_state_solution(M:Mesh, ctrl:NumericalControls, lhs_ctrl:ctrl_LHS, pdb:PhaseDataBase, ioctrl:IOControls, sc:Scal)-> int:
    from .phase_db import PhaseDataBase
    from .phase_db import _generate_phase
    from .thermal_structure_ocean import compute_initial_LHS
    from .scal import Scal 
    
    
    
    print_ph(f'// -- // --- Steady State temperature // -- // --- > ')

    element_p           = M.element_p#basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = M.element_PT#basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = M.element_V#basix.ufl.element("Lagrange","triangle",2,shape=(2,))

    #==================== Phase Parameter ====================
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)  
          
    # Define Problem
    
    # Pressure 
    
    energy_global               = Global_thermal (M = M, name = ['energy','domainG']  , elements = (element_PT,                   ), pdb = pdb, ctrl = ctrl)
    
    lithostatic_pressure_global = Global_pressure(M = M, name = ['pressure','domainG'], elements = (element_PT,                     ), pdb = pdb                                ) 
    
    wedge                       = Wedge          (M = M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT  ), pdb = pdb                                )
    
    slab                        = Slab           (M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p,element_PT  ), pdb = pdb                                )
    
    g = fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Define Solution 
    sol                         = Solution()
     
    sol.create_function(lithostatic_pressure_global,slab,wedge,[element_V,element_p])
    
    sol.T_O = energy_global.initial_temperature_field(M.domainG, ctrl, lhs_ctrl,M.g_input)

    it_outer = 0 

    res = 1.0
    
    output = OUTPUT(M.domainG.mesh, ioctrl, ctrl, sc,0)
    
    #output_W = OUTPUT_WEDGE(M.domainB.mesh,ioctrl,ctrl,sc,0)
         

    while it_outer < ctrl.it_max and res > ctrl.tol: 
        
        print_ph(f'// -- // --- Outer iteration {it_outer:d} for the coupled problem // -- // --- > ')
        
        time_A_outer = timing.time()
        # Copy the old solution for the residuum computation 
        
        Told        = sol.T_O.copy()
        PLold       = sol.PL.copy()
        u_globalold = sol.u_global.copy()
        p_globalold = sol.p_global.copy()
        
        if (ctrl.adiabatic_heating != 0) and (it_outer==0) :
            sol = initial_adiabatic_lithostatic_thermal_gradient(sol,lithostatic_pressure_global,pdb,M,g,it_outer,ctrl)
        else: 
            sol.T_i = sol.T_O
        
        if lithostatic_pressure_global.typology == 'NonlinearProblem' or it_outer == 0:  
            lithostatic_pressure_global.Solve_the_Problem(sol,ctrl,pdb,M,g,it_outer,ts=0)

        # Interpolate from global to wedge/slab

        sol.t_owedge = interpolate_from_sub_to_main(sol.t_owedge,sol.T_O, M.domainB.cell_par,1)
        sol.p_lwedge = interpolate_from_sub_to_main(sol.p_lwedge,sol.PL, M.domainB.cell_par,1)

        if it_outer == 0: 
            slab.Solve_the_Problem(sol,ctrl,pdb,M,g,sc,it = it_outer,ts=0)

        if wedge.typology == 'NonlinearProblem' or it_outer == 0:  
            wedge.Solve_the_Problem(sol,ctrl,pdb,M,g,sc,M.g_input,it = it_outer,ts=0)


        # Interpolate from wedge/slab to global
        sol.u_global = interpolate_from_sub_to_main(sol.u_global,sol.u_wedge, M.domainB.cell_par)
        sol.u_global = interpolate_from_sub_to_main(sol.u_global,sol.u_slab, M.domainA.cell_par)
        
        sol.Hs_global = interpolate_from_sub_to_main(sol.Hs_global,sol.Hs_wedge, M.domainB.cell_par)
        sol.Hs_global = interpolate_from_sub_to_main(sol.Hs_global,sol.Hs_slab, M.domainA.cell_par)
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global,sol.p_wedge, M.domainB.cell_par)
        sol.p_global = interpolate_from_sub_to_main(sol.p_global,sol.p_slab, M.domainA.cell_par)
        
        
        energy_global.Solve_the_Problem_SS(sol,ctrl,pdb,M,lhs_ctrl,M.g_input,sc,it = it_outer, ts = 0)
        
        # Compute residuum 
        res = compute_residuum_outer(sol,Told,PLold,u_globalold,p_globalold,it_outer,sc, time_A_outer)


        print_ph(f'// -- // :( --- ------- ------- ------- :) // -- // --- > ')

            
        it_outer = it_outer + 1
        
    
    
    output.print_output(sol,M.domainG,pdb,ioctrl,sc,ctrl,ts=it_outer)
    #output_W.print_output(sol,M.domainB,pdb,ioctrl,sc,ctrl,ts=it_outer)
    if ctrl.van_keken == 1:
        from .output import _benchmark_van_keken
        _benchmark_van_keken(sol,ioctrl,sc)
    
    # Destroy KSP
    energy_global.solv.ksp.destroy()
    lithostatic_pressure_global.solv.ksp.destroy()
    slab.solv.ksp.destroy()
    wedge.solv.ksp.destroy()
    # Ahaha! 
    
    return 0 
#------------------------------------------------------------------------------------------------------------

@timing_function
def time_dependent_solution(M:Mesh, ctrl:NumericalControls, lhs_ctrl:ctrl_LHS, pdb:PhaseDataBase, ioctrl:IOControls, sc:Scal):
    from .phase_db import PhaseDataBase
    from .phase_db import _generate_phase
    from .thermal_structure_ocean import compute_initial_LHS
    from .scal import Scal 
    
    
    
    print_ph(f'// -- // --- Time dependent temperature // -- // --- > ')

    element_p           = M.element_p#basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = M.element_PT#basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = M.element_V#basix.ufl.element("Lagrange","triangle",2,shape=(2,))

    #==================== Phase Parameter ====================
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)  
          
    # Define Problem
    
    # Pressure 
    
    energy_global               = Global_thermal (M = M, name = ['energy','domainG']  , elements = (element_PT,                   )  ,  pdb = pdb, ctrl = ctrl                  )
    
    lithostatic_pressure_global = Global_pressure(M = M, name = ['pressure','domainG'], elements = (element_PT,                     ), pdb = pdb                                ) 
    
    wedge                       = Wedge          (M = M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT  ), pdb = pdb                                )
    
    slab                        = Slab           (M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p,element_PT  ), pdb = pdb                                )
    
    g = fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Define Solution 
    sol                         = Solution()
     
    sol.create_function(lithostatic_pressure_global,slab,wedge,[element_V,element_p])
    
    # Initial temperature field
    sol.T_O = energy_global.initial_temperature_field(M.domainG, ctrl, lhs_ctrl,M.g_input)
    
    # Initial new temperature guess
    sol.T_N = sol.T_O.copy()
    
    ts = 0 
    time = 0.0
    dt   = ctrl.dt
    sol.T_O = sol.T_N.copy()
    output   = OUTPUT(M.domainG.mesh, ioctrl, ctrl, sc)
    save = 0 
    dt_save = 0.5*sc.scale_Myr2sec/sc.T  # Save every 0.5 Myr
    
    if (ctrl.adiabatic_heating != 0) & (it_outer==0) :
        sol = initial_adiabatic_lithostatic_thermal_gradient(sol,lithostatic_pressure_global,pdb,M,g,0,ctrl)
    
    while time<ctrl.time_max:
        time_A_ts = timing.time()

        it_outer = 0 

        res = 1.0

        # Update old temperature 
        
        while it_outer < ctrl.it_max and res > 1e-3: 

            time_A_outer = timing.time()
            # Copy the old solution for the residuum computation 

            Tit_outer        = sol.T_O.copy()
            PLold            = sol.PL.copy()
            u_globalold      = sol.u_global.copy()
            p_globalold      = sol.p_global.copy()


            # Solve lithostatic pressure problem
            if lithostatic_pressure_global.typology == 'NonlinearProblem' or it_outer == 0:  
                lithostatic_pressure_global.Solve_the_Problem(sol,ctrl,pdb,M,g,it_outer,ts=0)

            # Interpolate from global to wedge/slab

            sol.t_owedge = interpolate_from_sub_to_main(sol.t_owedge,sol.T_O, M.domainB.cell_par,1)
            sol.p_lwedge = interpolate_from_sub_to_main(sol.p_lwedge,sol.PL, M.domainB.cell_par,1)

            if it_outer == 0 and ts == 0 : #or vel0 != velc: 
                slab.Solve_the_Problem(sol,ctrl,pdb,M,g,sc,it = it_outer,ts=0)

            if wedge.typology == 'NonlinearProblem' or it_outer == 0: #or vel0 != velc:  
                wedge.Solve_the_Problem(sol,ctrl,pdb,M,g,sc,M.g_input,it = it_outer,ts=0)


            # Interpolate from wedge/slab to global
            sol.u_global = interpolate_from_sub_to_main(sol.u_global,sol.u_wedge, M.domainB.cell_par)
            sol.u_global = interpolate_from_sub_to_main(sol.u_global,sol.u_slab, M.domainA.cell_par)

            sol.Hs_global = interpolate_from_sub_to_main(sol.Hs_global,sol.Hs_wedge, M.domainB.cell_par)
            sol.Hs_global = interpolate_from_sub_to_main(sol.Hs_global,sol.Hs_slab, M.domainA.cell_par)

            sol.p_global = interpolate_from_sub_to_main(sol.p_global,sol.p_wedge, M.domainB.cell_par)
            sol.p_global = interpolate_from_sub_to_main(sol.p_global,sol.p_slab, M.domainA.cell_par)

            sol =energy_global.Solve_the_Problem_TD(sol,ctrl,pdb,M,lhs_ctrl,M.g_input,sc,it = it_outer, ts = ts) 

            # Compute residuum 
            res = compute_residuum_outerTD(sol,Tit_outer,PLold,u_globalold,p_globalold,it_outer,sc, time_A_outer)


            it_outer = it_outer + 1
        

        
        # 
        time_B_ts = timing.time()

        print_ph(f'=========== Timestep {ts:d} t = {time*sc.T/sc.scale_Myr2sec:.3f} [Myr], in {time_B_ts-time_A_ts:.1f} sec // -- // --->')
        
        ts = ts + 1 
        time = time + dt
        
        if (ts == 1) or (ts==0) or np.floor(time/dt_save) != save: 
            output.print_output(sol,M.domainG,pdb,ioctrl,sc,ctrl,ts=ts,time=time*sc.T/sc.scale_Myr2sec)
            save  = np.floor(time/dt_save)
        sol.T_O = sol.T_N.copy()
        

    output.close()
        
    # Destroy KSP
    energy_global.solv.ksp.destroy()
    lithostatic_pressure_global.solv.ksp.destroy()
    slab.solv.ksp.destroy()
    wedge.solv.ksp.destroy()
    # Ahaha! 
    
    return 0 

#------------------------------------------------------------------------------------------------------------


def compute_residuum(a,b):
    dx = a.copy()
    
    dx1 = a.copy()
    
    dx.x.array[:]  = b.x.array[:] - a.x.array[:];dx.x.scatter_forward()
    
    dx1.x.array[:]  = b.x.array[:] + a.x.array[:];dx1.x.scatter_forward()
    
    
    res = L2_norm_calculation(dx)/L2_norm_calculation(dx1)
    
    return res

#------------------------------------------------------------------------------------------------------------

def min_max_array(a, vel = False):
    
    if vel == True: 
        a1 = a.sub(0).collapse()
        a2 = a.sub(1).collapse()
        array = np.sqrt(a1.x.array[:]**2 + a2.x.array[:]**2)
    else: 
        array = a.x.array[:]
    
    local_min = np.min(array[:])
    local_max = np.max(array[:])
    
    global_min = a.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN)
    global_max = a.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)
    
    return np.array([global_min, global_max],dtype=np.float64)

#------------------------------------------------------------------------------------------------------------
def compute_residuum_outer(sol,T,PL,u,p,it_outer,sc,tA):
    # Prepare the variables 

    
    
    
    res_u = compute_residuum(sol.u_global,u)
    res_p = compute_residuum(sol.p_global,p)
    res_T = compute_residuum(sol.T_O,T)
    res_PL= compute_residuum(sol.PL,PL)
    
    minMaxU = min_max_array(sol.u_global, vel=True)
    minMaxP = min_max_array(sol.p_global)
    minMaxT = min_max_array(sol.T_O)
    minMaxPL= min_max_array(sol.PL)
    
    # scal back 
    
    minMaxU = minMaxU*(sc.L/sc.T)*sc.scale_vel 
    minMaxP = minMaxP*sc.stress/1e9 
    minMaxT = minMaxT*sc.Temp -273.15
    minMaxPL= minMaxPL*sc.stress/1e9
    
    
    res_total = np.sqrt(res_u**2+res_p**2+res_T**2)
    
    time_B_outer = timing.time()

    print_ph(f'')
    print_ph(f' Outer iteration {it_outer:d} with tolerance {res_total:.3e}, in {time_B_outer-tA:.1f} sec // -- // --->')
    print_ph(f'    []Res velocity       =  {res_u:.3e} [n.d.], max= {minMaxU[1]:.3e}, min= {minMaxU[0]:.3e} [cm/yr]')
    print_ph(f'    []Res Temperature    =  {res_T:.3e} [n.d.], max= {minMaxT[1]:.3f}, min= {minMaxT[0]:.3f} [C]')
    print_ph(f'    []Res pressure       =  {res_p:.3e} [n.d.], max= {minMaxP[1]:.3e}, min= {minMaxP[0]:.3e} [GPa]')
    print_ph(f'    []Res lithostatic    =  {res_PL:.3e}[n.d.], max= {minMaxPL[1]:.3e}, min= {minMaxPL[0]:.3e} [GPa]')
    print_ph(f'    []Res total (sqrt(rp^2+rT^2+rPL^2+rv^2)) =  {res_total:.3e} [n.d.] ')
    print_ph(f'=============================================// -- // --->')
    print_ph(f'')

    
    
    
    
    return res_total 
#------------------------------------------------------------------------------------------------------------
def compute_residuum_outerTD(sol,T,PL,u,p,it_outer,sc,tA):
    # Prepare the variables 

    
    
    
    res_u = compute_residuum(sol.u_global,u)
    res_p = compute_residuum(sol.p_global,p)
    res_T = compute_residuum(sol.T_N,T)
    res_PL= compute_residuum(sol.PL,PL)
    
    minMaxU = min_max_array(sol.u_global, vel=True)
    minMaxP = min_max_array(sol.p_global)
    minMaxT = min_max_array(sol.T_N)
    minMaxPL= min_max_array(sol.PL)
    
    # scal back 
    
    minMaxU = minMaxU*(sc.L/sc.T)*sc.scale_vel 
    minMaxP = minMaxP*sc.stress/1e9 
    minMaxT = minMaxT*sc.Temp -273.15
    minMaxPL= minMaxPL*sc.stress/1e9
    
    
    res_total = np.sqrt(res_u**2+res_p**2+res_T**2)
    
    time_B_outer = timing.time()

    print_ph(f'')
    print_ph(f'             Outer iteration {it_outer:d} with tolerance {res_total:.3e}, in {time_B_outer-tA:.1f} sec // -- // --->')
    print_ph(f'             []Res velocity       =  {res_u:.3e} [n.d.], max= {minMaxU[1]:.3e}, min= {minMaxU[0]:.3e} [cm/yr]')
    print_ph(f'             []Res Temperature    =  {res_T:.3e} [n.d.], max= {minMaxT[1]:.2f}, min= {minMaxT[0]:.2f} [C]')
    print_ph(f'             []Res pressure       =  {res_p:.3e} [n.d.], max= {minMaxP[1]:.3e}, min= {minMaxP[0]:.3e} [GPa]')
    print_ph(f'             []Res lithostatic    =  {res_PL:.3e}[n.d.], max= {minMaxPL[1]:.3e}, min= {minMaxPL[0]:.3e} [GPa]')
    print_ph(f'              =============================================// -- // --->')
    print_ph(f'')

    
    
    
    
    return res_total 

#------------------------------------------------------------------------------------------------------------



