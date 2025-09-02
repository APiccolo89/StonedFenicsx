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
import scal as sc_f 
import basix.ufl
from utils import timing_function, print_ph,time_the_time
from compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX
from create_mesh import Mesh 
from phase_db import PhaseDataBase
import time                          as timing
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block


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



"""
direct_solver = 1 


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


#--------------------------------------------------------------------------------------------------------------
class Solution():
    def __init__(self):
        self.PL       : dolfinx.fem.function.Function 
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
        self.p_lwedge = fem.Function(PW.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_nwedge = fem.Function(PW.FSPT) # same stuff as before, again, this is a nightmare: why the fuck. 
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
        if nl == 1: 
            self.J = fem.petsc.create_matrix(fem.form(J))
            self.r = fem.petsc.create_vector(fem.form(r))
            
        else: 
            self.J = None
            self.r = None 
    
class SolverStokes(): 
    
    def __init__(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0):
        if direct_solver == 1: 
            self.set_direct_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0)
            self.offset = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        elif direct_solver ==0: 
            self.set_iterative_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0)    
    
    
    def set_direct_solver(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0):
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

        self.A = assemble_matrix_block(fem.form(a), bcs=bcs)   ; self.A.assemble()
        self.P = assemble_matrix_block(fem.form(a_p), bcs=bcs) ; self.P.assemble()
        self.b = assemble_vector_block(fem.form(L), a, bcs=bcs); self.b.assemble()
        self.null_vec = self.A.createVecLeft()
        self.nloc_u = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        self.nloc_p = F1.dofmap.index_map.size_local
        self.null_vec.array[:self.nloc_u]  = 0.0
        self.null_vec.array[self.nloc_u:self.nloc_u+self.nloc_p] = 1.0
        self.null_vec.normalize()
        self.nsp = PETSc.NullSpace().create(vectors=[self.null_vec])
        self.A.setNullSpace(self.nsp)

        
        
        
        
        
        

    
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
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
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

    def Solve_the_Problem(self,S,ctrl,pdb,M,g,it=0,ts=0): 
        
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

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        
        if nl == 0: 
            S = self.solve_the_linear(S,a,L) 
        else: 
            S = self.solve_the_non_linear(M,S,ctrl,pdb,g)

        return S 
    
    def solve_the_linear(self,S,a,L,isPicard=0,it=0,ts=0):
        
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
        self.solv.ksp.solve(self.solv.b, x)
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            S.PL.x.array[:] = x.array_r
            return S 
        else:
            return x.array_r
    
    def solve_the_non_linear(self,M,S,ctrl,pdb,g):  
        
        tol = 1e-3  # Tolerance Picard  
        max_it = 10  # Max iteration before Newton
        isPicard = 1 # Flag for the linear solver. 
        tol = 1.0 
        p_k = S.PL.copy() 
        p_k1 = S.PL.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it = 0 
        time_A = timing.time()
        while it < max_it and tol > 1e-3:
            time_ita = timing.time()
            
            if it == 0:
                A,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),pdb,g)
            else: 
                _,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),pdb,g,1)
            
            p_k1.x.array[:] = self.solve_the_linear(S,A,L,1,it,1) 
            
            # L2 norm 
            du.x.array[:]  = p_k1.x.array[:] - p_k.x.array[:];du.x.scatter_forward()
            du2.x.array[:] = p_k1.x.array[:] + p_k.x.array[:];du2.x.scatter_forward()
            tol= L2_norm_calculation(du)/L2_norm_calculation(du2)
            
            time_itb = timing.time()
            print_ph(f'[]   --->L_2 norm is {tol:.3e}, it_th {it:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            p_k.x.array[:] = p_k1.x.array[:]*0.7 + p_k.x.array[:]*(1-0.7)
            
            it = it + 1 
            
        # --- Newton =>         
        F,J = self.set_newton(p_k,getattr(M,'domainG'),S.T_O,g,pdb)
        
        problem = fem.petsc.NonlinearProblem(F, p_k, bcs=self.bc[0], J=J)

        # Newton solver
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "residual"
        solver.rtol = 1e-8
        solver.report = True
        
        n, converged = solver.solve(p_k)
        print(f"[]    ---> Newton iterations: {n}, converged = {converged}")   
        
        S.PL.x.array[:] = p_k.x.array[:]
        local_max = np.max(p_k.x.array[:])
        global_max = M.comm.allreduce(local_max, op=MPI.MAX)
        print_ph(f"// - - - /Global max lithostatic pressure is    : {global_max:.2f}[n.d.]/")
        
        time_B = timing.time()
        print_ph(f'// -- // --- Solution of Lithostatic pressure problem finished in {time_B-time_A:.2f} sec')
        print_ph(f'')

        print_ph("               ")
        print_ph("               _")
        print_ph("               :")
        print_ph("[] - - - -> Lithostatic <- - - - []")
        print_ph("               :")
        print_ph("               _")
        print_ph("                ")

        print_ph(f'')

        
        
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
        
    
    def set_linear_picard(self,u,T,PL,D,pdb,ctrl, a_p = None,it=0, ts = 0):
        
        """
        The problem is always LINEAR when depends on T/P -> Becomes fairly 
        non linear when the temperature is involved. 
        """
        
        u, p  = self.trial0, self.trial1
        v, q  = self.test0,  self.test1
        dx    = ufl.dx

        eta = fem.Constant(D.mesh, PETSc.ScalarType(float(pdb.eta[0])))

        a1 = ufl.inner(2*eta*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
        a2 = - ufl.inner(ufl.div(v), p) * dx             # build once
        a3 = - ufl.inner(q, ufl.div(u)) * dx             # build once
        a_p0 = ufl.inner(q, p) * dx                      # pressure mass (precond)

        f  = fem.Constant(D.mesh, PETSc.ScalarType((0.0,)*D.mesh.geometry.dim))
        f2 = fem.Constant(D.mesh, PETSc.ScalarType(0.0))
        L  = fem.form([ufl.inner(f, v)*dx, ufl.inner(f2, q)*dx])    
    
        return a1, a2, a3 , L , a_p0       

#----------------------------------------------------------------------------




        
#----------------------------------------------------------------------------          
def compute_strain_rate(u):
    
    e = ufl.sym(ufl.grad(u))
    
    return e  

def compute_eII(e):
    e_II  = sqrt(0.5*inner(e, e) + 1e-15)    
    return e_II
#---------------------------------------------------------------------------
        
def L2_norm_calculation(f):
    return fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx))**0.5       
        
        
#-----------------------------------------------------------------
class Wedge(Stokes_Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
 # Needed for accounting the pressure. 
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
                
    
    def compute_nitsche_FS(self,D, S, dS, a1,a2,a3,pdb ,gamma,it = 0):
        """
        Compute the Nitsche free slip boundary condition for the slab problem.
        This is a placeholder function and should be implemented with the actual Nitsche    method.
        """
        def tau(eta, u):
            return 2 * eta * ufl.sym(ufl.grad(u))
        
        
        # Linear 
        e   = compute_strain_rate(S.u_slab)   
        eta = pdb.eta[0]#compute_viscosity_FX(e,S.t_oslab,S.p_lslab,pdb,D.phase,D)
        
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
    
    def Solve_the_Problem(self,S,ctrl,pdb,M,g,it=0,ts=0):
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
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_slab,S.t_oslab,S.p_lslab,M,pdb,ctrl)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl,M,theta) 
        # Set Nietsche FS boundary condition 
        dS_top = M.bc_dict["top_subduction"]
        dS_bot = M.bc_dict["bot_subduction"]
        # 1 Extract ds 
        a1,a2,a3 = self.compute_nitsche_FS(M, S, dS_bot, a1, a2 ,a3,pdb ,gamma=50.0)
        
        if ctrl.slab_bc == 0: 
            a1,a2,a3 = self.compute_nitsche_FS(M, S, dS_top, a1, a2 ,a3,pdb ,gamma=50.0)
        
        # Slab problem is ALWAYS LINEAR
        
        a   = fem.form([[a1, a2],[a3, None]])
        a_p0  = fem.form([[a1, None],[None, a_p]])

        #block_direct_solver(a, a_p, L,  self.bc, self.F0, self.F1, M.mesh)

        #u,p = block_iterative_solver(a, a_p0, L, self.bc, V_subs, p_subs, M.mesh,ctrl)

        self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,ctrl ,J = None, r = None,it = 0, ts = 0)
        
        
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
            S.u_slab.x.array[:] = u.x.array[:]
            S.p_slab.x.array[:] = p.x.array[:]
        else: 
            S.u_slab.x.array[:self.solv.offset] = x.array_r[:self.solv.offset]
            S.p_slab.x.array[: (len(x.array_r) - self.solv.offset)] = x.array_r[self.solv.offset:]
        


        return S 
    
        


def set_ups_problem():
    from phase_db import PhaseDataBase
    from phase_db import _generate_phase
    from thermal_structure_ocean import compute_initial_LHS
    from scal import Scal 
    import scal as sc_f 
    from numerical_control import IOControls,NumericalControls,ctrl_LHS
    
    from create_mesh import unit_test_mesh
    # Create scal 
    sc = Scal(L=660e3,Temp = 1350,eta = 1e21, stress = 1e9)
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    # Create mesh 
    M = unit_test_mesh(ioctrl, sc)
            
    print_ph("[] - - - -> Creating numerical controls <- - - - []")
    ctrl = NumericalControls()
    
    ctrl = sc_f._scaling_control_parameters(ctrl, sc)
    
    print_ph("[] - - - -> Phase Database <- - - - []")    
        
    # This function can be a canvas for the main solver update. 
    # Define elements -> would be defined in the controls 

    element_p           = basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = basix.ufl.element("Lagrange","triangle",2,shape=(2,))
    
    #===============    
    # Define PhaseDataBase
    pdb = PhaseDataBase(6)
    # Slab
    
    pdb = _generate_phase(pdb, 1, rho0 = 3300 , option_rho = 3, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)
    # Oceanic Crust
    
    pdb = _generate_phase(pdb, 2, rho0 = 2900 , option_rho = 3, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)
    # Wedge
    
    pdb = _generate_phase(pdb, 3, rho0 = 3300 , option_rho = 3, option_rheology = 3, option_k = 0, option_Cp = 0, eta=1e22)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # 
    
    pdb = _generate_phase(pdb, 4, rho0 = 3250 , option_rho = 3, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 5, rho0 = 2800 , option_rho = 3, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 6, rho0 = 2700 , option_rho = 3, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = sc_f._scaling_material_properties(pdb,sc)
    # Define LHS 

    #lhs_ctrl = ctrl_LHS()

    #lhs_ctrl = sc_f._scale_parameters(lhs_ctrl, sc)
    
    #lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)  
          
    # Define Problem
    
    # Pressure 
    
    lithostatic_pressure_global = Global_pressure(M = M, name = ['pressure','domainG'], elements = (element_PT,                   ), pdb = pdb)
    
    wedge                       = Wedge          (M = M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT), pdb = pdb)
    
    slab                        = Slab           (M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p,element_PT), pdb = pdb)
    
    g = fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Define Solution 
    sol                         = Solution()
    
    sol.create_function(lithostatic_pressure_global,slab,wedge,[element_V,element_p])
    
    #lithostatic_pressure_global.Solve_the_Problem(sol,ctrl,pdb,M,g,it=0,ts=0)
    
    slab.Solve_the_Problem(sol,ctrl,pdb,M,g,it=0,ts=0)
    
    
    
    # Update and set up variational problem
    
    # non-linear itearion betwee pressure/stk/temperature 
    #===============
    # -> [a] Solve Pressure: newton picard+newton
    #        [/] interpolate solutions from global to local 
    #    [b] Solve Slab if (it == 0 or time_step == 0) & vel_time == 1 (mix Picard & Newton)=> iterate picard till 1e-3 -> try newton  
    #    [c] Solve Wedge if (it==0  or time_step == 0) & (rheo == nonlinear or vel_time == 1)
    #        [/] interpolate solutions from local to global 
    #    [d] Solve thermal problem -> steady state / time dependent
    #    [e] Check residuum of PL,T,u(G),p(G) -> if variation of norm(dT,dPL,du(G),dp(G))<tol -> iteration finished 
    # ==============         
    #    [Output] : interpoalte solution variable DG1 -> save output -> Extract additional data 
    #             : density,k,Cp,viscosity, strain rate, pressure 
    #    [->done<-] Create relational database, {Here I need to do a decent job, for showing off my skills to potential hiring, like the manager of an obnoxious McDonald}
    # -------------
    
    
    

    
    return 0 

def block_operators(a, a_p, L, bcs, V, Q):
    """Return block operators and block RHS vector for the Stokes
    problem"""
    A = assemble_matrix_block(a, bcs=bcs); A.assemble()
    P = assemble_matrix_block(a_p, bcs=[]); P.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # nullspace vector [0_u; 1_p] locally
    null_vec = A.createVecLeft()
    nloc_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    nloc_p = Q.dofmap.index_map.size_local
    null_vec.array[:nloc_u]  = 0.0
    null_vec.array[nloc_u:nloc_u+nloc_p] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    A.setNullSpace(nsp)

    return A, P, b

def block_iterative_solver(a, a_p, L, bcs, V, Q, msh,ctrl):
    """Solve the Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, P, b = block_operators(a, a_p, L, bcs, V, Q)

    # Build PETSc index sets for each field (global dof indices for each
    # field)
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    bs_u  = V.dofmap.index_map_bs  # = mesh.dim for vector CG
    nloc_u = V_map.size_local * bs_u
    nloc_p = Q_map.size_local

    # local starts in the *assembled block vector*
    offset_u = 0
    offset_p = nloc_u

    is_u = PETSc.IS().createStride(nloc_u, offset_u, 1, comm=PETSc.COMM_SELF)
    is_p = PETSc.IS().createStride(nloc_p, offset_p, 1, comm=PETSc.COMM_SELF)


    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setTolerances(rtol=1e-10)
    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # The matrix A combined the vector velocity and scalar pressure
    # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
    # cannot infer the correct near-nullspace from the matrix block
    # size. Therefore, we set block size on the top-left block of the
    # preconditioner so that GAMG can infer the appropriate near
    # nullspace.
    ksp.getPC().setUp()
    Pu, _ = ksp_u.getPC().getOperators()
    Pu.setBlockSize(msh.topology.dim)

    # Create a block vector (x) to store the full solution and solve
    x = A.createVecRight()
    ksp.solve(b, x)

    xu = x.getSubVector(is_u)
    xp = x.getSubVector(is_p)
    u, p = fem.Function(V), fem.Function(Q)
    
    u.x.array[:] = xu.array_r
    p.x.array[:] = xp.array_r
    
    u.name, p.name = "Velocity", "Pressure"
    

        


    return  u, p


def tau(eta, u):
    return 2 * eta * ufl.sym(ufl.grad(u))

def compute_nitsche_FS(mesh, eta, tV, TV, tP, TP, dS, a1, a2, a3, gamma=100.0):
    """
    Compute the Nitsche free slip boundary condition for the slab problem.
    This is a placeholder function and should be implemented with the actual Nitsche method.
    """
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)

    a1 += (
        - ufl.inner(tau(eta, tV), ufl.outer(ufl.dot(TV, n) * n, n)) * dS
        - ufl.inner(ufl.outer(ufl.dot(tV, n) * n, n), tau(eta, TV)) * dS
        + (2 * eta * gamma / h)
        * ufl.inner(ufl.outer(ufl.dot(tV, n) * n, n), ufl.outer(TV, n))
        * dS
    )
    a2 += ufl.inner(tP, ufl.dot(TV, n)) * dS
    a3 += ufl.inner(TP, ufl.dot(tV, n)) * dS

    return a1, a2, a3


@timing_function    
def block_direct_solver(a, a_p, L, bcs, V, Q, mesh):
    """Solve the Stokes problem using blocked matrices and a direct
    solver."""

    # Assembler the block operator and RHS vector
    A, _, b = block_operators(a, a_p, L, bcs, V, Q)

    # Create a solver
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Set the solver type to MUMPS (LU solver) and configure MUMPS to
    # handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    use_superlu = PETSc.IntType == np.int64
    if PETSc.Sys().hasExternalPackage("mumps") and not use_superlu:
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    else:
        pc.setFactorSolverType("superlu_dist")

    # Create a block vector (x) to store the full solution, and solve
    x = A.createVecLeft()
    ksp.solve(b, x)

    # Create Functions and scatter x solution
    u, p = Function(V), Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

    # Compute the $L^2$ norms of the u and p vectors


    return u,p

if __name__ == '__main__': 
    
    set_ups_problem()