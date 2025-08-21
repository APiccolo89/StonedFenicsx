import numpy as np
import gmsh 

import ufl
import meshio
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile, gmshio
import gmsh 
from ufl import exp, conditional, eq, as_ufl
import scal as sc_f 
import basix.ufl
from utils import timing_function, print_ph
from compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX

#--------------------------------------------------------------------------------------------------------------
class Solution():
    def __init__(self):
        self.PL      : dolfinx.fem.function.Function 
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
        
        
    def create_function(self,M): 
        
        def gives_Function(space):
            Va = space.sub(0)
            Pa = space.sub(1)
            V,_  = Va.collapse()
            P,_  = Va.collapse()
            a = fem.Function(V)
            b = fem.Function(P)
            return a,b 
        
        self.PL       = fem.Function(M.Sol_SpaceT)
        self.T_O      = fem.Function(M.Sol_SpaceT)
        self.T_N      = fem.Function(M.Sol_SpaceT)
        self.p_lwedge = fem.Function(M.domainB.solPT)
        self.t_owedge = fem.Function(M.domainB.solPT)
        self.p_lslab  = fem.Function(M.domainA.solPT)
        self.t_oslab  = fem.Function(M.domainA.solPT)
        self.u_global, self.p_global = gives_Function(M.Sol_SpaceSTK)
        self.u_slab  , self.p_slab   = gives_Function(M.domainA.solSTK)
        self.u_wedge , self.p_wedge  = gives_Function(M.domainB.solSTK)

        return self 
#---------------------------------------------------------------------------
class Solver():
    """
    Solver class: it is call for setting up the problem, 
    """
    
    def __init__(self,Linear,NonLinear): 
        self.Linear    = NonLinear
        self.NonLinear = Linear 
    
    def 
    
         
        
class Problem(): 
        name     : list                               # name of the problem, domain [global, domainA...]
        mixed    : bool                               # is a mixed problem (e.g. Stokes problem has two function spaces: velocity and pressure)
        FS       : dolfinx.fem.function.FunctionSpace # Function space of the problem 
        F0       : dolfinx.fem.function.FunctionSpace # Function space of the subspace 
        F1       : dolfinx.fem.function.FunctionSpace # Function space of the subspace
        trial0   : ufl.argument.Argument              # Trial 
        trial1   : ufl.argument.Argument              # Trial
        test0    : ufl.argument.Argument              # Test
        test1    : ufl.argument.Argument              # Test 
        bilinearF: ufl.form.Form                      # Bilinearform of the problem
        linearF  : ufl.form.Form                      # Linear form of the problem,
        type     : str                                # Linear/Non Linear -> decide on the phase database 
        dofs     : np.int32                           # List of array [[tag_bc, type,array_dofs],....]
        bc       : list                               # List of dirichlecht bc 
        J        : ufl.form.Form                      # Jacobian in case non newtonian and newton solver 
    # -- 
    def __init__(self, M:MESH, elements:touple, name:list)->self:
        """
        Arguments: 
        self    : the class its self 
        M       : the mesh object 
        elements: touple containing the elements if they are more than 1 -> it assumes that the problem is mixed and populate accordingly 
                  the trial and test function. 
                  if elements == 1 -> FS -> trial0,test0 are going to use to form the problem 
        -> dofs and type of boundary conditions will be populated accordingly as a function of the subclasses that are created out of this superclass 
        name    : list ['nameoftheproblem', 'domain']
        """
        self.name = name 
        if name[1] != 'global':  
            M = getattr(M,name[1]) # extract the subdomain 
        elif name[1] != 'domainA' and name[1] != 'domainB' and name[1] != 'domainC':
            raise NameError('Wrong domain name')
        elif name[1] == 'domainC':
            printph('Are you sure? DomainC for this problem is basically junk for and is solved in thermal-pressure_lit -> stokes should not be used there')
        if len(elements) == 1: 
            self.FS       = dolfinx.fem.functionspace(M.mesh,elements[0]) 
            self.trial0   = ufl.TrialFunction(self.FS)
            self.test0    = ufl.TrialFunction(self.FS)
        if len(elements) >1: 
            mixed_element = basix.ufl.mixed_element([elements[0],elements[1]])
            self.FS       = dolfinx.fem.functionspace(M.mesh,mixed_elemet)
            self.F0       = self.FS.sub(0).collapse()
            self.F1       = self.FS.sub(1).collapse()
            self.trial0   = ufl.TrialFunction(self.F0)
            self.test0    = ufl.TrialFunction(self.F0)
            self.trial1   = ufl.TrialFunction(self.F1)
            self.test1    = ufl.TrialFunction(self.F1)
        
        return self                    
#------------------------------------------------------------------
class Global_thermal(Problem):
    def __init__(self,M:MESH, elements:tuple, name:list,S:Solution,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
#-----------------------------------------------------------------
class Global_pressure(Problem): 
    def __init__(self,M:MESH, elements:tuple, name:list,S:Solution,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
#-----------------------------------------------------------------
class Wedge(Problem): 
    def __init__(self,M:MESH, elements:tuple, name:list,S:Solution,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
#------------------------------------------------------------------
class Slab(Problem): 
    def __init__(self,M:MESH, elements:tuple, name:list,S:Solution,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
#-------------------------------------------------------------------
@timing_function
def solve_lithostatic_problem(S, pdb, sc, g, M ):
    """
    PL  : function
    T_o : previous Temperature field

    pdb : phase data base 
    sc  : scaling 
    g   : gravity vector 
    M   : Mesh object 
    --- 
    Output: current lithostatic pressure. 
    
    To do: Improve the solver options and make it faster
    create an utils function for timing. 
    
    """
    print_ph("[] - - - -> Solving Lithostatic pressure problem <- - - - []")

    tPL = ufl.TrialFunction(M.Sol_SpaceT)  
    TPL = ufl.TestFunction(M.Sol_SpaceT)
    
    flag = 1 
    fdim = M.mesh.topology.dim - 1    
    top_facets   = M.mesh_Ftag.find(1)
    top_dofs    = fem.locate_dofs_topological(M.Sol_SpaceT, M.mesh.topology.dim-1, top_facets)
    bc = fem.dirichletbc(0.0, top_dofs, M.Sol_SpaceT)
    
    # -> yeah only rho counts here. 
    if (np.all(pdb.option_rho) == 0):
        flag = 0
        bilinear = ufl.dot(ufl.grad(TPL), ufl.grad(tPL)) * ufl.dx
        linear   = ufl.dot(ufl.grad(TPL), density_FX(pdb, S.T_O, S.PL, M.phase,M.mesh)*g) * ufl.dx
        problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"})
        
        S.PL = problem.solve()
                
    
    if flag !=0: 
        # Solve lithostatic pressure - Non linear 
        F = ufl.dot(ufl.grad(S.PL), ufl.grad(TPL)) * ufl.dx - ufl.dot(ufl.grad(TPL), density_FX(pdb, S.T_O, S.PL, M.phase,M.mesh)*g) * ufl.dx

        problem = fem.petsc.NonlinearProblem(F, S.PL, bcs=[bc])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-4
        solver.report = True
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}ksp_rtol"] = 1.0e-4
        ksp.setFromOptions()
        n, converged = solver.solve(PL)
    local_max = np.max(S.PL.x.array)

    # Global min and max using MPI reduction
    global_max = M.comm.allreduce(local_max, op=MPI.MAX)
    print_ph(f"// - - - /Global max lithostatic pressure is    : {global_max*sc.stress/1e9:.2f}[GPa]/")
    print_ph("               ")
    print_ph("               _")
    print_ph("               :")
    print_ph("[] - - - -> Finished <- - - - []")
    print_ph("               :")
    print_ph("               _")
    print_ph("                ")

    from utils import interpolate_from_sub_to_main
    
    S.p_lwedge = interpolate_from_sub_to_main(S.p_lwedge,S.PL,M.domainB.scell,1)
    S.p_lslab  = interpolate_from_sub_to_main(S.p_lslab,S.PL,M.domainA.scell,1)

    return S  

#---------------------------------------------------------------------------
def solution_stokes_problem(Sol,M,ctrl,bc):
    
    asl,Lsl,rsl,bcsl = form_stokes_variational_problem(D, S, ctrl, pdb)
    awe,Lwe,rwe,bcwe = form_stokes_variational_problem(D, S, ctrl, pdb)
    
    

        
        
        
    return 0

def form_stokes_variational_problem(D, S, ctrl, pdb):
    
    
    
    return a,L,r 

def solve_linear_stokes(S): 
    
    
    
    return S