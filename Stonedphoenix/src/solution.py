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
    

        
        
        
    return 0

def form_stokes_variational_problem():
    
    
    
    return a,L,r 

def solve_linear_stokes(S): 
    
    
    
    return S