import sys, os
sys.path.insert(0, "/Users/wlnw570/Work/Leeds/Fenics_tutorial/Stonedphoenix")
import gmsh 
import meshio
from mpi4py                          import MPI
from petsc4py                        import PETSc
from dolfinx                         import mesh, fem, io, nls, log
from dolfinx.fem.petsc               import NonlinearProblem
from dolfinx.nls.petsc               import NewtonSolver
from dolfinx.io                      import XDMFFile, gmshio
from ufl                             import exp, conditional, eq, as_ufl
from src.create_mesh                 import Mesh
from src.numerical_control           import NumericalControls
from src.numerical_control           import IOControls 
from src.solution                    import Solution 
from src.compute_material_property   import density_FX
import ufl

from   src.scal                      import Scal
from   src.create_mesh               import Mesh
from   src.phase_db                  import PhaseDataBase
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import matplotlib.pyplot             as plt
import compute_material_property     as cmp 
import src.scal                      as sc_f 

def _solve_lithostatic_pressure(M : Mesh , S : Solution ,ctrl : NumericalControls, ioctrl : IOControls , pdb : PhaseDataBase, sc : Scal):

    x                = ufl.SpatialCoordinate(M.mesh)
        

    # Define the lithostatic pressure function
    P                = fem.Function(M.PT)
    q                = ufl.TestFunction(M.PT) # Trial function for the nonlinear solver
    phase            = M.phase
    T                = M.T_i   
    
    # Set the boundary conditions for this specific problem
    fdim = M.mesh.topology.dim - 1    
    top_facets   = M.mesh_Ftag.find(1)
    top_dofs    = fem.locate_dofs_topological(M.PT, M.mesh.topology.dim-1, top_facets)
    bc = fem.dirichletbc(0.0, top_dofs, M.PT)

    
    g = fem.Constant(M.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    
    F = ufl.dot(ufl.grad(P), ufl.grad(q)) * ufl.dx - ufl.dot(ufl.grad(q), density_FX(pdb, T, P, M.phase,M.mesh)*g) * ufl.dx

    problem = NonlinearProblem(F, P, bcs=[bc])

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-9
    solver.report = True


    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.INFO)
    n, converged = solver.solve(P)
    
    
    pr = P.x.array
    pr = pr * sc.stress  # Scale the pressure to the correct units
    print(f"Scaled lithostatic pressure: {np.min(pr/1e9):.2f} GPa, {np.max(pr/1e9):.2f} GPa")
    
    rho = fem.Function(ph)
    v = ufl.TestFunction(ph)
    rho_trial = ufl.TrialFunction(ph)
    rho_expr = density(phase, P, T,sc)

    a = ufl.inner(rho_trial, v) * ufl.dx
    L = ufl.inner(rho_expr, v) * ufl.dx
    fem.petsc.LinearProblem(a, L, bcs=[], u=rho).solve()
    cell_density = rho.x.array*sc.rho  # Scale the density to the correct units
    print(f"Scaled density: {np.min(cell_density):.2f} kg/m^3, {np.max(cell_density):.2f} kg/m^3")
    
    print(f"Number of interations: {n:d}")

    
    return S


def unit_test_pressure():
    
    from phase_db import PhaseDataBase
    from phase_db import _generate_phase
    
    from create_mesh import unit_test_mesh
    # Create scal 
    sc = Scal(L=660e3,Temp = 1350,eta = 1e21, stress = 1e9)
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    
    # Create mesh 
    M = unit_test_mesh(ioctrl, sc)
        
    print('mesh done')
    
    # Create initial temperature field
    
    M.T_i.x.array[:] = 1300/sc.Temp
    
    # Create Controls 
    
    ctrl = NumericalControls()
    
    ctrl = sc_f._scaling_control_parameters(ctrl, sc)
    
    # Create phases array 
    
    
    pdb = PhaseDataBase(8)
    pdb = _generate_phase(pdb, 1, rho0 = 3300 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 2, rho0 = 2900 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 3, rho0 = 3300 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 4, rho0 = 3250 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 5, rho0 = 2800 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 6, rho0 = 2700 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 7, rho0 = 3250 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    pdb = _generate_phase(pdb, 8, rho0 = 3250 , option_rho = 2, option_rheology = 4, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')

    pdb = sc_f._scaling_material_properties(pdb,sc)
    
    # call the lithostatic pressure 
    
    S   = Solution()
    
    S   = _solve_lithostatic_pressure(M, S, ctrl, ioctrl, pdb, sc)
    
    # create vtk file and test wheter or not is good 
    pass



if __name__ == '__main__':

    unit_test_pressure()
    
    print('passed')

