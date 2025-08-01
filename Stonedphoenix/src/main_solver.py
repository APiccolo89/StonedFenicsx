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
from src.numerical_control           import NumericalControls, ctrl_LHS
from src.numerical_control           import IOControls 
from src.solution                    import Solution 
from src.compute_material_property   import density_FX
from src.compute_material_property   import heat_conductivity_FX
from src.compute_material_property   import heat_capacity_FX

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
import basix.ufl
import time                          as timing

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_discrete_colormap(n_colors, base_cmap='viridis'):
    """
    Create a discrete colormap with `n_colors` from a given base colormap.
    
    Parameters:
        n_colors (int): Number of discrete colors.
        base_cmap (str or Colormap): Name of the matplotlib colormap to base on.
    
    Returns:
        matplotlib.colors.ListedColormap: A discrete version of the colormap.
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_colors))
    return mcolors.ListedColormap(color_list, name=f'{base_cmap}_{n_colors}')

def initial_temperature_field(X, ph, M, ctrl, lhs):
    from scipy.interpolate import griddata
    from ufl import conditional, Or, Eq
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
    T_i_A = fem.Function(X)
    cd_dof = X.tabulate_dof_coordinates()
    T_i_A.x.array[:] = griddata(-lhs.z, lhs.LHS, c_dofs[:,1], method='nearest')
    T_i_A.x.scatter_forward() 
    #- 
    T_gr = (-M.g_input.lt_d-0)/(ctrl.Tmax-ctrl.Ttop)
    T_gr = T_gr**(-1) 
    
    T_expr = fem.Function(X)
    ind_A = np.where(cd_dof[:,1] >= -M.g_input.lt_d)[0]
    ind_B = np.where(cd_dof[:,1] < -M.g_input.lt_d)[0]
    T_expr.x.array[ind_A] = ctrl.Ttop + T_gr * cd_dof[ind_A,1]
    T_expr.x.array[ind_B] = ctrl.Tmax
    T_expr.x.scatter_forward()
        

    expr = conditional(
        reduce(Or,[eq(phase, i) for i in [2, 3, 4, 5]]),
        T_expr,
        T_i_A
    )
    
    v = ufl.TestFunction(X)
    u = ufl.TrialFunction(X)
    T_i = fem.Function(X)
    a = u * v * ufl.dx 
    L = expr * v * ufl.dx
    prb = fem.petsc.LinearProblem(a,L,u=T_i)
    prb.solve()
    return T_i 


def set_lithostatic_problem(P, T, q ,  pdb, sc, g, M):
    
    F = ufl.dot(ufl.grad(P), ufl.grad(q)) * ufl.dx - ufl.dot(ufl.grad(q), density_FX(pdb, T, P, M.phase,M.mesh)*g) * ufl.dx
    
    return F


def set_steady_state_thermal_problem(P, T, vel ,q ,  pdb, sc, M):
    
    
    
    adv  = ufl.inner( density_FX(pdb, T, P, M.phase,M.mesh) * heat_capacity_FX(pdb, T, M.phase,M.mesh) * vel,  ufl.grad(T)) * q * ufl.dx 
    cond = ufl.inner( heat_conductivity_FX(pdb, T, P, M.phase,M.mesh) * ufl.grad(T), ufl.grad(q)) * q * ufl.dx
    
    F = adv + cond
    
    return F


def strain_rate(vel):
    
    return ufl.sym(ufl.grad(vel))
    
def eps_II(vel):
 
    e = strain_rate(vel)
    
    eII = ufl.sqrt(2 * ufl.inner(e,e)) 
    
    # Portion of the model do not have any strain rate, for avoiding blow up, I just put a fictitious low strain rate
    
    eII = conditional(
        eq(e, 0.0), 1e-18,
    )
    
    
    return eII 

def compute_sigma():
    pass



def compute_nitsche_BC():



    pass 
    



def set_Stokes_equationset_stokes(PL,T_on,pdb,sc,g,M):

    flag_linear = 'False'
    if np.all(pdb.opt_eta) == 0: 
        flag_linear = 'True':
    
    if flag_linear == 'True'


        # BiLinear 
    
    
    
        # Linear
    

        # Solver 
    
    
        # -> output the field that are relevant 






    
    pass 

def main_solver_steady_state(M, S, ctrl, pdb, sc ): 
    # Segregate solvers seems the most reasonable solution -> I can use also a switch between the options, such that I can use linear/non linear solver 
    # -> BC -> 
    
    
    # Create mixed function space 
    element_p       = basix.ufl.element("DG", "triangle", 0) 
    element_PT      = basix.ufl.element("Lagrange","triangle",2)
    element_V       = basix.ufl.element("Lagrange","triangle",2,shape=(M.mesh.geometry.dim,))
    
    mixed_el        = basix.ufl.mixed_element([element_V,element_p,element_PT, element_PT])
    
    Sol_space       = fem.functionspace(M.mesh,mixed_el)
    
    sol_spaceV,  _  = Sol_space.sub(0).collapse()
    sol_spacep,  _  = Sol_space.sub(1).collapse()
    sol_spaceT,  _  = Sol_space.sub(2).collapse()
    sol_spacePl, _  = Sol_space.sub(3).collapse()
    
    
    tV , tP , ten, tpl     = ufl.TrialFunctions(Sol_space)
    TV , TP , Ten, Tpl     = ufl.TestFunctions(Sol_space)
    # -- 
    T_on        = fem.Function(sol_spaceT)
    T_on.interpolate(M.T_i)
    # -- 
    PL          = fem.Function(sol_spacePl)
    vel         = fem.Function(sol_spaceV)
    p           = fem.Function(sol_spacep)
    

    # -- 
    
    g = fem.Constant(M.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Boundary condition 
    # Define 
    
    # Set the lithostatic problem 
    # -> set boundary condition [To do ]
    
    # -> Produce non linear 
    
    F_l = set_lithostatic_problem(PL, T_on, Tpl, pdb, sc, g, M )
    
    F_S = set_stokes(PL,T_on,pdb,sc,g,M )
    
    # Set the temperature problem 
    F_T = set_steady_state_thermal_problem(PL, T_on, vel, Ten ,pdb , sc, M)
    
    # Set the Stokes equation
    
    

    
    return 0 

def unit_test(): 
    
    
    from phase_db import PhaseDataBase
    from phase_db import _generate_phase
    from thermal_structure_ocean import compute_initial_LHS

    
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
    

    
    pdb = PhaseDataBase(8)
    # Slab
    
    pdb = _generate_phase(pdb, 1, rho0 = 3300 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e22, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # Oceanic Crust
    
    pdb = _generate_phase(pdb, 2, rho0 = 2900 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # Wedge
    
    pdb = _generate_phase(pdb, 3, rho0 = 3300 , option_rho = 2, option_rheology = 3, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # 
    
    pdb = _generate_phase(pdb, 4, rho0 = 3250 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e23, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 5, rho0 = 2800 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 6, rho0 = 2700 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 7, rho0 = 3250 , option_rho = 2, option_rheology = 3, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 8, rho0 = 3250 , option_rho = 2, option_rheology = 3, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')

    pdb = sc_f._scaling_material_properties(pdb,sc)
    
    timeA = timing.time()
    lhs_ctrl = ctrl_LHS()

    lhs_ctrl = sc_f._scale_parameters(lhs_ctrl, sc)
    
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)
    timeB = timing.time()
    print('Time is %.3f sec'%(timeB-timeA))
    
    # call the lithostatic pressure 
    
    S   = Solution()
    
    main_solver_steady_state(M, S, ctrl, pdb, sc )
    
    pass 
    

if  __name__ == '__main__':
    unit_test()