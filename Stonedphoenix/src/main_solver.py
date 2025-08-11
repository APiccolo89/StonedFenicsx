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
from utils import timing_function, print_ph


#---------------------------------------------------------------------------
def get_discrete_colormap(n_colors, base_cmap='viridis'):
    """
    Create a discrete colormap with `n_colors` from a given base colormap.
    
    Parameters:
        n_colors (int): Number of discrete colors.
        base_cmap (str or Colormap): Name of the matplotlib colormap to base on.
    
    Returns:
        matplotlib.colors.ListedColormap: A discrete version of the colormap.
        copied from internet
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_colors))
    return mcolors.ListedColormap(color_list, name=f'{base_cmap}_{n_colors}')
#---------------------------------------------------------------------------

@timing_function
def initial_temperature_field(M, ctrl, lhs):
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
    X     = M.Sol_SpaceT 
    T_i_A = fem.Function(X)
    cd_dof = X.tabulate_dof_coordinates()
    T_i_A.x.array[:] = griddata(-lhs.z, lhs.LHS, cd_dof[:,1], method='nearest')
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
        reduce(Or,[eq(M.phase, i) for i in [2, 3, 4, 5]]),
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

#---------------------------------------------------------------------------

@timing_function
def set_lithostatic_problem(PL, T_o, tPL, TPL, pdb, sc, g, M ):
    """
    PL  : function
    T_o : previous Temperature field
    tPL : trial function for lithostatic pressure 
    TPL : test function for lithostatic pressure
    pdb : phase data base 
    sc  : scaling 
    g   : gravity vector 
    M   : Mesh object 
    --- 
    Output: current lithostatic pressure. 
    
    To do: Improve the solver options and make it faster
    create an utils function for timing. 
    
    """
    flag = 1 
    fdim = M.mesh.topology.dim - 1    
    top_facets   = M.mesh_Ftag.find(1)
    top_dofs    = fem.locate_dofs_topological(M.Sol_SpaceT, M.mesh.topology.dim-1, top_facets)
    bc = fem.dirichletbc(0.0, top_dofs, M.Sol_SpaceT)
    
    # -> yeah only rho counts here. 
    if (np.all(pdb.option_rho) == 0):
        flag = 0
        bilinear = ufl.dot(ufl.grad(TPL), ufl.grad(tPL)) * ufl.dx
        linear   = ufl.dot(ufl.grad(TPL), density_FX(pdb, T_o, PL, M.phase,M.mesh)*g) * ufl.dx
        problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"})
        
        PL = problem.solve()
                
    
    if flag !=0: 
        # Solve lithostatic pressure - Non linear 
        F = ufl.dot(ufl.grad(PL), ufl.grad(TPL)) * ufl.dx - ufl.dot(ufl.grad(TPL), density_FX(pdb, T_o, PL, M.phase,M.mesh)*g) * ufl.dx

        problem = fem.petsc.NonlinearProblem(F, PL, bcs=[bc])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-5
        solver.report = True
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}ksp_rtol"] = 1.0e-5
        ksp.setFromOptions()
        n, converged = solver.solve(PL)
    local_max = np.max(PL.x.array)

    # Global min and max using MPI reduction
    global_max = M.comm.allreduce(local_max, op=MPI.MAX)
    print_ph('.  Global max lithostatic pressure is %.2f GPa'%(global_max*sc.stress/1e9))
    return PL 

#---------------------------------------------------------------------------

def set_steady_state_thermal_problem(PL, T_on, tPL, TPL, pdb, sc, g, M ):
    
    
    
    adv  = ufl.inner( density_FX(pdb, T, P, M.phase,M.mesh) * heat_capacity_FX(pdb, T, M.phase,M.mesh) * vel,  ufl.grad(T)) * q * ufl.dx 
    cond = ufl.inner( heat_conductivity_FX(pdb, T, P, M.phase,M.mesh) * ufl.grad(T), ufl.grad(q)) * q * ufl.dx
    
    F = adv + cond
    
    return F

#---------------------------------------------------------------------------

def strain_rate(vel):
    
    return ufl.sym(ufl.grad(vel))
#---------------------------------------------------------------------------

    
def eps_II(vel):
 
    e = strain_rate(vel)
    
    eII = ufl.sqrt(2 * ufl.inner(e,e)) 
    
    # Portion of the model do not have any strain rate, for avoiding blow up, I just put a fictitious low strain rate
    
    
    return eII 
#---------------------------------------------------------------------------


def compute_sigma():
    pass


#---------------------------------------------------------------------------
def linear_stokes_solver(M, pdb, sc, T_on, PL, g,):
    

    
    
    
    
    
    
    
    
    


#---------------------------------------------------------------------------
@timing_fun    
def set_Stokes_Slab(PL,T_on,pdb,sc,g,M):
    """
    Input:
    PL  : lithostatic pressure field
    T_on: temperature field
    pdb : phase data base
    sc  : scaling
    g   : gravity vector
    M   : Mesh object
    Output:
    F   : Stokes problem for the slab/Solution -> still to decide
    ---
    The slab proble is by definition linear. First, we are speaking of an object that moves at constant velocity from top to bottom, by definition, it is not deforming. Secondly
    why introducing non-linear rheology? Would be a waste of time 
    Considering the slab problem linear implies that can be computed only once or potentially whenever the slab velocity changes. For example, if the age of the incoming slab changes, the problem is still linear as the temperature is moving as a function of the velocity field. 
    Then depends wheter or not the velocity field is constantly changing or stepwise changing.
    ---
    """
    #-----
    # Extract the relevant information from the mesh object
    mesh = M.domainA.submesh
    V_subs = M.domainA.solSTK
    p_subs = M.domainA.solPT
    # Trial function for the velocity: naming convection is t(V) -> trial/ and T(V)-> test
    tV = M.domainA.TrialV
    TV = M.domainA.TestV
    # Trial function for the pressure
    tP = M.domainA.TrialP
    TP = M.domainA.TestP
    eta_slab = pdb.eta[0]  # Assuming slab is the first phase in pdb [oceanic crust has its own viscosity, but it is not relevant for the slab problem] Most of the routine that 
    # I created are also for expanding the code to something more complex in the future, for now, remains simple.
    dS = ufl.
    
    # -------
    # Project temperature and lithostatic pressure on the submesh 
    # Form the linear problem for the slab
    # -> 
    # -> Nitsche free slip boundary condition upper slab 
    # -> Nitsche free slip boundary condition lower slab
    # -> Nitche free slip boundary condition for inflow 
    # -> Nitsche free slip boundary condition for outflow     

    
    # Select the subspace of the slab 
    
    # Slab problem is always linear.
        


#---------------------------------------------------------------------------
@timing_fun    
def main_solver_steady_state(M, S, ctrl, pdb, sc, lhs ): 
    """
    To Do explanation: 
    
    
    
    -- 
    Random developing notes: The idea is to solve temperature, and lithostatic pressure for the entire domain and solve two small system for the slab [a pipe of 130 km] and the wedge. from these two small system-> interpolate the velocity into the main mesh and resolve it. 
    --- 
    A. Solve lithostatic pressure: 
    ->whole mesh 
    B. Solve Slab -> function to solve stokes problem -> class for the stokes solver -> bc -> specific function ? 
    C. Solve Wedge -> function to solve stokes problem -> class for the stokes solver -> bc 
    
    D. Merge the velocities field in only one big field -> plug in the temperature solve and solve for the whole mesh
    ----> Slab by definition with a fixed geometry is undergoing to rigid motion 
            -> No deformation -> no strain rate -> constant viscosity is good enough yahy 
            => SOLVE ONLY ONE TIME and whenever you change velocity of the slab. [change age is for free]
    ----> Wedge -> Non linear {T & ε, with/out P_l}/ Linear {T,with/out P_l}
            -> In case of temperature dependency, or pressure dependency -> each time the solution must be computed [each iteration/timestep]
    ---- Crust in any case useless and it is basically there for being evolving only thermally 
    """
    
    
    
    
    
    # Segregate solvers seems the most reasonable solution -> I can use also a switch between the options, such that I can use linear/non linear solver 
    # -> BC -> 
    
    # -- 
    # -- 
    PL          = fem.Function(M.Sol_SpaceT)
    vel         = fem.Function(V_subs)
    p           = fem.Function(p_subs)
    T_o = initial_temperature_field(M, ctrl, lhs)
    # -- Test and trial function for pressure and temperature 
    tT =  ufl.TrialFunction(M.Sol_SpaceT); tPL = ufl.TrialFunction(M.Sol_SpaceT) 
    TT = ufl.TestFunction(M.Sol_SpaceT)  ; TPL = ufl.TestFunction(M.Sol_SpaceT)
    # -- 
    g = fem.Constant(M.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    
    
    # Main Loop for the convergence -> check residuum of each subsystem 
    
    F_l = set_lithostatic_problem(PL, T_o, tPL, TPL, pdb, sc, g, M )
    
    
    F_S = set_stokes_slab()
    
    #F_S = set_stokes_wedge(PL,T_on,pdb,sc,g,M )
    
    
    # Set the temperature problem 
    #F_T = set_steady_state_thermal_problem(PL, T_on, vel, Ten ,pdb , sc, M)
    
    # Check residuum and plot the residuum plot 
    
    # Save the solution 
    
    return  

#---------------------------------------------------------------------------
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
            
    
    ctrl = NumericalControls()
    
    ctrl = sc_f._scaling_control_parameters(ctrl, sc)
    

    
    pdb = PhaseDataBase(6)
    # Slab
    
    pdb = _generate_phase(pdb, 1, rho0 = 3300 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)
    # Oceanic Crust
    
    pdb = _generate_phase(pdb, 2, rho0 = 2900 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e21)
    # Wedge
    
    pdb = _generate_phase(pdb, 3, rho0 = 3300 , option_rho = 0, option_rheology = 3, option_k = 0, option_Cp = 0, eta=1e21)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # 
    
    pdb = _generate_phase(pdb, 4, rho0 = 3250 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e23)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 5, rho0 = 2800 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e21)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 6, rho0 = 2700 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e21)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = sc_f._scaling_material_properties(pdb,sc)
    
    lhs_ctrl = ctrl_LHS()

    lhs_ctrl = sc_f._scale_parameters(lhs_ctrl, sc)
    
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)

    
    # call the lithostatic pressure 
    
    S   = Solution()
    
    main_solver_steady_state(M, S, ctrl, pdb, sc, lhs_ctrl )
    
    pass 
    

if  __name__ == '__main__':
    unit_test()