# input for iFieldstone 
import make_mesh.Create_Mesh as C_msh
import os
from material_property.phase_db import PhaseDataBase
from solver_function.numerical_control import NumericalControls
from material_property.phase_db import _generate_phase
from solver_function.numerical_control import IOControls
from solver_function.numerical_control import ctrl_LHS
from numba.experimental import jitclass
from solver_function.numerical_control import dict_k, dict_Cp, dict_rho, dict_rheology
import numpy as np
from solver_function.numerical_control import bc_controls
from solver_function.scal import Scal 
import ufl
import numpy

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

def set_initial_():
    v_ST = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

    n_phase = 3 
    v_s = [5.0,0.0]
    v_s_arr = np.asarray(v_s, dtype=np.float64)
    slab_age = 50.e6   # slab age [yr]

    X = [0,1000e3]
    Z = [-600e3,0]

    # ========== Bending function ========== #

    option_k_ = dict_k['Tosi2013']
    option_Cp_ = dict_Cp['base']
    option_rho_ =dict_rho['pres_dep']
    rheology_ = dict_rheology['default'] # 0: default, 1: diffusion only, 2: composite, 3: dislocation only, 4: composite iter
    r_channel = 2000
    s_ = '_SUPG_Steady_State_relax'

    t_name = 'T_ok%d_Cp%d_r%d_v%d_vel%d_R_%d_%s'%(option_k_,option_Cp_,option_rho_,rheology_,int(v_s[0]),int(r_channel),s_)
    p_save = '../time_crank'

    sname        = t_name

    v_s=np.asarray(v_s, dtype=np.float64)
    scal = Scal(L=660e3,stress = 1e9,eta = 1e22,Temp=1350)
    # ========== constants ========== #
    num_ctrl = NumericalControls(option_k = option_k_,
                            option_Cp = option_Cp_,
                            option_rho = option_rho_, 
                            rheology = rheology_,
                            it_max = 100,
                            tol    = 1e-4,
                            relax  = 0.8,
                            eta_max = 1e26, 
                            eta_def = 1e21, 
                            eta_min = 1e18,
                            Ttop    = 0.0,
                            Tmax    = 1300.0,
                            g = 9.81, 
                            pressure_scaling=1e22/1000e3,
                            slab_age = slab_age,
                            v_s = v_s,
                            advect_temperature = 1,
                            pressure_bc = 0,
                            petsc = 1,
                            dt = 1.0,
                            time_dependent_v = 1)

    num_ctrl.Ttop = num_ctrl.Ttop/scal.Temp 
    num_ctrl.Tmax = num_ctrl.Tmax/scal.Temp
    num_ctrl.g    = num_ctrl.g/scal.ac
    num_ctrl.v_s    = num_ctrl.v_s/(scal.L/scal.T)
    num_ctrl.pressure_scaling = (num_ctrl.pressure_scaling*scal.L)/scal.eta
    num_ctrl.t_max = 20e6 * num_ctrl.scal_year/scal.T # 20 Myr in seconds



    IOCtrl = IOControls(test_name = t_name,
                        path_save = p_save,
                        sname = sname)
    IOCtrl.generate_io()

    lhs_temp = ctrl_LHS(dz=1e3,
                        nz=107,
                        end_time = 200e6,
                        dt = 5e3,
                        recalculate = 1,
                        van_keken = 1,)
                        #d_RHS = -100e3)
    lhs_temp._scale_parameters(num_ctrl)
    lhs_temp.d_RHS /= scal.L 

    make_mesh = True
    meshfilename = '%s.msh'%(sname)
#    if os.path.isfile(meshfilename) == False or make_mesh == True: 
#        C_msh._create_mesh_(sname=sname,
#                            width_channel=r_channel,
#                            depth_high_res_trench=r_channel,
#                            lc_normal = 2e3,
#                            lc_litho=2e3,
#                            lc_top = 2e3,
#                            X=X,
#                            Z=Z,
#                            path_test=IOCtrl.path_test,
#                            van_keken = 1)

    BC_STOKES = [['Top','NoSlip',np.array([0.0,0.0],dtype=np.float64)],
                 ['Bottom','DoNothing',np.array([0.0,0.0],dtype=np.float64)],
                 ['Right','DoNothing',np.array([0.0,1.0],dtype=np.float64)],
                 ['Left','DoNothing',np.array([0.0,1.0],dtype=np.float64)],
                 ['Feature_on']]
    BC_ENERGY = [['Top','Isothermal',np.array([num_ctrl.Ttop,0.0],dtype=np.float64)],
                 ['Bottom','Open',np.array([num_ctrl.Tmax,0.0],dtype=np.float64)],
                 ['Right','Open',np.array([0.0,0.0],dtype=np.float64)],
                 ['Left','Open',np.array([0.0,0.0],dtype=np.float64)],
                 ['Feature_on']]
    BC_spec = bc_controls(BC_STOKES,BC_ENERGY)

    # model setup

    PDB = PhaseDataBase(2)
    _generate_phase(PDB,2,id=0,name_diffusion='Van_Keken_diff',name_dislocation='Van_Keken_disl')
    _generate_phase(PDB,2,id=1,name_diffusion='Van_Keken_diff',name_dislocation='Van_Keken_disl')

    #_generate_phase(PDB,2,id=1,name_diffusion='Hirth_Wet_Olivine_diff',name_dislocation='Hirth_Wet_Olivine_disl')
    PDB.Vdis[0] = 0e-6 # cm3/mol
    PDB.Vdif[0] = 0e-6 # cm3/mol

    PDB.Vdis[1] = 0e-6 # cm3/mol
    PDB.Vdif[1] = 0e-6 # cm3/mol
    
    return PDB,BC_spec,num_ctrl,scal,IOCtrl


def create_parallel_mesh(ctrl,ctrlio):
    from dolfinx.io import XDMFFile, gmshio
    from mpi4py import MPI
    import gmsh 

    oc_crust      = 6e3 
    crust         = 35e3
    lc            = 0.3 
    lit_mantle_tk = 60e3 
    lit_tk        = crust+lit_mantle_tk
    
    # Set up slab top surface a
    Data_real = False; S = []
    van_keken = 1
    if (Data_Real==False) & (isinstance(S, Slab)== False):
        if van_keken == 1: 
            min_x           = 0.0 # The beginning of the model is the trench of the slab
            max_x           = 660e3                 # Max domain x direction
            min_y           = -600e3 # Min domain y direction
            max_y           = 0.0 # Max domain y direction 
            S = ffm.Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=5, theta_max = 45.0, num_segment=100,flag_constant_theta=True,y_min=min_y)
        else: 
            S = fmm.Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=1.0, theta_max = 45.0, num_segment=100,flag_constant_theta=False,y_min=min_y)

        for a in dir(S):
            if (not a.startswith('__')):
                att = getattr(S, a)
                if (callable(att)==False) & (np.isscalar(att)):
                    print('%s = %.2f'%(a, att))

    # Create the subduction interfaces using either the real data set, or the slab class
    slab_x, slab_y, theta_mean,channel_x,channel_y,extra_x,extra_y,isch = fmm.function_create_slab_channel(Data_Real,width_channel,lithosphere_thickness,S,real_slab_file)
    
    
    gmsh.initialize()
    
    
    
    
    
    # -> Generate the subduction node and channel node 
    
    
    # -> Generate the loop of the phase 
    # -> Generate the mesh 
    
    
    
    
    
    
    return 0


def Poisson_lithostatic_tutorial(pdb,BC,ctrl,IOCtrl,sc): 
    
    
    def q(u):
        return 1 + u**2
    # Load mesh with standard data: 
    
    create_parallel_mesh(ctrl,IOCtrl)

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    x = ufl.SpatialCoordinate(domain)
    u_ufl = 1 + x[0] + 2 * x[1]
    f = - ufl.div(q(u_ufl) * ufl.grad(u_ufl))

    V = fem.functionspace(domain, ("Lagrange", 1))
    def u_exact(x): return eval(str(u_ufl))

    u_D = fem.Function(V)
    u_D.interpolate(u_exact)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))


    uh = fem.Function(V)
    v = ufl.TestFunction(V)
    F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

    problem = NonlinearProblem(F, uh, bcs=[bc])

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
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
    n, converged = solver.solve(uh)
    assert (converged)
    print(f"Number of interations: {n:d}")
    
    
    



if __name__ == "__main__":
    
    pdb,BC_spec,ctrl,sc = set_initial_()
    Poisson_lithostatic_tutorial(pdb,BC_spec,ctrl,sc)
    
    
    
    
    

    
    
    
    
