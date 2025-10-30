

# -- Main solver script --
from petsc4py                        import PETSc
from mpi4py                          import MPI
import numpy                          as np
from scipy.interpolate                import griddata
import ufl
from dolfinx                          import mesh, fem, io, nls, log, plot
from dolfinx.fem                       import (Function, FunctionSpace, dirichletbc,
                                           locate_dofs_topological, form)
from dolfinx.fem.petsc import (LinearProblem,
                               NonlinearProblem)
from dolfinx.io                        import XDMFFile
from dolfinx.mesh                      import CellType, create_rectangle, locate_entities_boundary
from dolfinx.cpp.mesh                  import to_type

import basix
import time                          as timing
import sys 
import os 

#---------------------------------------------------------------------------------------------------------
# My modules 
#---------------------------------------------------------------------------------------------------------
from src.create_mesh                 import create_mesh as cm 
from src.solution                    import Solution as sol 
from src.phase_db                    import PhaseDataBase 
from src.numerical_control           import ctrl_LHS 
from src.utils                       import print_ph
from src.utils                       import timing_function
from src.utils                       import time_the_time
from src.utils                       import get_discrete_colormap
from src.phase_db                    import _generate_phase
from src.scal                        import Scal
from src.scal                        import _scaling_material_properties
from src.numerical_control           import NumericalControls
from src.numerical_control           import IOControls
from src.create_mesh                 import Mesh
from src.create_mesh                 import Geom_input
from src.scal                        import _scaling_control_parameters
from src.scal                        import _scale_parameters
from src.scal                        import _scaling_material_properties
from src.solution                    import steady_state_solution,time_dependent_solution

dict_options = {'Linear':1,
                'SelfConsistent':2}



def StonedFenicsx():
    #---------------------------------------------------------------------------------------------------------
    # Input parameters 
    #---------------------------------------------------------------------------------------------------------
    
    import argparse
    import importlib.util
    parser = argparse.ArgumentParser(
        description="Run simulation with a given input path."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input file or directory."
    )

    args = parser.parse_args()
    input_path = args.input_path
    
    module_name = os.path.splitext(os.path.basename(input_path))[0]  # e.g. "input_case"

    spec = importlib.util.spec_from_file_location(module_name, input_path)
    IP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(IP)
    print(f"Loaded {module_name} from {input_path}")
    print(IP)
    print(IP.test_name)
    
    # Numerical controls
    ctrl = NumericalControls(g               = IP.g,
                            v_s              = np.asarray(IP.v_s),
                            slab_age         = IP.slab_age,
                            time_max         = IP.time_max,
                            time_dependent_v = IP.time_dependent_v,
                            steady_state     = IP.steady_state,
                            slab_bc          = IP.slab_bc,
                            decoupling       = IP.decoupling_ctrl,
                            tol_innerPic     = IP.tol_innerPic,
                            tol_innerNew     = IP.tol_innerNew,
                            van_keken        = IP.van_keken,
                            van_keken_case   = IP.van_keken_case,
                            model_shear      = dict_options[IP.model_shear],
                            phase_wz         = IP.phase_wz,
                            dt = IP.dt_sim)
    # IO controls
    io_ctrl = IOControls(test_name = IP.test_name,
                        path_save = IP.path_save,
                        sname = IP.sname)
    io_ctrl.generate_io()
    # Scaling parameters
    sc = Scal(L=IP.L,Temp = IP.Temp,eta = IP.eta, stress = IP.stress)
    # LHS parameters
    lhs = ctrl_LHS(nz=IP.nz,
                    end_time = IP.end_time,
                    dt = IP.dt,
                    recalculate = IP.recalculate,
                    van_keken = IP.van_keken,
                    c_age_plate = IP.c_age_plate)
                           
    # Phase properties
    Pdb = PhaseDataBase(7,IP.friction_angle*np.pi/180)
    # Phase 1
    Pdb = _generate_phase(Pdb, 1, rho0 = IP.Phase1.rho0 , 
                          option_rho = IP.Phase1.option_rho, 
                          option_rheology = IP.Phase1.option_rheology, 
                          option_k = IP.Phase1.option_k, option_Cp = IP.Phase1.option_Cp, 
                          eta=IP.Phase1.eta,
                          radio = IP.Phase1.radio)
    # Phase 2
    Pdb = _generate_phase(Pdb,
                          2,
                          rho0 = IP.Phase2.rho0,
                          option_rho = IP.Phase2.option_rho, 
                          option_rheology = IP.Phase2.option_rheology,
                          option_k = IP.Phase2.option_k, 
                          option_Cp = IP.Phase2.option_Cp, 
                          eta=IP.Phase2.eta,
                          radio = IP.Phase2.radio)
    # Phase 3
    Pdb = _generate_phase(Pdb, 
                          3, 
                          rho0 = IP.Phase3.rho0 , 
                          option_rho = IP.Phase3.option_rho, 
                          option_rheology = IP.Phase3.option_rheology,
                          option_k = IP.Phase3.option_k, 
                          option_Cp = IP.Phase3.option_Cp, 
                          name_diffusion=IP.Phase3.name_diffusion, 
                          name_dislocation=IP.Phase3.name_dislocation,
                          eta = IP.Phase3.eta,
                          radio = IP.Phase3.radio)
    # Phase 4
    Pdb = _generate_phase(Pdb, 
                          4, 
                          rho0 = IP.Phase4.rho0 , 
                          option_rho = IP.Phase4.option_rho, 
                          option_rheology = IP.Phase4.option_rheology,
                          option_k = IP.Phase4.option_k, 
                          option_Cp = IP.Phase4.option_Cp, 
                          eta=IP.Phase4.eta,
                          radio = IP.Phase4.radio)
    # Phase 5
    Pdb = _generate_phase(Pdb, 
                          5, 
                          rho0 = IP.Phase5.rho0 , 
                          option_rho = IP.Phase5.option_rho, 
                          option_rheology = IP.Phase5.option_rheology,
                          option_k = IP.Phase5.option_k, 
                          option_Cp = IP.Phase5.option_Cp, 
                          eta=IP.Phase5.eta,
                          radio = IP.Phase5.radio)
    # Phase 6
    Pdb = _generate_phase(Pdb, 
                          6, 
                          rho0 = IP.Phase6.rho0 , 
                          option_rho = IP.Phase6.option_rho, 
                          option_rheology = IP.Phase6.option_rheology,
                          option_k = IP.Phase6.option_k, 
                          option_Cp = IP.Phase6.option_Cp, 
                          eta=IP.Phase6.eta,
                          radio = IP.Phase6.radio)
        # Phase 3
    Pdb = _generate_phase(Pdb, 
                          7, 
                          option_rheology = IP.Phase7.option_rheology,
                          name_diffusion=IP.Phase7.name_diffusion, 
                          name_dislocation=IP.Phase7.name_dislocation)
    # ---
    # Create mesh 
    g_input = Geom_input(x = np.asarray(IP.x),
                 y = np.array(IP.y),
                 cr=IP.cr,
                 ocr=IP.ocr,
                 lit_mt=IP.lit_mt,
                 lc = IP.lc,
                 wc = IP.wc,
                 slab_tk = IP.slab_tk, 
                 decoupling = IP.decoupling,
                 lab_d = IP.lab_d)
    
    
    M = cm(io_ctrl, sc,g_input,ctrl)
    
    M.element_p       = IP.element_p
    M.element_PT      = IP.element_PT
    M.element_V       = IP.element_V
    
    
    # Scaling
    ctrl = _scaling_control_parameters(ctrl, sc)
    Pdb = _scaling_material_properties(Pdb,sc)
    lhs = _scale_parameters(lhs, sc)

    if ctrl.steady_state == 1:
        steady_state_solution(M, ctrl, lhs, Pdb, io_ctrl, sc)
    else:
        time_dependent_solution(M, ctrl, lhs, Pdb, io_ctrl, sc)

         
    # Create mesh
    return 0    





 
if __name__ == '__main__': 
        
    StonedFenicsx()   
        
    
    
    

