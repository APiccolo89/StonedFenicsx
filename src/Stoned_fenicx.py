

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
from src.thermal_structure_ocean     import compute_initial_LHS

dict_options = {'NoShear':0,
                'Linear':1,
                'SelfConsistent':2}

def generate_phase_database(IP,Phin)->PhaseDataBase:
    
    pdb = PhaseDataBase(7,IP.phi*np.pi/180)
    
    it = 1 
    for i in dir(Phin):
        if 'Phase' in i:
            phase = getattr(Phin,i)
            print_ph(f"Generating phase {it} : {i}, Phase Name : {phase.name_phase}")
            
            print_ph('-----Rheological Parameters------')
            print_ph(f"Diffusion law  : {phase.name_diffusion if hasattr(phase, 'name_diffusion') else 'Constant'}")
            if phase.Edif != -1e23:
                print_ph(f"   Edif : {phase.Edif} ")
            if phase.Vdif != -1e23:
                print_ph(f"   Vdif : {phase.Vdif} ")
            if phase.Bdif != -1e23:
                print_ph(f"   Bdif : {phase.Bdif} ")
            
            
            print_ph(f"Dislocation law: {phase.name_dislocation if hasattr(phase, 'name_dislocation') else 'Constant'}")
            if phase.n != -1e23:
                print_ph(f"   n    : {phase.n} ")
            if phase.Edis != -1e23:
                print_ph(f"   Edis : {phase.Edis} ")
            if phase.Vdis != -1e23:
                print_ph(f"   Vdis : {phase.Vdis} ")
            if phase.Bdis != -1e23:
                print_ph(f"   Bdis : {phase.Bdis} ")
            if phase.name_diffusion == 'Constant' and phase.name_dislocation == 'Constant':
                print_ph(f"   eta  : {phase.eta} [Pas] ")    
            
            print_ph('-----------------------------------')
            
            print_ph('-----Thermal Parameters------')
            print_ph(f"Density law       : {phase.name_density if hasattr(phase, 'name_density') else 'Constant'}")
            print_ph(f"Thermal capacity  : {phase.name_capacity if hasattr(phase, 'name_capacity') else 'Constant'}")
            print_ph(f"Thermal conductivity : {phase.name_conductivity if hasattr(phase, 'name_conductivity') else 'Constant'}")
            print_ph(f"Thermal expansivity : {phase.name_alpha if hasattr(phase, 'name_conductivity') else 'Constant'}")
            print_ph(f"Radiogenic heating:  {phase.Hr if phase.Hr !=0.0 else 'Radiogenic heating is not active'}")
            
            if hasattr(phase, 'radio_flag'):
                print_ph(f"   radiative conductivity flag : {phase.radio_flag} ")
            if hasattr(phase, 'rho0'):
                print_ph(f"   rho0 : {phase.rho0} ")
            print_ph('-----------------------------------') 
            if phase.name_capacity == 'Constant':
                print_ph(f"Heat capacity {phase.Cp} J/kg/K")
                print_ph('-----------------------------------')
            if phase.name_conductivity == 'Constant':
                print_ph(f"Thermal conductivity {phase.k} W/m/K")
                print_ph('-----------------------------------') 
            print_ph('\n')
            
            pdb = _generate_phase(pdb,
                                  it, 
                                    radio_flag        = phase.radio_flag if hasattr(phase, 'radio_flag') else 0.0,
                                    rho0              = phase.rho0 if hasattr(phase, 'rho0') else 3300,
                                    name_diffusion    = phase.name_diffusion if hasattr(phase, 'name_diffusion') else 'Constant',
                                    name_dislocation  = phase.name_dislocation if hasattr(phase, 'name_dislocation') else 'Constant',
                                    name_alpha        = phase.name_alpha if hasattr(phase, 'name_alpha') else 'Constant',
                                    name_capacity     = phase.name_capacity if hasattr(phase, 'name_capacity') else 'Constant',
                                    name_density      = phase.name_density if hasattr(phase, 'name_density') else 'Constant',
                                    name_conductivity = phase.name_conductivity if hasattr(phase, 'name_conductivity') else 'Constant',
                                    Edif              = phase.Edif if hasattr(phase, 'Edif') else -1e23,
                                    Vdif              = phase.Vdif if hasattr(phase, 'Vdif') else -1e23,
                                    Bdif              = phase.Bdif if hasattr(phase, 'Bdif') else -1e23,
                                    n                 = phase.n if hasattr(phase, 'n') else -1e23,
                                    Edis              = phase.Edis if hasattr(phase, 'Edis') else -1e23,
                                    Vdis              = phase.Vdis if hasattr(phase, 'Vdis') else -1e23,
                                    Bdis              = phase.Bdis if hasattr(phase, 'Bdis') else -1e23,
                                    eta               = phase.eta if hasattr(phase, 'eta') else 1e20)
            it += 1

    return pdb 

def StonedFenicsx(IP,Ph_input):
    #---------------------------------------------------------------------------------------------------------
    # Input parameters 
    #---------------------------------------------------------------------------------------------------------
        
    
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
                            dt = IP.dt_sim,
                            adiabatic_heating = IP.adiabatic_heating,
                            Tmax=IP.Tmax)
    # IO controls
    io_ctrl = IOControls(test_name = IP.test_name,
                        path_save = IP.path_test,
                        sname = IP.sname)
    io_ctrl.generate_io()
    # Scaling parameters
    sc = Scal(L=IP.L,Temp = IP.Temp,eta = IP.eta, stress = IP.stress)
    # LHS parameters
    lhs = ctrl_LHS(nz=IP.nz,
                    end_time = IP.end_time,
                    dt = IP.dt,
                    recalculate = IP.recalculate,
                    van_keken = 0,#IP.van_keken,
                    non_linearities=0,
                    c_age_plate = IP.c_age_plate)
    
    Pdb = generate_phase_database(IP,Ph_input)                      
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
    

    # Scaling
    ctrl = _scaling_control_parameters(ctrl, sc)
    Pdb = _scaling_material_properties(Pdb,sc)
    lhs = _scale_parameters(lhs, sc)

    M = cm(io_ctrl, sc,g_input,ctrl)
    
    M.element_p = basix.ufl.element("Lagrange","triangle", 1) 
    M.element_PT = basix.ufl.element("Lagrange","triangle",2)
    M.element_V = basix.ufl.element("Lagrange","triangle",2,shape=(2,))
    
    if ctrl.steady_state == 1:
        steady_state_solution(M, ctrl, lhs, Pdb, io_ctrl, sc)
    else:
        time_dependent_solution(M, ctrl, lhs, Pdb, io_ctrl, sc)

         
    # Create mesh
    return 0    

#---------------------------------------------------------------------------
 
if __name__ == '__main__': 

        
    StonedFenicsx()   
    
    
        
    
    
    

