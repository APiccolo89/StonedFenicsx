from stonedfenicsx.numerical_control import NumericalControls,IOControls 
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input 
from stonedfenicsx.scal import Scal
from stonedfenicsx.utils import Input,print_ph
from stonedfenicsx.create_mesh.create_mesh import create_mesh
from stonedfenicsx.Stoned_fenicx import fill_geometrical_input
import sys
import numpy as np
from numpy.typing import NDArray
import os
from stonedfenicsx.package_import import *

# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False


# The nonlinear stopping tolerance is set to 1e-1 for the following reasons:
#
# 1) Parallel floating-point round-off:
#    In MPI parallel reductions (e.g., norms, dot products), floating-point
#    operations are not strictly associative. The ordering of reductions
#    differs between runs and processor counts, leading to unavoidable
#    round-off variations (typically O(1e-5) in our tests).
#    Therefore, convergence criteria below this threshold do not reflect
#    physically meaningful changes in the solution.
#
# 2) Iterative vs direct Stokes solve:
#    The Krylov (KSP) solver computes an approximate solution whose accuracy
#    depends on the relative tolerance (rtol). For nonlinear rheologies,
#    especially dislocation creep, the Picard fixed-point iteration becomes
#    sensitive to the linear solver tolerance.
#
#    We observe that:
#      - For linear viscosity and diffusion creep, the solution is robust
#        across a wide range of KSP tolerances.
#      - For dislocation creep, tightening or loosening the KSP rtol
#        changes the nonlinear fixed-point trajectory.
#
#    Below a certain linear tolerance threshold, the nonlinear iteration
#    may converge to a different fixed point or exhibit divergence,
#    indicating sensitivity of the coupled nonlinear–linear scheme.
#
#    The chosen tolerance represents a compromise between numerical
#    stability, physical consistency, and computational cost.
#-------------------------------------------------------------------------------
def perform_test(option_viscous, name = 'Tonga'):
    from stonedfenicsx.utils import parse_input, time_the_time, timing
    from stonedfenicsx.Stoned_fenicx import StonedFenicsx
    
    time_A = timing.time()

    path_test = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_test}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    # option for the benchmark

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    van_keken = 0
    # The input path for saving the results
    inp.path_test = f'{path_test}/Tests_real'

    # Geometrical input
    inp.cr = 0.0   # Overriding crust 
    inp.lc = 0.3   # relative amount of lower crust
    inp.ocr = 6.0e3  # Crustal thickness
    inp.lit_mt = 50e3  # Lithospheric mantle depth 
    inp.lab_d = inp.lit_mt  # depth of the lab 
    inp.decoupling = 50e3  # decoupling depth
    inp.ns_depth = 50e3
    inp.Tmax = 1300.0  # mantle potential temperature
    # inp.model_shear = 'SelfConsistent'
    inp.steady_state = 1
    inp.slab_type = 'File'
    inp.sub_path = f'/Users/wlnw570/Work/Leeds/Fenics_tutorial/example_slab_surfaces/{name}_slab.pz'
    print_ph('Starting the benchmark tests with different options')

    alpha_nameC = 'Constant'
    alpha_nameM = 'Constant'
    density_nameC = 'Constant'
    density_nameM = 'Constant'
    capacity_nameM = 'Constant'
    capacity_nameC = 'Constant'
    conductivity_nameM = 'Constant'
    conductivity_nameC = 'Constant'
    rho0_M = 3300.0
    rho0_C = 3300.0
    radio_flag = 1 

    if option_viscous == 0:
        name_diffusion = 'Constant'
        name_dislocation = 'Constant'              
    elif option_viscous == 1: 
        name_diffusion = 'Van_Keken_diff'
        name_dislocation = 'Constant'       
    elif option_viscous == 2: 
        name_diffusion = 'Van_Keken_diff'
        name_dislocation = 'Van_Keken_disl'                 

    # Modify the phase with the new data: 
    Ph.subducting_plate_mantle.rho0 = rho0_M
    Ph.subducting_plate_mantle.name_capacity = capacity_nameM
    Ph.subducting_plate_mantle.name_conductivity = conductivity_nameM
    Ph.subducting_plate_mantle.name_alpha = alpha_nameM
    Ph.subducting_plate_mantle.name_density = density_nameM
    Ph.subducting_plate_mantle.radio_flag = radio_flag


    Ph.oceanic_crust.rho0 = rho0_C
    Ph.oceanic_crust.name_capacity = capacity_nameC
    Ph.oceanic_crust.name_conductivity = conductivity_nameC
    Ph.oceanic_crust.name_alpha = alpha_nameC
    Ph.oceanic_crust.name_density = density_nameC
    Ph.oceanic_crust.radio_flag = radio_flag

    Ph.wedge_mantle.name_diffusion = name_diffusion
    Ph.wedge_mantle.name_dislocation = name_dislocation
    Ph.wedge_mantle.rho0 = rho0_M
    Ph.wedge_mantle.name_capacity = capacity_nameM 
    Ph.wedge_mantle.name_conductivity = conductivity_nameM
    Ph.wedge_mantle.name_alpha = alpha_nameM
    Ph.wedge_mantle.name_density = density_nameM
    Ph.wedge_mantle.radio_flag = radio_flag

    Ph.overriding_mantle.rho0 = rho0_M 
    Ph.overriding_mantle.name_capacity = capacity_nameM
    Ph.overriding_mantle.name_conductivity = conductivity_nameM
    Ph.overriding_mantle.name_alpha = alpha_nameM
    Ph.overriding_mantle.name_density = density_nameM
    Ph.overriding_mantle.radio_flag = radio_flag

    Ph.overriding_upper_crust.rho0 = rho0_C 
    Ph.overriding_upper_crust.name_capacity = capacity_nameC
    Ph.overriding_upper_crust.name_conductivity = conductivity_nameC
    Ph.overriding_upper_crust.name_alpha = alpha_nameC
    Ph.overriding_upper_crust.name_density = density_nameC
    Ph.overriding_upper_crust.radio_flag = radio_flag

    Ph.overriding_lower_crust.rho0 = rho0_C 
    Ph.overriding_lower_crust.name_capacity = capacity_nameC
    Ph.overriding_lower_crust.name_conductivity = conductivity_nameC
    Ph.overriding_lower_crust.name_alpha = alpha_nameC
    Ph.overriding_lower_crust.name_density = density_nameC
    Ph.overriding_lower_crust.radio_flag = radio_flag

    Ph.virtual_weak_zone.name_diffusion = 'Hirth_Wet_Olivine_disl'
    Ph.virtual_weak_zone.name_dislocation = 'Hirth_Wet_Olivine_disl' 


    inp.sname = f'T_{name}_viscous{option_viscous}_{inp.stokes_solver_type}_{MPI.COMM_WORLD.Get_size():d}'

    # Initialise the input
    inp.van_keken = van_keken


    StonedFenicsx(inp, Ph)

    time_B = timing.time()
    dt = time_B - time_A
    print('#---------------------------------------------------#')
    if dt > 60.0:
        m, s = divmod(dt, 60)
        print(f"{inp.sname} took {m:.2f} min and {s:.2f} sec")
    elif dt > 3600.0:
        m, s = divmod(dt, 60)
        h, m = divmod(m, 60)
        print(f"{inp.sname} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
    else:
        print(f"{inp.sname} took  {dt:.2f} sec")
    print('#---------------------------------------------------#')
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------   
def test_composite(name='Tonga'):
    # Test Van Keken
    perform_test(2,name)
    # Read Data Base and compare data
    #if MPI.COMM_WORLD.rank == 0: 
    #    read_data_base(2)
    # Remove folder after completing the test
    #if DEBUG == False:
        
   #     os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True

    test_composite()
#---------------------------------------------------------------------------------
