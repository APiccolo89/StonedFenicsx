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
#-------------------------------------------------------------------------------
def perform_test(option_viscous:int=2
                 ,direct:str='Direct'
                 ,rtol:int =0)->None:
    
    
    from stonedfenicsx.utils import parse_input, time_the_time, timing
    from stonedfenicsx.Stoned_fenicx import StonedFenicsx
    
    time_A = timing.time()

    path_test = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_test}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    # option for the benchmark

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    van_keken = 1
    # The input path for saving the results
    inp.path_test = f'{path_test}/Tests_Van_keken'

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
    inp.stokes_solver_type = direct
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
    radio_flag = 0 

    if option_viscous == 0:
        name_diffusion = 'Constant'
        name_dislocation = 'Constant'              
    elif option_viscous == 1: 
        name_diffusion = 'Van_Keken_diff'
        name_dislocation = 'Constant'       
    elif option_viscous == 2: 
        name_diffusion = 'Van_Keken_diff'
        name_dislocation = 'Van_Keken_disl'                 
    
    if rtol == 0: 
        inp.iterative_solver_tol = 1e-10 
    if rtol == 1: 
        inp.iterative_solver_tol = 1e-9 
    if rtol == 2: 
        inp.iterative_solver_tol = 1e-8 
    if rtol == 3: 
        inp.iterative_solver_tol = 1e-11 
        
    

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
    Ph.wedge_mantle.name_conductivity = capacity_nameM
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


    inp.sname = f'T_viscous{option_viscous}_direct{direct}_rtol{rtol}'

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
def read_data_base(option_viscous):
    import h5py as h5 
    import os
    def extract_relevant_information(direct, rtol):
            

    
        
    return pass_flag
#-------------------------------------------------------------------------------   
def test_composite_direct()->None:
    # Test Van Keken
    perform_test(2, 'Direct')


def test_composite_iterative(rel_tol:int=0)->None:
    perform_test(2,'Iterative',0)
    



#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    
    test_composite_direct()
    
    test_composite_iterative(0)
    
    test_composite_iterative(1)
    
    test_composite_iterative(2)
    
    test_composite_iterative(3)
#---------------------------------------------------------------------------------
