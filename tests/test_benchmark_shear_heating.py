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

#-------------------------------------------------------------------------------
def perform_test(option_thermal:int = 0
                 ,weakzone:str = 'WetQuartzite'
                 ,friction_angle:float = np.radians(5))->None:
    """_summary_

    Args:
        option_viscous (int, optional): Define the rheology of the wedge. Defaults to 0.
        
        option_thermal (int, optional): Define if the material properties are constant or not, and if the crustal unit have different properties. Defaults to 0.
                        [0]: Everything is constant 
                        [1]: Non linear properties -> only mantle properties 
                        [2]: Non linear properties -> crustal properties 
        
        option_decoupling (int, optional): Activate the decoupling. Defaults to 0. [0: not active, 1: active {decoupling depth = 80 km}]
        
        option_shear_heating (int, optional): Activate the shear heating. Defaults to 0. [0: not active, 1: active]
    """
    
    from stonedfenicsx.utils import parse_input, time_the_time, timing
    from stonedfenicsx.Stoned_fenicx import StonedFenicsx
    
    time_A = timing.time()

    path_test = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_test}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    # option for the benchmark

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    # The input path for saving the results
    inp.path_test = f'{path_test}/Tests_Van_keken'

    # Geometrical input
    inp.cr = 0.0   # Overriding crust 
    inp.lc = 0.3   # relative amount of lower crust
    inp.ocr = 6.0e3  # Crustal thickness
    inp.lit_mt = 50e3  # Lithospheric mantle depth 
    inp.lab_d = inp.lit_mt  # depth of the lab 

    inp.dislocation_creep_wz = 'WetQuartzite_disl'
    inp.decoupling = 80e3 
    inp.decoupling_ctrl = 1 
        
    inp.ns_depth = 50e3
    inp.Tmax = 1350.0  # mantle potential temperature
    
    inp.steady_state = 1
    inp.model_shear = 'SelfConsistent'
    
    print_ph('Starting the benchmark tests with different options')
    if option_thermal == 0: 
    
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
    elif option_thermal == 1: 
        
        alpha_nameC = 'Mantle'
        alpha_nameM = 'Mantle'
        density_nameC = 'PT'
        density_nameM = 'PT'
        capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'
        capacity_nameC = 'Bermann_Aranovich_Fo_Fa_0_1'
        conductivity_nameM = 'Mantle'
        conductivity_nameC = 'Mantle'
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 1 
        
    elif option_thermal == 2: 

        alpha_nameC = 'Crust'
        alpha_nameM = 'Mantle'
        density_nameC = 'PT'
        density_nameM = 'PT'
        capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'
        capacity_nameC = 'Oceanic_Crust'
        conductivity_nameM = 'Mantle'
        conductivity_nameC = 'Oceanic_Crust'
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 1 
        
        


    name_diffusion = 'Constant'
    name_dislocation = 'Van_Keken_disl'   
    
    inp.slab_type = 'Costum'
    inp.sub_theta_max = 20 
    inp.x = [0.0,652.0e3]
    inp.y = [-230.0e3, 0.0]
    inp.sub_constant_flag = 1


    # Modify the phase with the new data: 
    inp.phi = friction_angle
    # Phase 
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



    inp.sname = f'Shear_heating_tests_oTh{option_thermal}_wz{inp.dislocation_creep_wz}_fr{int(inp.phi)}'

    # Initialise the input
    inp.van_keken = 0


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
def test_composite_shear_heating(friction_angle = np.radians(0)):
    # Test Van Keken
    perform_test(option_thermal=2,friction_angle=friction_angle)
    # Read Data Base and compare data
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    test_composite_shear_heating()
    test_composite_shear_heating(friction_angle=2)
    test_composite_shear_heating(friction_angle=5)
    test_composite_shear_heating(friction_angle=10)
    test_composite_shear_heating(friction_angle=15)
    

    

#---------------------------------------------------------------------------------
