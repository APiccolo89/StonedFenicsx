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
def perform_test(option_viscous):
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


    inp.sname = f'T_viscous{option_viscous}'

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
    
    # File h5 that stores data of benchmarks
    file_h5 = f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken/benchmark_van_keken.h5'
    # Current test Name
    Test_name = f'T_viscous{option_viscous}'
    # Open the file 
    f = h5.File(file_h5)
    # Read the data from the database: 
    field_a = f'{Test_name}/T_11_11'
    T_11_11 = np.array(f[field_a])
    field_b = f'{Test_name}/L2_A'
    L2_A = np.array(f[field_b])
    field_c = f'{Test_name}/L2_B'
    L2_B = np.array(f[field_c])
    
    if option_viscous==0:  
        
        data = np.array([
        [397.55, 505.70, 850.50],
        [391.57, 511.09, 852.43],
        [387.78, 503.10, 852.97],
        [387.84, 503.13, 852.92],
        [389.39, 503.04, 851.68],
        [388.73, 504.03, 854.99],
        [390.40, 488.0, 847.70],
         ])
        v1 = 383.1834
        v2 = 500.3690
        v3 = 852.6929
        
        # 
           
    if option_viscous == 1:
        data = np.array([
        [570.30, 614.09, 1007.31],
        [580.52, 606.94, 1002.85],
        [580.66, 607.11, 1003.20],
        [577.59, 607.52, 1002.15],
        [581.30, 607.26, 1003.35],
        [584.20, 592.8, 1000.0],
        ])
        v1 = 573.3623
        v2 = 603.1777
        v3 = 1002.6859

    if option_viscous==2: 
        
        data = np.array([
        [550.17, 593.48, 994.11],
        [551.60, 608.85, 984.08],
        [582.65, 604.51, 998.71],
        [583.36, 605.11, 1000.01],
        [574.84, 603.80, 995.24],
        [583.11, 604.96, 1000.05],
        [585.70, 591.30, 996.60]
        ])
        v1 = 575.8219
        v2 = 601.0150
        v3 = 999.1961

    db_vk1 = [np.mean(data[:,0]), np.min(data[:,0]), np.max(data[:,0])]
    db_vk2 = [np.mean(data[:,1]), np.min(data[:,1]), np.max(data[:,1])]
    db_vk3 = [np.mean(data[:,2]), np.min(data[:,2]), np.max(data[:,2])]

    test_1 = np.isclose(T_11_11, v1, 1e-4, 1e-4)
    test_2 = np.isclose(L2_A, v2,1e-4, 1e-4)
    test_3 = np.isclose(L2_B, v3,1e-4, 1e-4)
    
    print_ph(f'Test_viscous{option_viscous}, T_11_11 is {T_11_11:.4f}. Tested against {v1:.4f}.')
    print_ph(f'                             Van Keken benchmark : mean T_11_11 = {db_vk1[0]:.2f}.')
    print_ph(f'                             Van Keken benchmark : range T_11_11 = {db_vk1[1]:.2f}-{db_vk1[2]:.2f}.')

    print_ph(f'Test_viscous{option_viscous}, L2_A is {L2_A:.4f}. Tested against {v2:.4f}.')
    print_ph(f'                             Van Keken benchmark : mean L2_A = {db_vk2[0]:.2f}.')
    print_ph(f'                             Van Keken benchmark : range L2_A = {db_vk2[1]:.2f}-{db_vk2[2]:.2f}.')
    
    print_ph(f'Test_viscous{option_viscous}, L2_B is {L2_B:.4f}. Tested against {v3:.4f}.')
    print_ph(f'                             Van Keken benchmark : mean L2_A = {db_vk3[0]:.2f}.')
    print_ph(f'                             Van Keken benchmark : range L2_A = {db_vk3[1]:.2f}-{db_vk3[2]:.2f}.')

    if test_1 & test_2 & test_3:         
        pass_flag = True
        print_ph(f'Test_viscous{option_viscous} passed... ')
    else: 
        assert test_1 
        assert test_2
        assert test_3 
    
    f.close()
        
    return pass_flag
#-------------------------------------------------------------------------------
def test_isoviscous():
    # Test Van Keken 
    perform_test(0) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(0)
    # Remove folder after completing the test
    if DEBUG == False:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
def test_diffusion():
    # Test Van Keken
    perform_test(1)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(1)
    # Remove folder after completing the test
    if DEBUG == False:
       
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------   
def test_composite():
    # Test Van Keken
    perform_test(2)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(2)
    # Remove folder after completing the test
    if DEBUG == False:
        
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    
    #test_isoviscous()

    #test_diffusion()

    test_composite()

