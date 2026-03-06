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
def evaluate_tests(option_viscous:int = 0
                   ,option_thermal:int = 0
                   ,option_decoupling:int = 0
                   ,option_shear_heating:int = 0
                   ,keyvalues:list = [])-> bool:

    if option_viscous == 0 and option_thermal == 0 and option_decoupling == 0 and option_shear_heating == 0: 
        vls = [383.1834, 500.3690, 852.6929] 
    if option_viscous ==1  and option_thermal == 0 and option_decoupling == 0 and option_shear_heating == 0:   
        vls = [573.3623, 603.1777, 1002.6859]
    if option_viscous ==2 and option_thermal == 0 and option_decoupling == 0 and option_shear_heating == 0:  
        vls=[575.8907, 601.0743, 999.3790]
    if option_viscous == 2 and option_thermal == 1 and option_decoupling == 0 and option_shear_heating == 0:  
        vls = [562.6966,612.0700,946.7930]
    if option_viscous == 2 and option_thermal == 2 and option_decoupling == 0 and option_shear_heating == 0: 
        vls = [588.4029,629.3958,950.6547] 
    if option_viscous == 0 and option_thermal == 0 and option_decoupling == 1 and option_shear_heating == 0: 
        vls = [221.7751,491.4044,816.8154]
    if option_viscous == 0 and option_thermal == 0 and option_decoupling == 1 and option_shear_heating == 1: 
        vls = [0,0,0]    
                
    test_1 = np.isclose(keyvalues[0], vls[0] ,5e-1, 1e-1)
    test_2 = np.isclose(keyvalues[1], vls[1] ,5e-1, 1e-1)
    test_3 = np.isclose(keyvalues[2], vls[2] ,5e-1, 1e-1)
    
    if test_1 and test_2 and test_3: 
        pass_flag = True 
    else: 
        pass_flag = False 
        if not test_1: 
            raise Warning('Test 1 failed')
        if not test_2: 
            raise Warning('Test 2 failed')
        if not test_3: 
            raise Warning('Test 3 failed')


    error = [np.abs(vls[i]-keyvalues[i]) for i in range(len(keyvalues))]

    print_ph(f'Test_vi{option_viscous}_th{option_thermal}_dc{option_decoupling}_sh{option_shear_heating}    : Absolute error: ')
    print_ph(f'                                                                                      T_11_11: {error[0]:.3f} ')
    print_ph(f'                                                                                      L2_A   : {error[1]:.3f} ')
    print_ph(f'                                                                                      L2_B   : {error[2]:.3f} ')
    print_ph(f'Test_vi{option_viscous}_th{option_thermal}_dc{option_decoupling}_sh{option_shear_heating}    : Relative error: ')
    print_ph(f'                                                                                      T_11_11: {error[0]/vls[0]:.3e} ')
    print_ph(f'                                                                                      L2_A   : {error[1]/vls[1]:.3e} ')
    print_ph(f'                                                                                      L2_B   : {error[2]/vls[2]:.3e} ')


    return pass_flag

#-------------------------------------------------------------------------------
def perform_test(option_viscous:int = 0
                 ,option_thermal:int = 0
                 ,option_decoupling:int = 0
                 ,option_shear_heating:int = 0)->None:
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
    van_keken = 1
    # The input path for saving the results
    inp.path_test = f'{path_test}/Tests_Van_keken'

    # Geometrical input
    inp.cr = 0.0   # Overriding crust 
    inp.lc = 0.3   # relative amount of lower crust
    inp.ocr = 6.0e3  # Crustal thickness
    inp.lit_mt = 50e3  # Lithospheric mantle depth 
    inp.lab_d = inp.lit_mt  # depth of the lab 
    if option_decoupling == 0: 
    
        inp.decoupling = 50e3  # decoupling depth
        inp.decoupling_ctrl = 0 
    
    elif option_decoupling == 1: 
    
        inp.decoupling = 80e3 
        inp.decoupling_ctrl = 1 
        
    inp.ns_depth = 50e3
    inp.Tmax = 1300.0  # mantle potential temperature
    
    if option_shear_heating == 1: 
        inp.model_shear = 'SelfConsistent'
    
    elif option_shear_heating==0: 
        print_ph('!!!No Shear heating active!!!')
    
    inp.steady_state = 1
    
    
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


    inp.sname = f'Test_vi{option_viscous}_th{option_thermal}_dc{option_decoupling}_sh{option_shear_heating}'

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
def read_data_base(option_viscous:int = 0, option_thermal:int=0,option_decoupling:int = 0, option_shear_heating:int=0)->None:
    import h5py as h5 
    import os
    
    # File h5 that stores data of benchmarks
    file_h5 = f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken/benchmark_van_keken.h5'
    # Current test Name
    Test_name = f'Test_vi{option_viscous}_th{option_thermal}_dc{option_decoupling}_sh{option_shear_heating}'
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

    keyvalues = [T_11_11, L2_A, L2_B]
    
    pass_flag = evaluate_tests(option_viscous=option_viscous
                               ,option_thermal=option_thermal
                               ,option_decoupling=option_decoupling
                               ,option_shear_heating=option_shear_heating
                               ,keyvalues=keyvalues)
    


    db_vk1 = [np.mean(data[:,0]), np.min(data[:,0]), np.max(data[:,0])]
    db_vk2 = [np.mean(data[:,1]), np.min(data[:,1]), np.max(data[:,1])]
    db_vk3 = [np.mean(data[:,2]), np.min(data[:,2]), np.max(data[:,2])]


    
    print_ph(f'Test_viscous{option_viscous}, T_11_11 is {T_11_11:.4f}.')
    print_ph(f'                             Van Keken benchmark : mean T_11_11 = {db_vk1[0]:.2f}.')
    print_ph(f'                             Van Keken benchmark : range T_11_11 = {db_vk1[1]:.2f}-{db_vk1[2]:.2f}.')

    print_ph(f'Test_viscous{option_viscous}, L2_A is {L2_A:.4f}.')
    print_ph(f'                             Van Keken benchmark : mean L2_A = {db_vk2[0]:.2f}.')
    print_ph(f'                             Van Keken benchmark : range L2_A = {db_vk2[1]:.2f}-{db_vk2[2]:.2f}.')
    
    print_ph(f'Test_viscous{option_viscous}, L2_B is {L2_B:.4f}.')
    print_ph(f'                             Van Keken benchmark : mean L2_A = {db_vk3[0]:.2f}.')
    print_ph(f'                             Van Keken benchmark : range L2_A = {db_vk3[1]:.2f}-{db_vk3[2]:.2f}.')

    assert pass_flag
    
    print_ph('Test Passed ! ')
    
    f.close()
        
#-------------------------------------------------------------------------------
def test_isoviscous():
    # Test Van Keken 
    perform_test(option_viscous=0) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(0)
    # Remove folder after completing the test
    if DEBUG == False:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
def test_diffusion():
    # Test Van Keken
    perform_test(option_viscous=1)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(option_viscous=2)
    # Remove folder after completing the test
    if DEBUG == False:
       
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------   
def test_composite():
    # Test Van Keken
    perform_test(2)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(option_viscous=2)
    # Remove folder after completing the test
    if DEBUG == False:
        
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#------------------------------------------------------------------------------------
def test_composite_thermal_non_linear():
    # Test Van Keken
    perform_test(option_viscous=2,option_thermal=1)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(option_viscous=2,option_thermal=1)
    # Remove folder after completing the test
    if DEBUG == False:
        
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#----------------------------------------------------------------------------------
def test_composite_thermal_non_linear_crust():
    # Test Van Keken
    perform_test(option_viscous=2,option_thermal=2)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(option_viscous=2,option_thermal=2)
    # Remove folder after completing the test
    if DEBUG == False:
        
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')

#----------------------------------------------------------------------------------
def test_composite_decoupling():
    # Test Van Keken
    perform_test(option_viscous=0, option_decoupling=1)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(option_viscous=0, option_decoupling=1)
    # Remove folder after completing the test
    if DEBUG == False:
        
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
def test_composite_shear_heating():
    # Test Van Keken
    perform_test(option_viscous=2,option_decoupling=1,option_shear_heating=1)
    # Read Data Base and compare data
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(option_viscous=2,option_decoupling=1,option_shear_heating=1)
    # Remove folder after completing the test
    if DEBUG == False:
        
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    
    #test_isoviscous()
    
    test_diffusion()
    
    test_composite()
    
    #test_composite_thermal_non_linear()
    
    #test_composite_thermal_non_linear_crust()
    
    #test_composite_decoupling()
    
    test_composite_shear_heating()

    

#---------------------------------------------------------------------------------
