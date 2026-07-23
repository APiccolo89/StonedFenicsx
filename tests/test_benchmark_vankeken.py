# Import the required path for processing the simulation
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx
from pathlib import Path
import os 
import numpy as np 
from mpi4py import MPI
# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False
#-------------------------------------------------------------------------------
def perform_test(option_viscous=0,option_thermal=0):
    # Path 2 test
    path_test = os.path.dirname(os.path.realpath(__file__))
    # Path 2 imput fie
    path_input = f"{path_test}/input_tests.yaml"
    # Parse the input: 
    # The input file is required to run a simulation. You can modify  
    # it and parse the input and then call the function for running simulation. 
    # Alternatively, you can generate the input file using it as blue print for the 
    # common property of the simulation, and modify the produced object for personalising 
    # the ensemble of simulations. 
    inp,ph_input = parse_input(path_input)
    # Geometric Input: [inp.g_input.attributes -> change]
    inp.g_input.cr = .0 
    inp.g_input.lc = .0
    inp.g_input.ocr = 6.0 
    inp.g_input.lit_mt = 50.
    inp.g_input.lab_d = 50.
    inp.g_input.decoupling = .0 
    inp.g_input.van_keken = True 
    # Control 
    inp.ctrl.decoupling_ctrl = 0 
    inp.ctrl.steady_state = 1 
    # In this case, for testing the Van Keken benchmark, I opted to create a simple script
    # that has: option viscosity and thermal for testing several potential configuration. 
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
        capacity_nameM = 'Mantle_Bernard_Ar_199x_FO_FA'
        capacity_nameC = 'Mantle_Bernard_Ar_199x_FO_FA'
        conductivity_nameM = 'Mantle_Richards_2018'
        conductivity_nameC = 'Mantle_Richards_2018'
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 1 
        inp.ctrl.pressure_dependency = 0
        
    elif option_thermal == 2: 

        alpha_nameC = 'Oceanic_crust'
        alpha_nameM = 'Mantle'
        density_nameC = 'PT'
        density_nameM = 'PT'
        capacity_nameM = 'Mantle_Bernard_Ar_199x_FO_FA'
        capacity_nameC = 'Oceanic_crust'
        conductivity_nameM = 'Mantle_Richards_2018'
        conductivity_nameC = 'Crust_Richards_2018'
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 1 
        inp.ctrl.pressure_dependency = 0
        inp.g_input.ocr = 6.0 
        inp.g_input.cr = 6.0 
        inp.g_input.lc = 0.0

    if option_viscous == 0:
        name_diffusion = 'Constant'
        name_dislocation = 'Constant'              
    elif option_viscous == 1: 
        name_diffusion = 'VK_Diffusion_creep'
        name_dislocation = 'Constant'       
    elif option_viscous == 2: 
        name_diffusion = 'Constant'
        name_dislocation = 'VK_Dislocation_creep'     
        

    # ph_input contains the compositional phase -> you can modify them. The problem 
    # of kinematic simulations does not give a lot of freedom, and indeed, the possibility 
    # to have different rheologies is a design choiche to allow extension of the code 
    # in the future. Would be easier to start a new branch with more complex dynamic with 
    # config module. 

    # Modify the phase with the new data: 
    ph_input.subducting_plate_mantle.rho0 = rho0_M
    ph_input.subducting_plate_mantle.name_capacity = capacity_nameM
    ph_input.subducting_plate_mantle.name_conductivity = conductivity_nameM
    ph_input.subducting_plate_mantle.name_alpha = alpha_nameM
    ph_input.subducting_plate_mantle.name_density = density_nameM
    ph_input.subducting_plate_mantle.radiative_conductivity = radio_flag


    ph_input.oceanic_crust.rho0 = rho0_C
    ph_input.oceanic_crust.name_capacity = capacity_nameC
    ph_input.oceanic_crust.name_conductivity = conductivity_nameC
    ph_input.oceanic_crust.name_alpha = alpha_nameC
    ph_input.oceanic_crust.name_density = density_nameC
    ph_input.oceanic_crust.radiative_conductivity = radio_flag

    ph_input.wedge_mantle.name_diffusion = name_diffusion
    ph_input.wedge_mantle.name_dislocation = name_dislocation
    ph_input.wedge_mantle.rho0 = rho0_M
    ph_input.wedge_mantle.name_capacity = capacity_nameM 
    ph_input.wedge_mantle.name_conductivity = conductivity_nameM
    ph_input.wedge_mantle.name_alpha = alpha_nameM
    ph_input.wedge_mantle.name_density = density_nameM
    ph_input.wedge_mantle.radiative_conductivity = radio_flag

    ph_input.overriding_mantle.rho0 = rho0_M 
    ph_input.overriding_mantle.name_capacity = capacity_nameM
    ph_input.overriding_mantle.name_conductivity = conductivity_nameM
    ph_input.overriding_mantle.name_alpha = alpha_nameM
    ph_input.overriding_mantle.name_density = density_nameM
    ph_input.overriding_mantle.radiative_conductivity = radio_flag

    ph_input.overriding_upper_crust.rho0 = rho0_C 
    ph_input.overriding_upper_crust.name_capacity = capacity_nameC
    ph_input.overriding_upper_crust.name_conductivity = conductivity_nameC
    ph_input.overriding_upper_crust.name_alpha = alpha_nameC
    ph_input.overriding_upper_crust.name_density = density_nameC
    ph_input.overriding_upper_crust.radiative_conductivity = radio_flag

    ph_input.overriding_lower_crust.rho0 = rho0_C 
    ph_input.overriding_lower_crust.name_capacity = capacity_nameC
    ph_input.overriding_lower_crust.name_conductivity = conductivity_nameC
    ph_input.overriding_lower_crust.name_alpha = alpha_nameC
    ph_input.overriding_lower_crust.name_density = density_nameC
    ph_input.overriding_lower_crust.radiative_conductivity = radio_flag

    #ph_input.virtual_weak_zone.name_diffusion = 'Hirth_Wet_Olivine_disl'
    #ph_input.virtual_weak_zone.name_dislocation = 'Hirth_Wet_Olivine_disl' 

    # -> Important: where to save and the name of the test. You can fully automatise the creation of new
    # folder. 
    inp.ctrl_io.test_name = f'T_vi{option_viscous}_th{option_thermal}'
    inp.ctrl_io.path_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'VanKeken')
    

    # Initialise the input
    # After the user change the required data, and update the input and phase input, he must 
    # call this function, and run the simulation - hopefully, without throwing errors. 
    stoned_fenicsx(inp = inp, ph_in=ph_input)

#-------------------------------------------------------------------------------
def read_data_base(option_viscous,option_thermal=0):
    import h5py as h5 
    import os
    
    # File h5 that stores data of benchmarks
    file_h5 = f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken/benchmark_van_keken.h5'
    # Current test Name
    Test_name = f'T_vi{option_viscous}_th{option_thermal}'
    # Open the file 
    f = h5.File(file_h5)
    # Read the data from the database: 
    field_a = f'{Test_name}/T_11_11'
    T_11_11 = np.array(f[field_a])
    field_b = f'{Test_name}/L2_A'
    L2_A = np.array(f[field_b])
    field_c = f'{Test_name}/L2_B'
    L2_B = np.array(f[field_c])
    
    if option_viscous==0 and option_thermal == 0:  
        
        data = np.array([
        [397.55, 505.70, 850.50],
        [391.57, 511.09, 852.43],
        [387.78, 503.10, 852.97],
        [387.84, 503.13, 852.92],
        [389.39, 503.04, 851.68],
        [388.73, 504.03, 854.99],
        [390.40, 488.0, 847.70],
         ])
        v1 = 390.60
        v2 = 505.50
        v3 = 853.55
        
        # 
           
    if option_viscous == 1 and option_thermal == 0: 
        data = np.array([
        [570.30, 614.09, 1007.31],
        [580.52, 606.94, 1002.85],
        [580.66, 607.11, 1003.20],
        [577.59, 607.52, 1002.15],
        [581.30, 607.26, 1003.35],
        [584.20, 592.8, 1000.0],
        ])
        v1 = 563.13
        v2 = 601.45
        v3 = 999.1

    if option_viscous==2 and option_thermal == 0: 
        
        data = np.array([
        [550.17, 593.48, 994.11],
        [551.60, 608.85, 984.08],
        [582.65, 604.51, 998.71],
        [583.36, 605.11, 1000.01],
        [574.84, 603.80, 995.24],
        [583.11, 604.96, 1000.05],
        [585.70, 591.30, 996.60]
        ])
        v1 = 571.50
        v2 = 599.36
        v3 = 996.35 
    if option_thermal==1: 
        v1 = 554.58
        v2 = 608.76
        v3 = 940.40
    if option_thermal==2: 
        v1 = 595.05
        v2 = 635.08
        v3 = 960.09

    if option_thermal == 0: 
        db_vk1 = [np.mean(data[:,0]), np.min(data[:,0]), np.max(data[:,0])]
        db_vk2 = [np.mean(data[:,1]), np.min(data[:,1]), np.max(data[:,1])]
        db_vk3 = [np.mean(data[:,2]), np.min(data[:,2]), np.max(data[:,2])]
 
    test_1 = np.isclose(T_11_11, v1, rtol=1e-3, atol=1e-1)
    test_2 = np.isclose(L2_A, v2,rtol=1e-3, atol=1e-1)
    test_3 = np.isclose(L2_B, v3,rtol=1e-3,atol= 1e-1)
    
    
    print(f'Test_viscous{option_viscous}, T_11_11 is {T_11_11:.4f}. Tested against {v1:.4f}.')
    if option_thermal == 0:
        rel_err = (T_11_11 - db_vk1[1])/(db_vk1[2]-db_vk1[1])
        print(f'                             Van Keken benchmark : mean T_11_11 = {db_vk1[0]:.2f}.')
        print(f'                             Van Keken benchmark : range T_11_11 = {db_vk1[1]:.2f}-{db_vk1[2]:.2f}.')
        print(f'                             Van Keken benchmark : rel_err = {rel_err:.2f}')

    print(f'Test_viscous{option_viscous}, L2_A is {L2_A:.4f}. Tested against {v2:.4f}.')
    if option_thermal == 0: 
        rel_err = (L2_A - db_vk2[1])/(db_vk2[2]-db_vk2[1])

        print(f'                             Van Keken benchmark : mean L2_A = {db_vk2[0]:.2f}.')
        print(f'                             Van Keken benchmark : range L2_A = {db_vk2[1]:.2f}-{db_vk2[2]:.2f}.')
        print(f'                             Van Keken benchmark : rel_err = {rel_err:.2f}')

    print(f'Test_viscous{option_viscous}, L2_B is {L2_B:.4f}. Tested against {v3:.4f}.')
    if option_thermal == 0: 
        rel_err = (L2_B - db_vk3[1])/(db_vk3[2]-db_vk3[1])
        print(f'                             Van Keken benchmark : mean L2_A = {db_vk3[0]:.2f}.')
        print(f'                             Van Keken benchmark : range L2_A = {db_vk3[1]:.2f}-{db_vk3[2]:.2f}.')
        print(f'                             Van Keken benchmark : rel_err = {rel_err:.2f}')


    if test_1 and test_2 and test_3:         
        pass_flag = True
        print(f'Test_viscous{option_viscous} passed... ')
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
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')

def test_diffusion():
    # Test Van Keken 
    perform_test(1) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(1)
    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')
#-------------------------------------------------------------------------------

def test_composite():
    # Test Van Keken 
    perform_test(2) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(2)
    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')
#-------------------------------------------------------------------------------

def test_composite_NL_no_crust():
    # Test Van Keken 
    perform_test(2,1) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(2,1)
    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')
#-------------------------------------------------------------------------------

def test_composite_NL_crust():
    # Test Van Keken 
    perform_test(2,2) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(2,2)
    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    
    #test_isoviscous()

    #test_diffusion()

    #test_composite()
    
    test_composite_NL_no_crust()
    
    test_composite_NL_crust()
#---------------------------------------------------------------------------------
