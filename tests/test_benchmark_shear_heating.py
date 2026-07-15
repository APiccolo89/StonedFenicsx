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
def perform_test(phi=5.0,test_name='phi_5'):
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
    inp.g_input.sub_theta_max = 30
    inp.g_input.cr = .0 
    inp.g_input.lc = .0
    inp.g_input.ocr = 6.0 
    inp.g_input.lit_mt = 50.
    inp.g_input.lab_d = 50.
    inp.g_input.decoupling = 80.0 
    inp.g_input.van_keken = True 
    # Control 
    inp.ctrl.decoupling_ctrl = 0 
    inp.ctrl.steady_state = 1 
    # In this case, for testing the Van Keken benchmark, I opted to create a simple script
    # that has: option viscosity and thermal for testing several potential configuration. 
    
    
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
    
    name_diffusion = 'VK_Diffusion_creep'
    name_dislocation = 'VK_Dislocation_creep'     
        
    inp.ctrl.model_shear = 'SelfConsistent'
    inp.ctrl.decoupling_ctrl = 1
    ph_input.shear_heating_disl_phi = phi 
    ph_input.shear_heating_disl_law = "Wet_Quartzite_2001_Dislocation_creep"
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


    # -> Important: where to save and the name of the test. You can fully automatise the creation of new
    # folder. 
    inp.ctrl_io.test_name = f'T_{test_name}'
    inp.ctrl_io.path_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'VanKeken')
    

    # Initialise the input
    # After the user change the required data, and update the input and phase input, he must 
    # call this function, and run the simulation - hopefully, without throwing errors. 
    stoned_fenicsx(inp = inp, ph_in=ph_input)

#-------------------------------------------------------------------------------
def read_data_base(test_name):
    import h5py as h5 
    import os
    
    # File h5 that stores data of benchmarks
    file_h5 = f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken/benchmark_van_keken.h5'
    # Current test Name
    Test_name = f'T_{test_name}'
    # Open the file 
    f = h5.File(file_h5)
    # Read the data from the database: 
    field_a = f'{Test_name}/T_11_11'
    T_11_11 = np.array(f[field_a])
    field_b = f'{Test_name}/L2_A'
    L2_A = np.array(f[field_b])
    field_c = f'{Test_name}/L2_B'
    L2_B = np.array(f[field_c])
    
    if test_name == '_phi3':
        v1 = 251.17
        v2 = 581.29
        v3 = 843.03
    elif test_name == '_phi5':
        v1 = 0
        v2 = 0
        v3 = 0
    elif test_name =='_phi10':
        v1 = 0 
        v2 = 0 
        v3 = 0
    elif test_name =='_phi15':
        v1 = 0 
        v2 = 0 
        v3 = 0
    else: 
        raise ValueError('Wrong test')

 
    test_1 = np.isclose(T_11_11, v1, rtol=1e-3, atol=1e-1)
    test_2 = np.isclose(L2_A, v2,rtol=1e-3, atol=1e-1)
    test_3 = np.isclose(L2_B, v3,rtol=1e-3,atol= 1e-1)
    
    
    print(f'T {test_name}, T_11_11 is {T_11_11:.4f}. Tested against {v1:.4f}.')
  
    print(f'T {test_name}, L2_A is {L2_A:.4f}. Tested against {v2:.4f}.')
    
    print(f'T {test_name}, L2_B is {L2_B:.4f}. Tested against {v3:.4f}.')



    if test_1 and test_2 and test_3:         
        pass_flag = True
        print(f'Test {test_name} passed... ')
    else: 
        assert test_1 
        assert test_2
        assert test_3 
    
    f.close()
        
    return pass_flag
#-------------------------------------------------------------------------------
def test_phi(phi=0.0,test_name='_phi5'):
    # Test Van Keken 
    
    perform_test(phi=phi,test_name=test_name) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(test_name)
    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')


#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    #test_phi(phi=3.0, test_name='_phi3')
    #test_phi(phi=5.0,test_name='_phi5')
    test_phi(phi=10.0,test_name='_phi10')
    test_phi(phi=15.0,test_name='_phi15')
#---------------------------------------------------------------------------------
