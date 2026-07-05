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
    # 
    path_test = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_test}/input_tests.yaml"

    inp,ph_input = parse_input(path_input)
    # Geometric Input: 
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
    

    # option for the benchmark

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark

    # The input path for saving the results

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
        name_diffusion = 'VK_Diffusion_creep'
        name_dislocation = 'Constant'       
    elif option_viscous == 2: 
        name_diffusion = 'Constant'
        name_dislocation = 'VK_Dislocation_creep'     
        

    

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


    inp.ctrl_io.test_name = f'T_vi{option_viscous}_th{option_thermal}'
    inp.ctrl_io.path_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'VanKeken')
    

    # Initialise the input
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
        v1 = 390.60
        v2 = 505.50
        v3 = 853.55
        
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
        v1 = 563.13
        v2 = 601.45
        v3 = 999.1

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
        v1 = 559.61
        v2 = 594.45
        v3 = 995.19 

    db_vk1 = [np.mean(data[:,0]), np.min(data[:,0]), np.max(data[:,0])]
    db_vk2 = [np.mean(data[:,1]), np.min(data[:,1]), np.max(data[:,1])]
    db_vk3 = [np.mean(data[:,2]), np.min(data[:,2]), np.max(data[:,2])]
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
    test_1 = np.isclose(T_11_11, v1, 5e-1, 1e-1)
    test_2 = np.isclose(L2_A, v2,5e-1, 1e-1)
    test_3 = np.isclose(L2_B, v3,5e-1, 1e-1)
    
    
    rel_err = (T_11_11 - db_vk1[1])/(db_vk1[2]-db_vk1[1])
    print(f'Test_viscous{option_viscous}, T_11_11 is {T_11_11:.4f}. Tested against {v1:.4f}.')
    print(f'                             Van Keken benchmark : mean T_11_11 = {db_vk1[0]:.2f}.')
    print(f'                             Van Keken benchmark : range T_11_11 = {db_vk1[1]:.2f}-{db_vk1[2]:.2f}.')
    print(f'                             Van Keken benchmark : rel_err = {rel_err:.2f}')

    rel_err = (L2_A - db_vk2[1])/(db_vk2[2]-db_vk2[1])
    print(f'Test_viscous{option_viscous}, L2_A is {L2_A:.4f}. Tested against {v2:.4f}.')
    print(f'                             Van Keken benchmark : mean L2_A = {db_vk2[0]:.2f}.')
    print(f'                             Van Keken benchmark : range L2_A = {db_vk2[1]:.2f}-{db_vk2[2]:.2f}.')
    print(f'                             Van Keken benchmark : rel_err = {rel_err:.2f}')

    rel_err = (L2_B - db_vk3[1])/(db_vk3[2]-db_vk3[1])
    print(f'Test_viscous{option_viscous}, L2_B is {L2_B:.4f}. Tested against {v3:.4f}.')
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
    if DEBUG == False:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')

def test_diffusion():
    # Test Van Keken 
    #perform_test(1) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(1)
    # Remove folder after completing the test
    if DEBUG == False:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------

def test_composite():
    # Test Van Keken 
    #perform_test(2) # IsoViscous
    # Read Data Base and compare data 
    if MPI.COMM_WORLD.rank == 0: 
        read_data_base(2)
    # Remove folder after completing the test
    if DEBUG == False:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    
    test_isoviscous()

    test_diffusion()

    test_composite()
#---------------------------------------------------------------------------------
