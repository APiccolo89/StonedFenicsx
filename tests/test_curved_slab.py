# Import the required path for processing the simulation
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx
from pathlib import Path
import os 
import numpy as np 
from mpi4py import MPI
import pytest
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
    inp.g_input.van_keken = False 
    inp.g_input.sub_constant_flag = False
    inp.g_input.slab_type = 'FromFile'
    inp.g_input.sub_path = '/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/data/Mexico_slab.pz'
    inp.g_input.resolution_normal = 10
    # Control 
    inp.ctrl.decoupling_ctrl = 0 
    inp.ctrl.steady_state = 0 
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

        
    elif option_thermal == 2 or option_thermal==3: 

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
        if option_thermal == 3: 
            inp.ctrl.pressure_dependency = 1


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
    inp.ctrl_io.path_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Curved')
    

    # Initialise the input
    # After the user change the required data, and update the input and phase input, he must 
    # call this function, and run the simulation - hopefully, without throwing errors. 
    stoned_fenicsx(inp = inp, ph_in=ph_input)



def test_composite():
    # Test Van Keken 
    perform_test(2) # IsoViscous

    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Curved')
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    test_composite()

#---------------------------------------------------------------------------------
