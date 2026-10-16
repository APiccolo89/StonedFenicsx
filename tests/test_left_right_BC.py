import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from .global_variables import _PATH_, _TEST_

from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.config.thermal_boundary_bc_set import compute_half_space_cooling_model_analytical
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx


def configure_material_properties(option_thermal: int) -> int:


    path_test = Path(__file__).resolve().parents[0]
    # Path 2 imput fie
    path_input = f"{path_test}/input_tests.yaml"
    # Parse the input:
    # The input file is required to run a simulation. You can modify
    # it and parse the input and then call the function for running simulation.
    # Alternatively, you can generate the input file using it as blue print for the
    # common property of the simulation, and modify the produced object for personalising
    # the ensemble of simulations.
    inp, ph_input = parse_input(path_input)


    if option_thermal == 0:
        alpha_nameC = "Constant"
        alpha_nameM = "Constant"
        density_nameC = "Constant"
        density_nameM = "Constant"
        capacity_nameM = "Constant"
        capacity_nameC = "Constant"
        conductivity_nameM = "Constant"
        conductivity_nameC = "Constant"
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 0
    elif option_thermal == 1:
        alpha_nameC = "Mantle"
        alpha_nameM = "Mantle"
        density_nameC = "PT"
        density_nameM = "PT"
        capacity_nameM = "Mantle_Bernard_Ar_199x_FO_FA"
        capacity_nameC = "Mantle_Bernard_Ar_199x_FO_FA"
        conductivity_nameM = "Mantle_Richards_2018"
        conductivity_nameC = "Mantle_Richards_2018"
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 1
        inp.ctrl.pressure_dependency = 0

    elif option_thermal == 2 or option_thermal == 3:
        alpha_nameC = "Oceanic_crust"
        alpha_nameM = "Mantle"
        density_nameC = "PT"
        density_nameM = "PT"
        capacity_nameM = "Mantle_Bernard_Ar_199x_FO_FA"
        capacity_nameC = "Oceanic_crust"
        conductivity_nameM = "Mantle_Richards_2018"
        conductivity_nameC = "Crust_Richards_2018"
        rho0_M = 3300.0
        rho0_C = 3300.0
        radio_flag = 1
        inp.ctrl.pressure_dependency = 0
        inp.g_input.ocr = 6.0
        inp.g_input.cr = 6.0
        inp.g_input.lc = 0.0
        if option_thermal == 3:
            inp.ctrl.pressure_dependency = 1



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

    return {'ph_input':ph_input, 'inp':inp}

def test_thermal_boundary_conditions_linear():
    """
    Test the thermal boundary conditions for different options.
    Check the numeric solution with the analytical solution for a fixed age. 
    """
    data = configure_material_properties(option_thermal=0)
    
    inp = data['inp']
    
    inp.g_input.van_keken = False

    ctrl_sim, _, _, sc = configure_simulation(data['ph_input'], inp)
    
    temp_plate = deepcopy(ctrl_sim.ctrl_tbc.temperature_1d) * sc.temp - 273.15
    
    temp_plate_right = deepcopy(ctrl_sim.ctrl_tbc.temp_1d_right) * sc.temp - 273.15
    
    compute_half_space_cooling_model_analytical(ctrl_sim.ctrl_tbc, np.abs(ctrl_sim.ctrl_tbc.z))
    
    temp_plate_anl = deepcopy(ctrl_sim.ctrl_tbc.temperature_1d) * sc.temp - 273.15
    
    dT = temp_plate_anl - temp_plate
    
    print("Max difference between analytical and numerical solution: ", np.max(np.abs(dT)))
    print("Mean difference between analytical and numerical solution: ", np.mean(np.abs(dT)))
    
    dTdz_an = (ctrl_sim.ctrl_tbc.temp_max*sc.temp - 273.15) / (np.abs(ctrl_sim.ctrl_tbc.z_right[-1]) * sc.length/1e3)
    
    dTdz_num = np.diff(temp_plate_right) / np.abs(np.diff(ctrl_sim.ctrl_tbc.z_right * sc.length/1e3))
    
    total_difference = np.sum(np.abs(dTdz_an - dTdz_num))

    a =  np.isclose(np.max(np.abs(dT)), 10.62, rtol=1e-2)
    b =  np.isclose(np.mean(np.abs(dT)), 7.2, rtol=1e-2)
    c = np.isclose(total_difference, 1.05e-10, rtol=1e-2)
    assert a and b and c
    
    


if __name__ == "__main__":
    test_thermal_boundary_conditions_linear()





##
#    import matplotlib.pyplot as plt
#
#    fig = plt.figure()
#    ax = fig.gca()
#    temp_plate = ctrl_sim.ctrl_tbc.temperature_1d.copy() * sc.temp - 273.15
#    ax.plot(
#        ctrl_sim.ctrl_tbc.temperature_1d * sc.temp - 273.15,
#        ctrl_sim.ctrl_tbc.z * sc.length / 1e3,
#        c="firebrick",
#    )
#    ax.plot(
#        ctrl_sim.ctrl_tbc.temp_1d_right * sc.temp - 273.15,
#        ctrl_sim.ctrl_tbc.z_right * sc.length / 1e3,
#        c="forestgreen",
#    )
#    ctrl_tbc = compute_half_space_cooling_model_analytical(ctrl_sim.ctrl_tbc, np.abs(ctrl_sim.ctrl_tbc.z))
#    ax.plot(
#        ctrl_tbc.temperature_1d * sc.temp - 273.15,
#        ctrl_tbc.z * sc.length / 1e3,
#        c="cadetblue",
#    )
#
#    plt.show()
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(
#        ctrl_tbc.temperature_1d * sc.temp - 273.15 - temp_plate,
#        ctrl_tbc.z * sc.length / 1e3,
#        c="cadetblue",
#    )
