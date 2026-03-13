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
import argparse


# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False

#-------------------------------------------------------------------------------
def perform_test(args:argparse.Namespace = None, plate:str = 'Tonga'):
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
    inp.path_test = f'{path_test}/Tonga'

    # Velocity of slab 
    inp.v_s = np.array([args.convergent_velocity,0],dtype=np.float64)
    inp.time_max = args.max_time
    inp.decoupling_ctrl = 1
    inp.model_shear = args.shear_heating
    # Geometrical input
    inp.cr = 7.0e3   # Overriding crust 
    inp.lc = 0.0   # relative amount of lower crust
    inp.ocr = 6.0e3  # Crustal thickness
    inp.lit_mt = 50e3  # Lithospheric mantle depth 
    inp.lab_d = inp.lit_mt  # depth of the lab 
    inp.decoupling = 80e3  # decoupling depth
    inp.ns_depth = 50e3
    inp.Tmax = 1300.0  # mantle potential temperature
    inp.steady_state = args.steady_state
    if args.steady_state == 0: 
        inp.tol = 1e-2 
        inp.tol_innerPic = 1e-2 
        inp.it_max = 5
    inp.slab_type = 'File'
    inp.sub_path = f'/Users/wlnw570/Work/Leeds/Fenics_tutorial/example_slab_surfaces/{plate}_slab.pz'
    print_ph('Starting the benchmark tests with different options')

    alpha_nameC = 'Crust'
    alpha_nameM = 'Mantle'
    density_nameC = 'PT'
    density_nameM = 'PT'
    capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'
    capacity_nameC = 'Oceanic_Crust'
    conductivity_nameM = 'Mantle'
    conductivity_nameC = 'Oceanic_Crust'
    rho0_M = 3300.0
    rho0_C = 2800.0
    radio_flag = 1 


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


    inp.sname = f'{plate}_tmax{int(np.floor(args.max_time))}_vc{int(np.floor(args.convergent_velocity))}_{args.shear_heating}_pr{MPI.COMM_WORLD.Get_size():d}'

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
def test_tonga(args:argparse.Namespace=None,plate:str='Tonga'):
    # Test Van Keken
    perform_test(args, plate)
    # Read Data Base and compare data
    #if MPI.COMM_WORLD.rank == 0: 
    #    read_data_base(2)
    # Remove folder after completing the test
    #if DEBUG == False:
        
   #     os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken')
#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--convergent_velocity",type=float,default=8.0)
    parser.add_argument("--steady_state",type=int,default=0)
    parser.add_argument("--shear_heating",type=str,default='SelfConsistent')
    parser.add_argument("--max_time",type=float,default=30.0)
    args = parser.parse_args()
    
    test_tonga(args)
#---------------------------------------------------------------------------------
