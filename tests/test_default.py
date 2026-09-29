"""
Module with the Van Keken classical benchmarks. 
ALLERT0: These tests relies on the input_test.yml. Do not modify the input_test.yml otherwise
these test will fail. 
ALLERT1: The benchmark are sensitive to the resolution of the mesh. These tests, then, are 
valid specifically for the resolution that has been set. In the manuscript, the tests have 
been performed with an higher resolution. 
ALLERT2: The tests have been updated 19.09.2026 to have a lower resolution, because with high
resolution the test took too much time. 
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False
@pytest.fixture(scope="session", autouse=True)
def cleanup_output():
    yield
    pt = str(Path(__file__).resolve().parents[0])
    shutil.rmtree(f"{pt}/VanKeken", ignore_errors=True)
#-------------------------------------------------------------------------------
def perform_test():
    # Path 2 test
    path_test = Path(__file__).resolve().parents[0]
    # Path 2 imput fie
    path_input = f"{path_test}/input_tests.yaml"
    # Parse the input: 
    # The input file is required to run a simulation. You can modify  
    # it and parse the input and then call the function for running simulation. 
    # Alternatively, you can generate the input file using it as blue print for the 
    # common property of the simulation, and modify the produced object for personalising 
    # the ensemble of simulations. 
    inp,ph_input = parse_input(path_input)
    inp.ctrl_io.test_name = f'T_default'
    inp.ctrl_io.path_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'VanKeken')
    
    name_diffusion = 'VK_Diffusion_creep'
    name_dislocation = 'VK_Dislocation_creep'   
    
    ph_input.wedge_mantle.name_diffusion = name_diffusion
    ph_input.wedge_mantle.name_dislocation = name_dislocation

    # Initialise the input
    # After the user change the required data, and update the input and phase input, he must 
    # call this function, and run the simulation - hopefully, without throwing errors. 
    stoned_fenicsx(inp = inp, ph_in=ph_input)

        
def test_default():
    # Test Van Keken 
    perform_test() # IsoViscous
    pt = os.path.join(os.path.dirname(os.path.realpath(__file__)),'VanKeken')
    test='T_default'
    pt_test= Path(f"{pt}/{test}")
    pt_output = Path(f"{pt_test}/{'Steady_State.h5'}")
    assert pt_test.is_dir
    assert pt_output.is_file()
    # Read Data Base and compare data 
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    test_default()