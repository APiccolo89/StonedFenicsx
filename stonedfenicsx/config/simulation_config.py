from stonedfenicsx.config.input_parser import PhInput,Input,parse_input
from stonedfenicsx.config.scal import Scal 
from stonedfenicsx.config.numerical_control import NumericalControls, CtrlTemperatureBC, CtrlK
from stonedfenicsx.config.geometry import Mesh
from stonedfenicsx.config.phase_db import PhaseDataBase
from pathlib import Path
from stonedfenicsx.create_mesh.create_mesh import create_mesh 
from dolfinx.io import XDMFFile, gmshio

def configure_simulation(ph_in:PhInput,inp:Input)\
    -> int: #tuple[NumericalControls,CtrlLHS,IOControls,PhaseDataBase,Mesh, Scal]:
    """Function that configure the numerical simulation and scale the property accordingly.
    It takes the information from the input and generate the mesh.

    Args:
        ph_in (PhInput): phase input [pre-processed or from the input]
        inp (Input): input class storing the information of the simulation

    Returns:
        tuple[NumericalControl,CtrlLHS,IOControls,PhaseDataBase,Mesh]: computational classes
    """
    
    ctrl = inp.ctrl
    ctrl_io = inp.ctrl_io
    ctrl_lhs = inp.ctrl_lhs
    sc = inp.sc
    sc.compute_the_derivative_scal()
    # Final Check 
    g_input = inp.g_input
    g_input.check_class_consistency()
    # update the input/output
    ctrl_io.generate_io()
    # Create the mesh    
    mesh = create_mesh(ioctrl=ctrl_io, sc= sc,g_input=g_input,ctrl=ctrl)
    
    # Generate the right boundary and left boundary thermal boundary condition
    
    # Create the phase input
    
    # Scale the properties
    
    # print the information
    
    # release the new pre-processed class
    
    return 0 #ctrl,ctrl_lhs,ctrl_io, pdb, mesh, sc
    

def test_configure()->int:
    """Test Function for configuring the simulation
    It serves for debugging purpose and as a stand-alone test. 
    """

    # Find the main folder of the package
    pkg_root = Path(__file__)
    # Select the appropriate path for the input file
    input_file = Path(pkg_root.parents[2], "input.yaml")
    # parse the input file
    input_data, ph_in = parse_input(input_file)
    # Set the path of the tests 
    path_save = pkg_root.parents[2]/'Results'
    test_name = 'Mock_test'
    input_data.ctrl_io.test_name = test_name
    input_data.ctrl_io.path_save = path_save
    
    ctrl, ctrl_lhs, ctrl_io, pdb, mesh, sc = configure_simulation(ph_in,input_data)
    

    return 0


if __name__ == '__main__':
    
    test_configure()
    
