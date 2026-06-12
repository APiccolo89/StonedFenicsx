"""Modules"""
from stonedfenicsx.config.input_parser import PhInput,Input,parse_input
from pathlib import Path
from stonedfenicsx.create_mesh.create_mesh import create_mesh
from stonedfenicsx.config.phase_db import generate_phase_database

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
    ctrl_tbc = inp.ctrl_tbc
    ctrl_ky = inp.ctrl_ky
    sc = inp.sc
    # Update the classes after the pre-processing
    sc.compute_the_derivative_scal()
    # Final Check
    g_input = inp.g_input
    g_input.check_class_consistency()
    ctrl_tbc.update_thermal_bc(g_input=g_input,ctrl=ctrl)
    ctrl_ky.check_kinematic_bc(ctrl=ctrl)
    # update the input/output
    ctrl_io.generate_io()
    # Create the mesh
    #mesh = create_mesh(ioctrl=ctrl_io, sc= sc,g_input=g_input,ctrl=ctrl)
    # Generate the phase data base
    pdb = generate_phase_database(pressure_dependency=ctrl.pressure_dependency,
                                  eta_max=ctrl.eta_max,
                                  phin=ph_in)
    
    # Call the function to not-dimensionalise 
    
    # 
    
    # Generate the right boundary and left boundary thermal boundary condition
            
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

    ph_in.oceanic_crust.name_alpha = 'Oceanic_crust'
    ph_in.oceanic_crust.name_capacity = 'Oceanic_crust'
    ph_in.oceanic_crust.radiative_conductivity = 1
    ph_in.oceanic_crust.rho0 = 2800
    ph_in.oceanic_crust.name_conductivity = 'Crust_Richards_2018'
    ph_in.oceanic_crust.name_density = 'PT'

    ph_in.subducting_plate_mantle.name_capacity = 'Mantle_Bernard_Ar_199x_FO_FA'
    ph_in.subducting_plate_mantle.name_conductivity = 'Mantle_Richards_2018'
    ph_in.subducting_plate_mantle.name_alpha = 'Mantle'
    ph_in.subducting_plate_mantle.rho0 = 3300
    ph_in.subducting_plate_mantle.name_density = 'PT'

    ph_in.wedge_mantle.name_capacity = 'Mantle_Bernard_Ar_199x_FO_FA'
    ph_in.wedge_mantle.name_conductivity = 'Mantle_Richards_2018'
    ph_in.wedge_mantle.name_alpha = 'Mantle'
    ph_in.wedge_mantle.rho0 = 3300
    ph_in.wedge_mantle.name_density = 'PT'
    ph_in.wedge_mantle.name_dislocation = 'VK_Dislocation_creep'
    ph_in.wedge_mantle.name_diffusion = 'VK_Diffusion_creep'


    configure_simulation(ph_in,input_data)

    return 0


if __name__ == '__main__':
    
    test_configure()
    
