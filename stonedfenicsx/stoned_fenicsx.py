from stonedfenicsx.config.input_parser import Input
from stonedfenicsx.config.phase_db import PhInput
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.solver_module.solution_routine import solution_routine
from stonedfenicsx.utils import timing_function
from pathlib import Path

@timing_function
def stoned_fenicsx(inp:Input,ph_in:PhInput) -> None:
    """Top-level entry point for the stonedfenicsx subduction simulation.

    Sequences the two top-level stages of the simulation:
      1. configure_simulation -- non-dimensionalises all inputs, builds the
         mesh and sub-meshes, and constructs the material-property database.
      2. solution_routine -- allocates FEM problem objects and drives the
         coupled Picard / time-stepping loop through to completion.

    Args:
        inp (Input): Parsed YAML input containing numerical controls, I/O
            settings, thermal boundary conditions, and kinematic boundary
            conditions.
        ph_in (PhInput): Parsed material-property input for all phases
            (wedge mantle, slab mantle, oceanic crust, overriding crust, etc.).
    """

    ctrl_sim, mesh, pdb, sc= configure_simulation(ph_in=ph_in,inp=inp)

    solution_routine(ctrl_sim=ctrl_sim,pdb=pdb,mesh=mesh,sc=sc)



# ---------------------------------------------------------------------------#


def test_function():
    """Test function to check the functionality of the main function.
    It creates a mock input and phase input, then calls the main function.
    """
    from stonedfenicsx.config.input_parser import  parse_input
    from stonedfenicsx.config.config_utils import dict_shear_modes
    # Create a mock input
        # Find the main folder of the package
    pkg_root = Path(__file__)
    # Select the appropriate path for the input file
    input_file = Path(pkg_root.parents[1], "input.yaml")
    # parse the input file
    input_data, ph_in = parse_input(input_file)
    # Set the path of the tests
    path_save = pkg_root.parents[1] / "Results"
    test_name = "Mock_test"
    input_data.ctrl_io.test_name = test_name
    input_data.ctrl_io.path_save = path_save
    input_data.ctrl_tbc.slab_age = 100.0
    #
    ph_in.oceanic_crust.name_alpha = "Oceanic_crust"
    ph_in.oceanic_crust.name_capacity = "Oceanic_crust"
    ph_in.oceanic_crust.radiative_conductivity = 1
    ph_in.oceanic_crust.rho0 = 3300
    ph_in.oceanic_crust.name_conductivity = "Crust_Richards_2018"
    ph_in.oceanic_crust.name_density = "PT"
    ph_in.oceanic_crust.radiogenic_heat = 0.25e-6

    ph_in.subducting_plate_mantle.name_capacity = "Mantle_Bernard_Ar_199x_FO_FA"
    ph_in.subducting_plate_mantle.name_conductivity = "Mantle_Richards_2018"
    ph_in.subducting_plate_mantle.name_alpha = "Mantle"
    ph_in.subducting_plate_mantle.rho0 = 3300
    ph_in.subducting_plate_mantle.name_density = "PT"

    ph_in.wedge_mantle.name_capacity = "Mantle_Bernard_Ar_199x_FO_FA"
    ph_in.wedge_mantle.name_conductivity = "Mantle_Richards_2018"
    ph_in.wedge_mantle.name_alpha = "Mantle"
    ph_in.wedge_mantle.rho0 = 3300
    ph_in.wedge_mantle.name_density = "PT"
    ph_in.wedge_mantle.name_dislocation = "VK_Dislocation_creep"
    ph_in.wedge_mantle.name_diffusion = "VK_Diffusion_creep"
    
    ph_in.overriding_lower_crust.radiogenic_heat = 0.5e-6
    ph_in.overriding_upper_crust.radiogenic_heat = 1.0e-6
    
    input_data.ctrl.model_shear = dict_shear_modes["SelfConsistent"]
    
    stoned_fenicsx(input_data,ph_in)
    
if __name__ == "__main__":
    test_function()