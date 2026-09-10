"""Modules"""

from pathlib import Path

from stonedfenicsx.config.geometry import Mesh
from stonedfenicsx.config.input_parser import Input, PhInput, parse_input
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.phase_db import PhaseDataBase, generate_phase_database
from stonedfenicsx.config.scal import Scal, scaling_simulation_physical
from stonedfenicsx.config.thermal_boundary_bc_set import configure_boundary_condition
from stonedfenicsx.create_mesh.create_mesh import create_mesh
from stonedfenicsx.utils import timing_function


@timing_function
def configure_simulation(
    ph_in: PhInput, inp: Input
) -> tuple[SimulationControls,PhaseDataBase,Mesh, Scal]:
    """Function that configure the numerical simulation and scale the property accordingly.
    It takes the information from the input and generate the mesh.

    Args:
        ph_in (PhInput): phase input [pre-processed or from the input]
        inp (Input): input class storing the information of the simulation

    Returns:
        tuple[SimulationControls,PhaseDaataBase,Mesh,Scal]: computational classes
    """

    ctrl = inp.ctrl
    inp.ctrl.convert_string()
    inp.ctrl.update_initial_guess()
    ctrl_io = inp.ctrl_io
    ctrl_tbc = inp.ctrl_tbc
    ctrl_ky = inp.ctrl_ky
    sc = inp.sc
    # Update the classes after the pre-processing
    sc.compute_the_derivative_scal()
    # Final Check
    g_input = inp.g_input
    g_input.check_class_consistency()
    ctrl_tbc.update_thermal_bc(g_input=g_input, ctrl=ctrl)
    ctrl_ky.check_kinematic_bc(ctrl=ctrl)
    # update the input/output
    ctrl_io.generate_io()
    # Create the mesh
    mesh = create_mesh(ioctrl=ctrl_io,g_input=g_input,ctrl=ctrl)
    # Generate the phase data base
    pdb = generate_phase_database(
        pressure_dependency=ctrl.pressure_dependency, eta_max=ctrl.eta_max, phin=ph_in
    ,scal_temp=sc.temp, scal_press=sc.stress)

    # Merge the controls into simulation controls
    ctrl_sim = SimulationControls( ctrl = ctrl
                                  ,ctrl_ky = ctrl_ky
                                  ,ctrl_tbc = ctrl_tbc
                                  ,ctrl_io = ctrl_io
                                  ,g_input = g_input)
    
    
    
   
    
    # Scale the simulation parameters
    scaling_simulation_physical(ctrl_sim=ctrl_sim,pdb=pdb,mesh=mesh,sc=sc)

    # Generate the right boundary and left boundary thermal boundary condition
    configure_boundary_condition(ctrl_tbc=ctrl_sim.ctrl_tbc
                                                          ,ctrl=ctrl_sim.ctrl
                                                          ,ioctrl = ctrl_sim.ctrl_io
                                                          ,sc=sc
                                                          ,pdb=pdb
                                                          ,g_input=mesh.g_input)
    # print the information

    # release the new pre-processed class

    return ctrl_sim, mesh, pdb, sc
