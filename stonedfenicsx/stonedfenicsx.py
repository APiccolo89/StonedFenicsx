from stonedfenicsx.config.input_parser import Input
from stonedfenicsx.config.phase_db import PhInput
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.solver_module.solution_routine import solution_routine
from stonedfenicsx.utils import timing_function

@timing_function
def stoned_fenicsx(inp:Input,ph_in:PhInput) -> None:
    """Main function : it calls the configure simulation with the input 
    data, then, send the simulation configuration into the main solver

    Args:
        inp (Input): _description_
        ph_in (PhInput): _description_
    """

    ctrl_sim, pdb, mesh, sc = configure_simulation(ph_in=ph_in,inp=inp)

    solution_routine(ctrl_sim,pdb,mesh,sc)



# ---------------------------------------------------------------------------#
