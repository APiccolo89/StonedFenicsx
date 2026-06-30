from stonedfenicsx.config.input_parser import Input
from stonedfenicsx.config.phase_db import PhInput
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.solver_module.solution_routine import solution_routine

def stoned_fenicsx(inp:Input,ph_in:PhInput) -> None:

    ctrl_sim, pdb, mesh, sc = configure_simulation(ph_in=ph_in,inp=inp)



    solution_routine(ctrl_sim,pdb,mesh,sc)



# ---------------------------------------------------------------------------#
