from stonedfenicsx.config.phase_db import PhInput
from stonedfenicsx.config.input_parser import Input
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.utils import print_ph


def stonedfenicsx(inp:Input,ph_in:PhInput)->str:
    """_summary_

    Args:
        inp (Input): _description_
        ph_in (PhInput): _description_

    Returns:
        str: _description_
    """
    

    ctrl_sim, pdb, mesh, sc = configure_simulation(ph_in=ph_in,inp = inp)
    
    # Running simulation
    
    
    
    
    return print_ph('stonedFEnicsX: The End. ')