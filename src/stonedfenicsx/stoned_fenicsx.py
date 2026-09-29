import subprocess
from pathlib import Path

from stonedfenicsx.config.input_parser import Input
from stonedfenicsx.config.phase_db import PhInput
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.solver_module.solution_routine import solution_routine
from stonedfenicsx.utils import print_ph, timing_function

_VERSION_ = "0.0.1"
_DATE_ = "26/08/2026"
_AUTHORS_="Andrea Piccolo, Timothy Craig" 

# From  https://medium.com/@wide4head/interfacing-git-from-python-abb55c548853
# Readapted, then I merged other sources: 
# run function collects the command information *args is basically arbitrary arguments
#run("rev-parse", "HEAD")) git -C repo rev-parse HEAD 
def git_info(repo=None):
    repo =  Path(__file__).parents[1]
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "short": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("status", "--porcelain") != "",
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # not a repo, or git not installed


def print_information_code() -> None:
    
    s=(
    "\033[0;37;48;2;0;0;0mS"
    "\033[0;97;48;2;0;0;0mΓ"
    "\033[0;37;48;2;0;0;0m0"
    "\033[0;97;48;2;0;0;0m∩Σ"
    "\033[0;37;48;2;0;0;0mD"
    "\033[0;90;48;2;0;0;0mƒ"
    "\033[0;97;48;2;0;0;0mΣ∩"
    "\033[0;90;48;2;0;0;0mi"
    "\033[0;97;48;2;0;0;0mC"
    "\033[0;37;48;2;0;0;0mSX"
    "\033[0m"
    )
    # https://patorjk.com/software/
    print_ph("================================================================")    
    print_ph(f"||||||||| ->        {s}       <- |||||||||")
    print_ph(f"      Authors = {_AUTHORS_}")
    print_ph(f"      Version = {_VERSION_}")
    print_ph(f"      Date = {_DATE_}")
    print_ph(" git informations:")
    meta_data = git_info()
    if meta_data is None:
        print_ph(" Branch = unavailable (not a git checkout)")
        print_ph(" Commit = unavailable")
        return
    dirty = " (uncommitted changes)" if meta_data["dirty"] else ""
    print_ph(f" Branch = {meta_data['branch']}")
    print_ph(f" Commit = {meta_data['short']}{dirty}")
    print_ph("main repo link: https://github.com/APiccolo89/StonedFenicsx")
    print_ph("================================================================")


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

    print_information_code()

    ctrl_sim, mesh, pdb, sc= configure_simulation(ph_in=ph_in,inp=inp)
    
    solution_routine(ctrl_sim=ctrl_sim,pdb=pdb,mesh=mesh,sc=sc)
