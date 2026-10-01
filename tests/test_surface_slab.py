
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.config.phase_db import read_capacity, read_diffusivity, read_expansivity, read_rheology
from stonedfenicsx.config.simulation_config import configure_simulation

from .global_variables import _PATH_, _TEST_, _TOL_





#TO DO LIST
# [] Check ribe geometry 
# [] Check parabolic geometry 
# [] Check realistic geometries 
# [] Check boundaries of the final mesh (?)
# workflow: produce slab surface -> save .pz with x,z,ell,theta -> compare with this file => success
#                   oceanic crust -> save.pz file 
#                   
# parabolic geometry => introduce analytical formulation and check directly with that 
# 