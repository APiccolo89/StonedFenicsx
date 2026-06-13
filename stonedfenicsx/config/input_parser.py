"""Modules"""
from stonedfenicsx.config.config_utils import update_ip_file
from stonedfenicsx.config.numerical_control import (
    NumericalControls,
    IOControls,
    CtrlTemperatureBC,
    CtrlKy
)
from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.phase_db import Phase,PhInput
from stonedfenicsx.utils import timing_function
from pathlib import Path
from dataclasses import field, dataclass
import numpy as np

@dataclass(slots=True)
class Input:
    """Data class containing all the input.
    The class stores all the information parsed from the input.yml file,
    and can be called to be modified in
    other script for configure ensemble of numerical experiments.

    """
    ctrl: NumericalControls = field(default_factory=NumericalControls)
    ctrl_io: IOControls = field(default_factory = IOControls)
    ctrl_tbc: CtrlTemperatureBC = field(default_factory = CtrlTemperatureBC)
    ctrl_ky: CtrlKy = field(default_factory=CtrlKy)
    g_input: GeomInput = field(default_factory=GeomInput)
    sc: Scal = field(default_factory=Scal)
# -----------------------------------------------------------------------------------
@timing_function
def parse_input(path: str) -> tuple[Input,PhInput]:
    """
    Read and parse a YAML input file.

    Parameters
    ----------
    path : str
        Path to the main input file.

    Returns
    -------
    Input
        Temporary container holding numerical and physical parameters.
        The returned object can be modified programmatically before
        starting the computation.

    Ph_input
        Temporary container storing material property definitions.

    Notes
    -----
    The YAML input file can be used directly as a standalone model
    configuration, or as a template for generating ensembles of models
    through external Python scripts. The typical workflow is to define
    a base scenario in YAML and then modify selected parameters
    programmatically.
    """
    # import yamlv
    import yaml

    with open(f"{path}", "r", encoding="utf-8") as f:
        input_file = yaml.safe_load(f)

        # Import numerical controls [basically structured data like numpy]
        nc = input_file["Input"]["NumericalControls"]  # Numerical controls
        iocr = input_file["Input"]["InputOutputControl"]  # Input Controls
        lhs = input_file["Input"][
            "thermal_boundary_condition"
        ]  # left boundary condition
        ky = input_file["Input"]['kinematic_boundary_condition']
        geom = input_file["Input"]["geometry"]  # Geometry
        mp = input_file["Input"]["Material_properties"]  # Material property
        scal = input_file["Input"]["scaling"]  # Scaling
        sheating = input_file["Input"]["Shear_Heating"]  # Shear Heating

    # Initialise the main classes:
    ctrl = NumericalControls()
    ctrl_tbc = CtrlTemperatureBC()
    ctrl_ky = CtrlKy()
    ctrl_io = IOControls()
    sc = Scal()
    g_input = GeomInput()

    # Fill the
    ctrl = update_ip_file(ctrl, nc)
    ctrl_io = update_ip_file(ctrl_io, iocr)
    g_input = update_ip_file(g_input, geom)
    sc = update_ip_file(sc, scal)
    sc.compute_the_derivative_scal()
    ctrl_tbc = update_ip_file(ctrl_tbc, lhs)
    ctrl_ky = update_ip_file(ctrl_ky,ky)

    input_obj = Input(ctrl=ctrl,ctrl_io = ctrl_io,ctrl_ky=ctrl_ky,ctrl_tbc=ctrl_tbc,g_input=g_input,sc=sc)

    ph_input = PhInput()

    ph_input = filling_the_phase_data_base(
        materialproperties=mp, shheating=sheating, phase_input=ph_input
    )
    return input_obj, ph_input




# -----------------------------------------------------------------------------------------
def filling_the_phase_data_base(
    materialproperties: dict, shheating: dict, phase_input: PhInput
) -> PhInput:
    """Function that fills the temporary class of the material properties

    Args:
        materialproperties (dict): Material database coming from input.yaml
        phase_input (PhInput): Phase database

    Returns:
        phase_input: Phase database
    """
    dict_phase_id = {
        "subducting_plate_mantle": 1,
        "oceanic_crust": 2,
        "wedge_mantle": 3,
        "overriding_mantle": 4,
        "overriding_upper_crust": 5,
        "overriding_lower_crust": 6,
    }

    phase_input = update_ip_file(phase_input, shheating)

    # Loop over the MP items. MP items, is a multilevel dictionary
    for k, v in materialproperties.items():
        buf = Phase()  # Prepare a Phase class to fill up with the new properties
        for j, vv in v.items():  # Loop over the properties of the class phase

            if vv is None:
                vv = (
                    0.0 if j in ("radiogenic_heat", "radiative_conductivity") else None
                )

            setattr(buf, j, vv)
        buf.name_phase = k
        buf.id_ph = dict_phase_id[k]
        setattr(phase_input, k, buf)  # Substitute the buf class with the default one
    return phase_input

def test_function():
    """Test function for debugging the configuration routines. 
    """
    # Find the main folder of the package
    pkg_root = Path(__file__)
    # Select the appropriate path for the input file
    input_file = Path(pkg_root.parents[2], "input.yaml")
    # parse the input file
    input_data, ph_in = parse_input(input_file)
    # Destroy the input data 
    del input_data
    del ph_in

# Building the unit test for the configuration of the numerical simulation routine.
if __name__ == "__main__":
    test_function()
