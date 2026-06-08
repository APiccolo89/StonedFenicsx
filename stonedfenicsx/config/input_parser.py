from typing import get_type_hints, get_origin, get_args, Callable
from stonedfenicsx.config.numerical_control import (
    NumericalControls,
    IOControls,
    CtrlLHS,
)
from stonedfenicsx.config.geometry import GeomInput, Mesh
from stonedfenicsx.material_property.phase_db import PhaseDataBase
from stonedfenicsx.config.scal import Scal
import beartype
from pathlib import Path
from dataclasses import field, dataclass
import numpy as np


dict_options = {"NoShear": 0, "SelfConsistent": 1}
dict_stokes = {"Direct": np.int32(1), "Iterative": np.int32(0)}


# ---------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Phase:
    """
    Phase: container for rheological and thermal material parameters.

    ------------------
    Rheology (viscosity)
    ------------------
    name_diffusion : str
        Diffusion creep flow law name.
        Options include (non-exhaustive):
          - 'Constant'                : constant viscosity
          - 'Hirth_Dry_Olivine_diff'  : Hirth & Kohlstedt (2003), dry olivine
          - 'Van_Keken_diff'          : Van Keken et al. (2008) style diffusion
          - 'Hirth_Wet_Olivine_diff'  : Hirth & Kohlstedt (2003), wet olivine

    e_dif : float
        Activation energy for diffusion creep [J/mol].
    v_dif : float
        Activation volume for diffusion creep [m³/mol].
    b_dif : float
        Pre-exponential factor for diffusion creep [1/Pa/s].

    name_dislocation : str
        Dislocation creep flow law name.
        Options include:
          - 'Constant'
          - 'Hirth_Dry_Olivine_disl'
          - 'Van_Keken_disl'
          - 'Hirth_Wet_Olivine_disl'

    n : float
        Stress exponent. **NB**: if you change this, Bdis must be updated consistently.
    e_dis : float
        Activation energy for dislocation creep [J/mol].
    v_dis : float
        Activation volume for dislocation creep [m³/mol].
    b_dis : float
        Pre-exponential factor for dislocation creep [1/Pa^n/s].

    eta : float
        Constant viscosity [Pa·s] (used if rheology is 'Constant').

    ------------------
    Thermal properties
    ------------------
    Cp : float
        constant heat capacity [J/kg/K].
    k : float
        constant thermal conductivity [W/m/K].
    rho0 : float
        Reference / constant density [kg/m³].

    name_capacity : str
        Heat capacity law.
        Options:
          - 'Constant'
          - 'Berman_Forsterite'
          - 'Berman_Fayalite'
          - 'Berman_Aranovich_Forsterite'
          - 'Berman_Aranovich_Fayalite'
          - 'Berman_Fo_Fa_01'
          - 'Bermann_Aranovich_Fo_Fa_0_1'
          - 'Oceanic_Crust'
          - 'ContinentalCrust' (not implemented / to be removed).

    name_density : str
        Density law.
        Options:
          - 'Constant' : ρ = ρ0.
          - 'PT'       : ρ(P,T) with constant bulk modulus K₀ ≈ 130e9 Pa and
                         thermal expansivity consistent with `name_alpha`.

    name_alpha : str
        Thermal expansivity law (α).
        Options:
          - 'Constant'      : α = 3e-5 K⁻¹.
          - 'Mantle'        : olivine / mantle α (e.g., Groose & Afonso 2013;
                              Richardson et al. 2020).
          - 'Oceanic_Crust' : basaltic crustal α.

    name_conductivity : str
        Thermal conductivity law (k).
        Options:
          - 'Constant'
          - 'Mantle'
          - 'Oceanic_Crust'

    ------------------
    Internal heating
    ------------------
    radiogenic_heat: float
        Radiogenic heat production [W/m³] (or [Pa/s] if used as source in σ units).
    radiative_conductivity: float
        Activation flag for radiogenic heating / radiative conductivity
        (0.0 = off, 1.0 = on, or a more general scaling factor).

    Notes
    -----
    This class is intended as a flexible container for building a PhaseDataBase.
    For your current kinematic slab work it may be somewhat overkill, but it
    should be reusable for other problems.
    """

    name_phase: str = "Undefined Phase"
    id: int = 0
    # Viscosity / rheology
    name_diffusion: str = "Constant"
    e_dif: float = -1e23
    v_dif: float = -1e23
    b_dif: float = -1e23

    name_dislocation: str = "Constant"
    n: float = -1e23
    e_dis: float = -1e23
    v_dis: float = -1e23
    b_dis: float = -1e23

    eta: float = 1e20  # constant viscosity

    # Thermal properties
    cp: float = 1250.0
    k: float = 3.0
    rho0: float = 3300.0

    name_capacity: str = "Constant"
    name_conductivity: str = "Constant"
    name_alpha: str = "Constant"
    name_density: str = "Constant"
    alpha0: float = 3e-5  # constant thermal expansivity
    radiogenic_heat: float = 0.0
    # Internal heating
    radiative_conductivity: float = 0.0


# –----------------------------------------------------------------------------------
@dataclass(slots=True)
class PhInput:
    """Container of the phases"""

    shear_heating_disl_law: str = 'WetQuartzite'
    shear_heating_disl_ch: float = 0.0
    shear_heating_disl_phi: float = 0.0
    subducting_plate_mantle: Phase = field(init=False)
    oceanic_crust: Phase = field(init=False)
    wedge_mantle: Phase = field(init=False)
    overriding_mantle: Phase = field(init=False)
    overriding_upper_crust: Phase = field(init=False)
    overriding_lower_crust: Phase = field(init=False)


# -----------------------------------------------------------------------------------
def correct_input(k: str, v: str) -> int | float | str:
    """_summary_

    Args:
        k (str): _description_
        v (str): _description_

    Returns:
        int|float|str: _description_
    """
    if k == "model_shear":
        v = dict_options[v]
    elif k == "stokes_solver_type" or k == "energy_solver_type":
        v = dict_stokes[v]

    return v


# ----------------------------------------------------------------------------------
def update_ip_file(obj: object, block: dict) -> object:
    """_summary_

    Args:
        obj (object): Target portion of the input (e.g., NumericalControls)
        block (dict): dictionary from the yaml file

    Returns:
        object: updated object
    """
    hints = get_type_hints(obj.__class__)
    for k, v in block.items():
        tp = hints[k]
        if isinstance(v, str):
            v = correct_input(k, v)
        setattr(obj, k, cast_type(v, tp))
    return obj


# -----------------------------------------------------------------------------------
@dataclass(slots=True)
class Input:
    """Data class containing all the input.
    The class stores all the information parsed from the input.yml file,
    and can be called to be modified in
    other script for configure ensemble of numerical experiments.

    """

    ctrl: NumericalControls
    ctrl_lhs: CtrlLHS
    ctrl_io: IOControls
    g_input: GeomInput


# –-----------------------------------------------------------------------------------------------
def parse_input(path: str) -> int:
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

    with open(f"{path}", "r") as f:
        input_file = yaml.safe_load(f)

    # Import numerical controls [basically structured data like numpy]
    nc = input_file["Input"]["NumericalControls"]  # Numerical controls
    iocr = input_file["Input"]["InputOutputControl"]  # Input Controls
    lhs = input_file["Input"]["thermal_boundary_condition"]  # left boundary condition
    geom = input_file["Input"]["geometry"]  # Geometry
    mp = input_file["Input"]["Material_properties"]  # Material property
    scal = input_file["Input"]["scaling"]  # Scaling
    sheating = input_file["Input"]["Shear_Heating"]  # Shear Heating

    # Initialise the main classes:
    ctrl = NumericalControls()
    ctrl_lhs = CtrlLHS()
    ctrl_io = IOControls()
    sc = Scal()
    g_input = GeomInput()

    # Fill the
    ctrl = update_ip_file(ctrl, nc)
    ctrl_io = update_ip_file(ctrl_io, iocr)
    g_input = update_ip_file(g_input, geom)
    sc = update_ip_file(sc, scal)
    sc.compute_the_derivative_scal()
    ctrl_lhs = update_ip_file(ctrl_lhs, lhs)

    input_data = Input(ctrl=ctrl, ctrl_io=ctrl_io, ctrl_lhs=ctrl_lhs, g_input=g_input)

    ph_input = PhInput()

    ph_input = filling_the_phase_data_base(
        materialproperties=mp, shheating=sheating, phase_input=ph_input
    )
    return input_data, ph_input


# ---------------------------------------------------------------------------------------------------
def cast_type(v: any, tp: any) -> any:
    """Ensure that the typing of input is the same of the target class

    Args:
        v (any): value of the class member
        tp (any): type of the class member

    Returns:
        v: converted value
    """

    # Get the type of the input -> if it is a list -> list
    origin = get_origin(tp)
    # if origin is a compound object like lists, tuple, array -> get the argument of each of the element.
    args = get_args(tp)

    if tp is str:
        return tp(v)

    if tp is int:
        return tp(v)

    if tp is float:
        return tp(v)

    if tp is np.float64:
        return np.float64(v)

    if tp is bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v in ("true", "yes", "y", "True", "TRUE", "YES"):
                return True
            if v in ("false", "no", "n", "NO", "FALSE", "False"):
                return False
        if isinstance(v, int) or isinstance(v, float):
            if int(v) == 1:
                return True
            else:
                return False

    if origin is list:
        subtype = args[0] if args else object
        return [cast_type(value, subtype) for value in v]

    if origin is tuple:
        subtype = args[0] if args else object
        return tuple(cast_type(value, subtype) for value in v)

    if origin is np.ndarray:
        return np.asarray(v)

    return tp(v)


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

    phase_input = update_ip_file(phase_input,shheating)       

    # Loop over the MP items. MP items, is a multilevel dictionary
    for k, v in materialproperties.items():
        buf = Phase()  # Prepare a Phase class to fill up with the new properties
        for j, vv in v.items():  # Loop over the properties of the class phase
            if vv is not None:
                setattr(buf, j, vv)  # Set the attribute
            else:
                if j == "radiogenic_heating" or j == "radiative_conductivity":
                    vv = 0.0
                else:
                    vv = -1e23
        buf.name_phase = k
        buf.id = dict_phase_id[k]
        setattr(phase_input, k, buf)  # Substitute the buf class with the default one
    return phase_input


# Building the unit test for the configuration of the numerical simulation routine.
if __name__ == "__main__":
    # Find the main folder of the package
    PKG_ROOT = Path(__file__)
    # Select the appropriate path for the input file
    input_file = Path(PKG_ROOT.parents[2], "input.yaml")
    # parse the input file
    input_data, ph_in = parse_input(input_file)

    pass
