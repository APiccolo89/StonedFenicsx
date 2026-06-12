"""Modules"""
from typing import get_type_hints, get_origin, get_args
from stonedfenicsx.config.numerical_control import (
    NumericalControls,
    IOControls,
    CtrlTemperatureBC,
    CtrlKy
)
from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.utils import timing_function
from pathlib import Path
from dataclasses import field, dataclass
import numpy as np
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
dict_options = {"NoShear": 0, "SelfConsistent": 1}
dict_stokes = {"Direct": np.int32(1), "Iterative": np.int32(0)}
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------

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
    id_ph: int = 0
    # Viscosity / rheology
    name_diffusion: str = "Constant"
    e_dif: float | None = None
    v_dif: float | None = None
    b_dif: float | None = None

    name_dislocation: str = "Constant"
    n: float | None = None
    e_dis: float | None = None
    v_dis: float | None = None
    b_dis: float | None = None

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

    shear_heating_disl_law: str = "WetQuartzite"
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
    """function that convert the string into int-flag variable. 
    Mode_shear or the solvers option are defined as string in the main input file. 
    In the code these options are evaluated as a function of a 0-1 flag system. 
    The dictionaries at the top of the file, convert the string into this flag system.
    Certain value are naturally interpreted as a string (e.g., eta_max = 1e26). These variable
    are not transformed, as an other function would handle the effective conversion
    to the float number. 

    Args:
        k (str): key of the block 
        v (str): value 

    Returns:
        v(int|float|str): transformed value. 
    """
    if k == "model_shear":
        v = dict_options[v]
    elif k in ("stokes_solver_type", "energy_solver_type"):
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


def cast_type(v: any, tp: any) -> any:
    """Ensure that the typing of input is the same of the target class

    Args:
        v (any): value of the class member
        tp (any): type of the class member

    Returns:
        v: converted value
    """
    def check_bool(vbuf:bool|str|int)->bool: 
        """_summary_

        Args:
            vbuf (bool | str | int): check the bool branches

        Returns:
            bool: return a bool value
        """

        if isinstance(vbuf, bool):
            pass
        if isinstance(vbuf, str):
            if vbuf in ("true", "yes", "y", "True", "TRUE", "YES"):
                vbuf = True
            if vbuf in ("false", "no", "n", "NO", "FALSE", "False"):
                vbuf = False
        if isinstance(vbuf, int) or isinstance(vbuf, float):
            vbuf = bool(vbuf)

        return vbuf

    # Get the type of the input -> if it is a list -> list
    origin = get_origin(tp)
    # if origin is a compound object like lists, tuple, array -> get the argument of each of the element.
    args = get_args(tp)

    if tp is np.float64:
        v = np.float64(v)
    elif tp is bool:
        v = check_bool(v)
    elif origin is list:
        subtype = args[0] if args else object
        v = [cast_type(value, subtype) for value in v]
    elif origin is tuple:
        subtype = args[0] if args else object
        v = tuple(cast_type(value, subtype) for value in v)
    elif origin is np.ndarray:
        v =  np.asarray(v)
    else:
        v = tp(v)
    return v


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
