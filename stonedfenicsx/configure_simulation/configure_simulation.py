from .package_import import *
from typing import get_type_hints, get_origin, get_args, Callable
from .numerical_control import NumericalControls, IOControls, CtrlLHS
from .geometry import Geom_input, Mesh 
from stonedfenicsx.material_property.phase_db import PhaseDataBase
import beartype

#---------------------------------------------------------------------------------------------
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
    c_p: float = 1250.0
    k: float = 3.0
    rho0: float = 3300.0
    radio_flag: float = 0.0

    name_capacity: str = "Constant"
    name_conductivity: str = "Constant"
    name_alpha: str = "Constant"
    name_density: str = "Constant"
    alpha0: float = 3e-5  # constant thermal expansivity
    radiogenic_heat: float = 0.0
    # Internal heating
    radiative_conductivity: float = 0.0
#–----------------------------------------------------------------------------------
@dataclass
class PhInput:
    """Container of the phases"""
    shear_heating_disl_law: str 
    shear_heating_disl_ch: str 
    shear_heating_disl_phi: float
    subducting_plate_mantle: Phase
    oceanic_crust: Phase
    wedge_mantle: Phase
    overriding_mantle: Phase
    overriding_upper_crust: Phase
    overriding_lower_crust: Phase
#-----------------------------------------------------------------------------------
def filling_the_numerical_control(numerical_control_input)->NumericalControls:
    
    # Initialise the class with the default values: 
    ctrl = NumericalControls()
    
    
    
    
    
    
    return ctrl

#-----------------------------------------------------------------------------------
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
    phase_db: Phase
    g_input: Geom_input

#–-----------------------------------------------------------------------------------------------
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

    with open(f"{path}", "r") as f:
        input_file = yaml.safe_load(f)

    PhaseInput = PhInput()

    # Import numerical controls [basically structured data like numpy]
    NC = input_file["Input"]["NumericalControls"]  # Numerical controls
    IOCr = input_file["Input"]["InputOutputControl"]  # Input Controls
    LHS = input_file["Input"]["left_thermal_bc"]  # left boundary condition
    GEOM = input_file["Input"]["geometry"]  # Geometry
    MP = input_file["Input"]["Material_properties"]  # Material property
    SCAL = input_file["Input"]["scaling"]  # Scaling
    SHeating = input_file["Input"]["Shear_Heating"]  # Shear Heating

    # Fill the 
    ctrl = filling_the_numerical_control(NC)
    g_input = filling_the_geometrical_input()

    filling_the_input(NC, IOCr, LHS, GEOM, SCAL, SHeating, IP)
    filling_the_phase_data_base(MP, PhaseInput)

    return IP, Ph
#---------------------------------------------------------------------------------------------------
def cast_type(v: any, tp: any) -> any:

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
#---------------------------------------------------------------------------------
def filling_the_input(
    a: dict, b: dict, c: dict, d: dict, e: dict, f: dict, IP: Input
) -> Input:
    """Read input.yaml file, and update of the input class
    a : dictionary of subblock
    b : dictionary of subblock
    c : dictionary of subblock
    d : dictionary of subblock
    e : dictionary of subblock
    IP data class with default values, that are ovewritten by the yaml file
    Returns:
        IP: updated data classes


    Note: I divided the input in yaml file to explicitly state the number of data structure within
    the numerical code. The Input data class is a object dumb, that can be flexibly modified.
    I know that is redundant, but I believed that an user would find easier to modify one or two classes
    out of the yaml canvas.
    """

    def update_IP_file(IP0: Input, block: dict) -> Input:

        hints = get_type_hints(IP0.__class__)

        for k, v in block.items():
            tp = hints[k]
            setattr(IP0, k, cast_type(v, tp))

        return IP0

    for block in (a, b, c, d, e, f):

        IP = update_IP_file(IP, block)

    return IP
#-----------------------------------------------------------------------------------------
def filling_the_phase_data_base(MP: dict, Ph: Ph_input) -> Ph_input:
    """Function that fills the temporary class of the material properties

    Args:
        MP (dict): Material database coming from input.yaml
        Ph (Ph_input): Phase database

    Returns:
        Ph_input: Phase database
    """
    dict_phase_id = {
        "subducting_plate_mantle": 1,
        "oceanic_crust": 2,
        "wedge_mantle": 3,
        "overriding_mantle": 4,
        "overriding_upper_crust": 5,
        "overriding_lower_crust": 6,
    }

    # Loop over the MP items. MP items, is a multilevel dictionary
    for k, v in MP.items():
        buf = Phase()  # Prepare a Phase class to fill up with the new properties
        for j, vv in v.items():  # Loop over the properties of the class phase
            if vv != None:
                setattr(buf, j, vv)  # Set the attribute
            else:
                if j == "Hr" or j == "flag_radio":
                    vv = 0.0
                else:
                    vv = -1e23
        buf.name_phase = k
        buf.id = dict_phase_id[k]
        setattr(Ph, k, buf)  # Substitute the buf class with the default one
    return Ph

# Building the unit test for the configuration of the numerical simulation routine. 
if __name__ == '__main__':
    pass
