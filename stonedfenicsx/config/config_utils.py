
import numpy as np
from typing import get_type_hints, get_origin, get_args


dict_options = {"NoShear": 0, "SelfConsistent": 1}
dict_stokes = {"Direct": np.int32(1), "Iterative": np.int32(0)}
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
        
        if k not in hints: 
            raise ValueError(f"Unknown field '{k}' for {obj.__class__.__name__}")
        
        tp = hints[k]

        if isinstance(v, str):
            v = correct_input(k, v)
        setattr(obj, k, cast_type(v, tp))
    return obj

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
