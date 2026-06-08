from .package_import import *
from typing import get_type_hints, get_origin, get_args, Callable


# ---------------------------------------------------------------------------------------------------------
def timing_function(fun: Callable) -> Callable:
    """Extract the execution time of the function.

    Args:
        fun (Callable): The function for which the time of execution needs to be evaluated

    Returns:
        (callable): the output of the given **fun**.
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        time_a = timing.time()
        result = fun(*args, **kwargs)
        time_b = timing.time()
        dt = time_b - time_a
        dt = comm.allreduce(dt, op=MPI.MAX)
        if comm.rank == 0:
            if dt > 60.0:
                m, s = divmod(dt, 60)
                print(f".  {fun.__name__} took {m:.2f} min and {s:.2f} sec")
            if dt > 3600.0:
                m, s = divmod(dt, 60)
                h, m = divmod(m, 60)
                print(
                    f".  {fun.__name__} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec"
                )
            else:
                print(f".  {fun.__name__} took {dt:.2f} sec")
        return result

    return wrapper


def time_the_time(delta_time: float) -> float:
    """_summary_

    Args:
        delta_time (float): delta time -> time of execution

    Returns:
        global_dt(float): the effective time accross all the processors.
    """
    comm = MPI.COMM_WORLD
    global_dt = comm.allreduce(delta_time, op=MPI.MAX)
    return global_dt


# ---------------------------------------------------------------------------------------------------------
def print_ph(string: str) -> int:
    """function to print information. Print information only in one processor.
    Args:
        string (str): string to print

    Returns:
        int:
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(string)
        return 0
    return -1


def interpolate_from_sub_to_main(u_dest, u_start, cells, parent2child=0):
    """
    Interpolate the solution from the subdomain to the main domain.

    Parameters:
        u_slab (Function): The solution in the subdomain.
        u_global (Function): The solution in the main domain.
        M (Mesh): The mesh of the main domain.
        V (FunctionSpace): The function space of the main domain.
    """
    if parent2child == 0:
        a = np.arange(len(cells))
        b = cells
    else:
        a = cells
        b = np.arange(len(cells))

    u_dest.interpolate(u_start, cells0=a, cells1=b)

    return u_dest


def gather_vector(v):
    """_summary_

    Args:
        v (function): vector/function space to syncronise from local to global

    Returns:
        _type_:
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Suppose v is fem.Function(V) (not the UFL test function!)
    v.x.scatter_forward()  # update ghosts from owners

    # Get number of owned dofs (exclude ghosts)
    imap = v.function_space.dofmap.index_map
    n_owned = imap.size_local

    # Slice only owned part
    lv = np.asarray(v.x.array[:n_owned], dtype=np.float64)  # ensure dtype/contiguous

    # Gather element counts on root
    sizes = comm.gather(lv.size, root=0)

    if rank == 0:
        counts = np.asarray(sizes, dtype="i")
        displs = np.insert(np.cumsum(counts[:-1]), 0, 0).astype("i")
        gv = np.empty(int(counts.sum()), dtype=np.float64)
    else:
        counts = None
        displs = None
        gv = None

    # Gather variable-length arrays; only root provides recv buffers/metadata
    comm.Gatherv(lv, (gv, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        return lv
    else:
        return gv  # TO CHECK!!!!!


def gather_coordinates(V):
    """
    Gather DOF coordinates for a dolfinx FunctionSpace V to rank 0.
    Returns an (ndofs_global, gdim) array on rank 0, else None.
    """
    comm = V.mesh.comm
    gdim = V.mesh.geometry.dim

    # Local coordinates (shape: n_local_total x gdim). Slice to owned DOFs only.
    coords_local = V.tabulate_dof_coordinates()
    n_owned = V.dofmap.index_map.size_local
    coords_owned = coords_local[:n_owned, :]

    # Flatten to 1D buffer for Gatherv
    sendbuf = np.ascontiguousarray(coords_owned.ravel(), dtype=np.float64)

    # Gather row counts, then convert to element counts by * gdim
    rows_local = coords_owned.shape[0]
    rows_counts = comm.gather(rows_local, root=0)

    if comm.rank == 0:
        elem_counts = np.asarray(rows_counts, dtype="i") * gdim  # elements, not bytes
        elem_displs = np.insert(np.cumsum(elem_counts[:-1]), 0, 0).astype("i")
        recvbuf = np.empty(int(elem_counts.sum()), dtype=np.float64)
    else:
        elem_counts = None
        elem_displs = None
        recvbuf = None

    # Gather – use MPI.DOUBLE (not a NumPy dtype)
    comm.Gatherv(sendbuf, (recvbuf, elem_counts, elem_displs, MPI.DOUBLE), root=0)

    if comm.rank == 0:
        return recvbuf.reshape(-1, gdim)
    else:
        return None


# ----------------------------------------------------------------------------
def compute_strain_rate(u):
    """Compute strain rate from the velocity field u.

    Args:
        u (function): velocity field
    Returns:
        e (function): strain rate field
    """

    e = ufl.sym(ufl.grad(u))

    return e


# ---------------------------------------------------------------------------


def compute_eii(e):
    """Compute the second invariant of the strain rate from the strain rate field.

    Args:
        e (function): strain rate field

    Returns:
        e_ii (function): second invariant of the deviatoric strain rate
    """
    e_ii = ufl.sqrt(0.5 * ufl.inner(e, e))
    return e_ii


# ---------------------------------------------------------------------------


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

    Edif : float
        Activation energy for diffusion creep [J/mol].
    Vdif : float
        Activation volume for diffusion creep [m³/mol].
    Bdif : float
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
    Edis : float
        Activation energy for dislocation creep [J/mol].
    Vdis : float
        Activation volume for dislocation creep [m³/mol].
    Bdis : float
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
    radio : float
        Radiogenic heat production [W/m³] (or [Pa/s] if used as source in σ units).
    radio_flag : float
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
    Edif: float = -1e23
    Vdif: float = -1e23
    Bdif: float = -1e23

    name_dislocation: str = "Constant"
    n: float = -1e23
    Edis: float = -1e23
    Vdis: float = -1e23
    Bdis: float = -1e23

    eta: float = 1e20  # constant viscosity

    # Thermal properties
    Cp: float = 1250.0
    k: float = 3.0
    rho0: float = 3300.0
    radio_flag: float = 0.0

    name_capacity: str = "Constant"
    name_conductivity: str = "Constant"
    name_alpha: str = "Constant"
    name_density: str = "Constant"
    alpha0: float = 3e-5  # constant thermal expansivity
    Hr: float = 0.0
    # Internal heating
    radio: float = 0.0

@dataclass
class Ph_input:
    """Container of the phases"""

    subducting_plate_mantle: Phase
    oceanic_crust: Phase
    wedge_mantle: Phase
    overriding_mantle: Phase
    overriding_upper_crust: Phase
    overriding_lower_crust: Phase


def evaluate_material_property(
    expression: ufl.Expression, function_space: fem.FunctionSpace
) -> fem.function:
    """Transform an ufl expression into a function
    Args:
        expression (ufl.Expression): ufl expression
        function_space (fem.Function): the function space
    Returns:
        target_function: The final function
    """

    target_function = fem.Function(function_space)
    target_function.interpolate(
        fem.Expression(expression, function_space.element.interpolation_points())
    )
    return target_function


@dataclass(slots=True)
class Input:
    """Data class containing all the input. 
    The class stores all the information parsed from the input.yml file,
    and can be called to be modified in 
    other script for configure ensemble of numerical experiments. 

    """
    it_max: int = 20  # It max outer iteration and inner iteration
    it_inner_max: int = 10  # maximal inner iteration
    tol: float = 1e-5  # residual for the outer iteration
    relax: float = 0.9  # Correction factor for the solution
    temp_max: float = 1333.0  # maximum temperature (mantle temperature) [deg C]
    temp_top: float = 0.0  # surface temperature [deg C]
    g: float = 9.81  # gravity module vector [m/s^2]
    pressure_dependency: int = (
        1  # Flag to activate the pressure dependency of the material properties
    )
    v_s: NDArray[np.float64] = field(
        default_factory=lambda: np.array([5.0, 0.0])
    )  # slab velocity vector [cm/yr]

    slab_age: float = 50.0  # age of the slab [Myr]
    time_max: float = 2.0  # maximum time of timedependent problem [Myr]
    time_dependent_v: int = 0  # Flag to activate change in velocity
    steady_state: int = (
        1  # Flag to decide wether the problem is steady state or transient
    )
    slab_bc: int = 1  # change bc kind -> depecrated to remove
    tol_inner_pic: float = 1e-2  # Inner tollerance residual of picard iteration
    decoupling_ctrl: int = (
        1  # Activate the decoupling between overriding material and subducting slab
    )
    time_dependent: int = 1  # Depecrated
    dt_sim: float = 15000 / 1e6  # Myr #timestep simulation
    eta_max: float = 1.0e26  # Maximum viscosity
    adiabatic_heating: int = 0  # adiabatic heating flag -> to remove
    stokes_solver_type: str = "Direct"
    energy_solver_type: str = "Direct"
    iterative_solver_tol: float = 1e-9
    self_consistent_flag: int = 1
    
    model_shear: str = "NoShear"
<<<<<<< HEAD
    phi: float = 5.0 
    tau_min: float = 10e6 
    dislocation_creep_wz:str = 'Constant'
    eta_wz:float = 1.0e20
    tau_ref:float = 100e6
    eps_ref:float = 1e-11 
    n_ref:float = 3.0 
    decay_vis_wz:float = 5.0
    # -----------------------------------------------------------------------------------------------------
    # input/output control
    # -----------------------------------------------------------------------------------------------------
=======
    phi: float = 5.0
    cohesion: float = 0.0
    dislocation_creep_wz: str = "Constant"
    eta_wz: float = 1.0e20

>>>>>>> 0f03425 (Slow refractor of the numerical code: Starting from the configuration routine of the simulation.)
    test_name: str = "Output"
    sname: str = "Output"
    path_test: str = "../Results"

    length: float = 600e3
    stress: float = 1e9
    eta: float = 1e21
    temp: float = 1333.0

    nz: int = 108
    end_time: float = 180.0
    dt: float = 0.001
    recalculate: int = 1
    van_keken: int = 1
    c_age_plate: float = 50.0
    flag_radio: float = 0.0

    constant_age: int = 1
    constant_vel: int = 1
    t_age: float = field(default_factory=lambda: np.array([0.0, 30.0]))
    t_vel: float = field(default_factory=lambda: np.array([0.0, 30.0]))
    age_plate: float = field(default_factory=lambda: np.array([0.0, 30.0]))
    vel_plate: float = field(default_factory=lambda: np.array([0.0, 30.0]))
    
    x: NDArray[float] = field(default_factory=lambda: np.array([0.0, 660e3]))
    y: NDArray[float] = field(default_factory=lambda: np.array([-600e3, 0.0]))

    slab_tk: float = 130e3
    cr: float = 30e3
    ocr: float = 7e3
    lit_mt: float = 20e3
    lc: float = 0.3
    wc: float = 2.0e3

    ns_depth: float = 50e3
    lab_d: float = 100e3
    decoupling: float = 80e3
    resolution_normal: float = 2.0e3
    resolution_refine: float = 2.0e3
    transition: float = 10e3
    wz_tk: float = 2e3

    # Slab Geometry
    sub_lb: float = 300e3
    sub_dl: float = 1e3
    sub_theta0: float = 5
    sub_theta_max: float = 45
    sub_trench: NDArray[float] = field(default_factory=lambda: np.array([0.0, 660e3]))
    slab_type: str = "Costum"
    sub_path: str = "Not Defined"
    sub_constant_flag: int = 0

    def __post_init__(self) -> None:
        if self.sname == "Output" and self.test_name != "Output":
            self.sname = self.test_name


def parse_input(path: str) -> tuple[Input, Ph_input]:
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

    IP = Input()
    Ph = Ph_input()

    # Import numerical controls [basically structured data like numpy]
    NC = input_file["Input"]["NumericalControls"]  # Numerical controls
    IOCr = input_file["Input"]["InputOutputControl"]  # Input Controls
    LHS = input_file["Input"]["left_thermal_bc"]  # left boundary condition
    GEOM = input_file["Input"]["geometry"]  # Geometry
    MP = input_file["Input"]["Material_properties"]  # Material property
    SCAL = input_file["Input"]["scaling"]  # Scaling
    SHeating = input_file["Input"]["Shear_Heating"]  # Shear Heating

    filling_the_input(NC, IOCr, LHS, GEOM, SCAL, SHeating, IP)
    filling_the_phase_data_base(MP, Ph)

    return IP, Ph


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
