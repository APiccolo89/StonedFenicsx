from .package_import import *

"""  Note of developing: I know that the idea to generate an input file in yaml, for then filling up
    a temporary class that will be used to fill the real objects of computation seems convoluted and redundant.
    But 1st: I wanted to learn a bit the basic of yaml file
    2nd: I wanted to create a flexible way to generate an input that is modified by scripts that 
    then can costumise the simulation. Peeking in the repository of Hobson I had strong headache to 
    understand the flow of input and output, I tried to just convert the approach for common mortals
"""


#---------------------------------------------------------------------------------------------------------
def timing_function(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        time_A = timing.time()
        result = fun(*args, **kwargs)
        time_B = timing.time()
        dt = time_B - time_A
        dt = comm.allreduce(dt, op=MPI.MAX)
        if comm.rank == 0:
            if dt > 60.0:
                m, s = divmod(dt, 60)
                print(f".  {fun.__name__} took {m:.2f} min and {s:.2f} sec")
            if dt > 3600.0:
                m, s = divmod(dt, 60)
                h, m = divmod(m, 60)
                print(f".  {fun.__name__} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
            else:
                print(f".  {fun.__name__} took {dt:.2f} sec")
        return result
    return wrapper

def time_the_time(dt):
    comm = MPI.COMM_WORLD 
    global_dt = comm.allreduce(dt,op=MPI.MAX)
    
    return global_dt 

    
#---------------------------------------------------------------------------------------------------------
def print_ph(string):
    
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0: 
        print(string)

#---------------------------------------------------------------------------
def get_discrete_colormap(n_colors, base_cmap='cmc.lipari'):
    import numpy as np
    """
    Create a discrete colormap with `n_colors` from a given base colormap.
    
    Parameters:
        n_colors (int): Number of discrete colors.
        base_cmap (str or Colormap): Name of the matplotlib colormap to base on.
    
    Returns:
        matplotlib.colors.ListedColormap: A discrete version of the colormap.
        copied from internet
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_colors))
    return mcolors.ListedColormap(color_list, name=f'{base_cmap}_{n_colors}')
#---------------------------------------------------------------------------

def interpolate_from_sub_to_main(u_dest, u_start, cells,parent2child=0):
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
    
    
    u_dest.interpolate(u_start,cells0=a,cells1=b)

    return u_dest

def gather_vector(v): 
    
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
        counts = np.asarray(sizes, dtype='i')
        displs = np.insert(np.cumsum(counts[:-1]), 0, 0).astype('i')
        gv = np.empty(int(counts.sum()), dtype=np.float64)
    else:
        counts = None
        displs = None
        gv = None

    # Gather variable-length arrays; only root provides recv buffers/metadata
    comm.Gatherv(lv, (gv, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:

        return lv 


def gather_coordinates(V):
    """
    Gather DOF coordinates for a dolfinx FunctionSpace V to rank 0.
    Returns an (ndofs_global, gdim) array on rank 0, else None.
    """
    comm  = V.mesh.comm
    gdim  = V.mesh.geometry.dim

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
        elem_counts = np.asarray(rows_counts, dtype='i') * gdim           # elements, not bytes
        elem_displs = np.insert(np.cumsum(elem_counts[:-1]), 0, 0).astype('i')
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


#----------------------------------------------------------------------------          
def compute_strain_rate(u):
    
    e = ufl.sym(ufl.grad(u))
    
    return e  
#---------------------------------------------------------------------------

def compute_eII(e):
    from ufl import inner, sqrt
    e_II  = sqrt(0.5*inner(e, e))    
    return e_II
#---------------------------------------------------------------------------
    
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
    name_phase:str = "Undefined Phase"
    id : int = 0 
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
    alpha0      : float = 3e-5  # constant thermal expansivity
    Hr : float = 0.0
    # Internal heating
    radio: float = 0.0


class Ph_input():
    def __init__(self):
        subducting_plate_mantle : Phase = Phase()
        oceanic_crust : Phase = Phase()
        wedge_mantle : Phase = Phase()
        overriding_mantle : Phase = Phase()
        overriding_upper_crust : Phase = Phase()
        overriding_lower_crust : Phase = Phase()
        virtual_weak_zone : Phase = Phase()


def evaluate_material_property(expr, V):
    F = fem.Function(V)
    F.interpolate(fem.Expression(expr, V.element.interpolation_points()))
    return F 


@dataclass(slots=True)
class Input:
    # -----------------------------------------------------------------------------------------------------
    # Numerical Controls
    # -----------------------------------------------------------------------------------------------------
    it_max: int = 20             # It max outer iteration and inner iteration
    tol: float = 1e-5            # residual for the outer iteration
    relax: float = 0.9           # Correction factor for the solution
    Tmax: float = 1333.0         # maximum temperature (mantle temperature) [deg C]
    Ttop: float = 0.0            # surface temperature [deg C]
    g: float = 9.81              # gravity module vector [m/s^2]
  
    v_s: NDArray[float] = field(default_factory=lambda: np.array([5.0, 0.0]))  # slab velocity vector [cm/yr]

    slab_age: float = 50.0          # age of the slab [Myr]
    time_max: float = 2.0           # maximum time of timedependent problem [Myr]
    time_dependent_v: int = 0       # Flag to activate change in velocity 
    steady_state: int = 1           # Flag to decide wether the problem is steady state or transient
    slab_bc: int = 1                # change bc kind -> depecrated to remove
    tol_innerPic: float = 1e-2      # Inner tollerance residual of picard iteration
    tol_innerNew: float = 1e-5      # To Remove
    van_keken_case: int = 2         # Van keken case -> to remove
    decoupling_ctrl: int = 1        # Activate the decoupling between overriding material and subducting slab
    model_shear: str = "NoShear"    # Model for shear heating along the interaface
    phase_wz: int = 7               # Virtual weakzone phase
    time_dependent: int = 1         # Depecrated 
    dt_sim: float = 15000 / 1e6  # Myr #timestep simulation
    eta_max: float = 1.0e26 # Maximum viscosity 
    adiabatic_heating: int = 0  # adiabatic heating flag -> to remove
    stokes_solver_type : str = 'Direct'
    phi: float = 5.0            # Friction angle
    self_consistent_flag:int = 1 # incoming plate thermal structure: 0 -> half space cooling model ; 1 -> self-consistent with material properties
    # -----------------------------------------------------------------------------------------------------
    # input/output control
    # -----------------------------------------------------------------------------------------------------
    test_name: str = "Output"
    sname: str = "Output"
    path_test: str = "../Results"

    # -----------------------------------------------------------------------------------------------------
    # Scaling parameters
    # -----------------------------------------------------------------------------------------------------
    L: float = 600e3
    stress: float = 1e9
    eta: float = 1e21
    Temp: float = 1333.0

    # -----------------------------------------------------------------------------------------------------
    # Left boundary condition
    # -----------------------------------------------------------------------------------------------------
    nz: int = 108
    end_time: float = 180.0
    dt: float = 0.001
    recalculate: int = 1
    van_keken: int = 1
    c_age_plate: float = 50.0
    flag_radio: float = 0.0

    # -----------------------------------------------------------------------------------------------------
    # Phase/property selection flags + defaults
    # -----------------------------------------------------------------------------------------------------
    capacity_nameM: str = "Constant"
    conductivity_nameM: str = "Constant"
    density_nameM: str = "Constant"
    alpha_nameM: str = "Constant"

    capacity_nameC: str = "Constant"
    conductivity_nameC: str = "Constant"
    density_nameC: str = "Constant"
    alpha_nameC: str = "Constant"

    rho0_M: float = 3300.0
    rho0_C: float = 3300.0
    radio_flag: float = 0.0

    # -----------------------------------------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------------------------------------
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
    wz_tk : float = 2e3
    
    # Slab Geometry 
    sub_Lb : float = 300e3 
    sub_dl : float = 1e3 
    sub_theta0 : float = 5 
    sub_theta_max : float = 45
    sub_trench :NDArray[float] = field(default_factory=lambda: np.array([0.0, 660e3]))
    slab_type : str = 'Costum'
    sub_path:str = 'Not Defined'
     
    def __post_init__(self) -> None:
        if self.sname == "Output" and self.test_name != "Output":
            self.sname = self.test_name


def parse_input(path:str)->tuple[Input,Ph_input]:
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
    
    with open(f'{path}','r') as f:
        input_file = yaml.safe_load(f)
    
    
    IP = Input()
    Ph = Ph_input()
    
    # Import numerical controls [basically structured data like numpy]
    NC = input_file['Input']['NumericalControls'] # Numerical controls
    IOCr = input_file['Input']['InputOutputControl'] # Input Controls
    LHS = input_file['Input']['left_thermal_bc'] # left boundary condition
    GEOM = input_file['Input']['geometry'] # Geometry
    MP = input_file['Input']['Material_properties'] # Material property
    SCAL = input_file['Input']['scaling'] # Scaling 
    
    filling_the_input(NC,IOCr,LHS,GEOM,MP,SCAL,IP)
    filling_the_phase_data_base(MP,Ph)
    
    
    
    
    
    return IP, Ph 

def filling_the_input(a:dict,b:dict,c:dict,d:dict,e:dict,f:dict,IP:Input)->Input: 
    
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
    
    for k, v in a.items():
        if type(getattr(IP,k)) is not str:
            v = np.float64(v)
            
        setattr(IP, k, v)
    
    for k, v in b.items():
        setattr(IP, k, v)    
        
    for k, v in c.items():
        setattr(IP, k, v)
    
    for k, v in d.items():
        setattr(IP, k, v)
    
    return IP

def filling_the_phase_data_base(MP : dict,Ph : Ph_input)->Ph_input:
    """Function that fills the temporary class of the material properties

    Args:
        MP (dict): Material database coming from input.yaml
        Ph (Ph_input): Phase database 

    Returns:
        Ph_input: Phase database 
    """
    dict_phase_id={'subducting_plate_mantle' : 1,
                   'oceanic_crust' : 2,
                   'wedge_mantle' : 3,
                   'overriding_mantle' : 4,
                   'overriding_upper_crust' : 5,
                   'overriding_lower_crust' : 6,
                   'virtual_weak_zone' : 7}
    
    
    # Loop over the MP items. MP items, is a multilevel dictionary 
    for k,v in MP.items():
        buf = Phase()# Prepare a Phase class to fill up with the new properties 
        for j,vv in v.items(): # Loop over the properties of the class phase
            if vv != None:
                setattr(buf, j, vv) # Set the attribute
            else: 
                if j == 'Hr' or j=='flag_radio':
                    vv = 0.0
                else:
                    vv = -1e23
        buf.name_phase = k 
        buf.id = dict_phase_id[k]
        setattr(Ph,k,buf)# Substitute the buf class with the default one 
        
    
    
    return Ph