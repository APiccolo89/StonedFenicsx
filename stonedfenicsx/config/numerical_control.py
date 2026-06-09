from pathlib import Path
import numpy as np
from numpy import ndarray
from dataclasses import field, dataclass



@dataclass(slots=True)
class NumericalControls:
    it_max: int = 20
    it_inner_max: int = 10
    tol: float = 1e-4
    tol_innerpic: float = 1e-4
    relax: float = 0.9
    temp_top: float = 0.0          # surface temperature [°C]
    temp_max: float = 1300.0       # mantle temperature [°C]
    g: float = 9.81                # gravity [m/s^2]
    v_s: ndarray[np.float64] = field(
        default_factory=lambda: np.array([5.0, 0.0], dtype=np.float64)
    )                              # slab velocity [cm/yr]
    slab_age: float = 0.0          # [Myr]
    time_max: float = 30.0         # [Myr]
    dt: float = 500.0              # [yr]
    steady_state: int = 1
    time_dependent: int = 0
    time_dependent_v: int = 0
    decoupling_ctrl: int = 1 # 1 decoupled, 0 coupled
    model_shear: int = 1           # 1 linear, 0 nonlinear
    adiabatic_heating: int = 1     # REMOVE (?)
    stokes_solver_type: int = 1
    energy_solver_type: int = 1
    iterative_solver_tol: float = 1e-10
    eta_max : float = 1e26
    pressure_dependency: int = 1


@dataclass(slots=True)
class IOControls:
    test_name: str = ""
    path_save: str = ""
    sname: str = ""
    ts_out: int = 10
    dt_out: float = 1
    path_test: Path = field(init=False) 
    path_cached_information: Path = 'Cached_information'

    def generate_io(self) -> None:
        """Create the output directories if they don't exist."""
        Path(self.path_save).mkdir(parents=True, exist_ok=True)
        self.path_test = Path(self.path_save) / self.test_name
        self.path_cached_information = Path(self.path_save) / self.path_cached_information
        self.path_test.mkdir(parents=True, exist_ok=True)
        self.path_cached_information.mkdir(parents=True,exist_ok=True)
        print("Directory created:", self.path_test)


@dataclass(slots=True)
class CtrlLHS:
    """Parameters for the 1D thermal LHS problem."""
    nz: int = 200                  # number of vertical cells
    slab_tk: float = 130e3         # slab thickness [m]
    depth_melt: float = 0.0
    dt: float = 5e-3               # [Myr]
    c_age_plate: float = 50.0
    c_age_var: ndarray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 100.0], dtype=np.float64)
    )
    t_res: int = 1000              # temporal resolution
    recalculate: int = 0
    van_keken: int = 1
    d_rhs: float = -50e3
    k: float = 3.0
    rho: float = 3300.0
    cp: float = 1250.0
    dz: float = field(init=False)
    end_time:float = 180
    right_boundary : str = 'Continental'
    z: ndarray[np.float64] = field(init=False)
    lhs: ndarray[np.float64] = field(init=False)
    lhs_var: ndarray[np.float64] = field(init=False)
    t_res_vec: ndarray[np.float64] = field(init=False)
    self_consistent_flag: ndarray[np.int32] = 1

    def __post_init__(self) -> None:
        if self.dt > 0.1:
            raise ValueError("dt must be in Myr; this timestep blows up the system.")

        self.dz = self.slab_tk / self.nz
        self.z = np.zeros(self.nz, dtype=np.float64)
        self.lhs = np.zeros(self.nz, dtype=np.float64)
        self.lhs_var = np.zeros((self.nz, self.t_res), dtype=np.float64)
        self.t_res_vec = np.zeros(self.t_res, dtype=np.float64)

@dataclass(slots=True)
class time_dependent_evolution:
    constant_age: int = 1 
    constant_vel:int =  1
    current_age : float = None 
    current_vel : float = None 
    t_age : float = field(default_factory=lambda: np.array([0.0, 30.0]))
    t_vel : float =  field(default_factory=lambda: np.array([0.0, 30.0]))
    age_plate : float =  field(default_factory=lambda: np.array([0.0, 30.0]))
    vel_plate : float = field(default_factory=lambda: np.array([0.0, 30.0]))    
    
    @staticmethod
    def update_vel_age(int_t:list,vls:list,t:float)->float:
        """Function that update the current age or velocity

        Args: 
            int_t: list = interval of time 
            vls: list = start vel/age and end vel/age
            t: float = current time 
            
        """
        dt = int_t[1]-int_t[0]
        dp = vls[1]-vls[0]
        val = vls[0]+(dp/dt)*t
        
        val = max(vls[1], val) if dp < 0 else min(vls[1], val)
        
        return val
