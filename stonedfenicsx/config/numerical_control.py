from pathlib import Path
from dataclasses import field, dataclass
import numpy as np
from numpy.typing import NDArray
from stonedfenicsx.config.geometry import GeomInput
# --- #
# --- #

@dataclass(slots=True)
class NumericalControls:#ctrl 
    """Numerical controls: class that stores the numerical control required for the simulation.
    main name: (ctrl) instance.  
    
    """
    it_max: int = 20
    it_inner_max: int = 10
    tol: float = 1e-4
    tol_innerpic: float = 1e-4
    relax: float = 0.9
    g: float = 9.81                # gravity [m/s^2]
    time_max: float = 30.0         # [Myr]
    dt: float = 500.0              # [yr]
    steady_state: int = 1
    decoupling_ctrl: int = 1 # 1 decoupled, 0 coupled
    model_shear: int = 1           # 1 linear, 0 nonlinear
    adiabatic_heating: int = 1     # REMOVE (?)
    stokes_solver_type: int = 1
    energy_solver_type: int = 1
    iterative_solver_tol: float = 1e-10
    eta_max : float = 1e26
    pressure_dependency: int = 1
# --- #
@dataclass(slots=True)
class IOControls: # ctrlio
    """IOControls: Input and output controls.
    Data class that stores the information of the outputs (simulation's results and temporary file)
    instance name: ctrlio
    """
    test_name: str =""
    path_save: str =""
    sname: str = "MockTest"
    ts_out: int = 10
    dt_out: float = 1
    path_test: Path = field(init=False)
    path_cached_information: Path = Path('Cached_information')

    def generate_io(self) -> None:
        """Create the output directories if they don't exist.
        method that creates the relative folders and create the path for the current simulation
        """
        Path(self.path_save).mkdir(parents=True, exist_ok=True)
        self.path_test = Path(self.path_save) / self.test_name
        self.path_cached_information = Path(self.path_save) / self.path_cached_information
        self.path_test.mkdir(parents=True, exist_ok=True)
        self.path_cached_information.mkdir(parents=True,exist_ok=True)
        print("Directory created:", self.path_test)
# --- #
@dataclass(slots=True)
class CTRLBC:
    """General class for the controls associated to the boundary condition
    """
    constant: int = 1
    interval_val: NDArray[np.float64] =  field(default_factory=lambda: np.array([5.0, 0.0])) #[age/velocity][Myr|cm/yr]
    interval_time:NDArray[np.float64] = field(default_factory=lambda: np.array([5.0, 0.0])) # [Myr]
    def check_time_variation(self, ctrl:NumericalControls):
        """Raise errors in case there are inconsistent option activated

        Args:
            ctrl (NumericalControls): _description_

        Raises:
            ValueError: Inconsistent input. The variation over time of a age or velocity of the slab
            is not consistent with steady state mode of the simulation
        """
        if ctrl.steady_state and not self.constant:
            raise ValueError(
                "Time-dependent BCs cannot be used in steady-state mode"
        )
        if (self.interval_time[0] == self.interval_time[1]) and not ctrl.steady_state:
            raise ValueError(
                "dt of the interval of time is equal to 0.0. Change the value."
        )
             
    def update_vel_age(self, t:float)->float:
        """update the current velocity and age
        Args:
            t (float): current time of the simulation
        Returns:
            age/velocity update.
        
        Usage: 
        ctrl_kin.vel = ctrl_kin.update_vel_age(t)-> direct update, it is better to not introduce
        if-else statement. 

        """
        dt = self.interval_time[1]-self.interval_time[0]
        dp = self.interval_val[1]-self.interval_val[0]
        val = self.interval_val[0]+(dp/dt)*(t-self.interval_time[0])
        val = max(self.interval_val[1], val) if dp < 0 else min(self.interval_val[1], val)
        return val
# --- #
@dataclass(slots=True)
class CtrlTemperatureBC(CTRLBC): # ctrltbc 
    """Parameters for the 1D thermal boundary condition 
    main instance name:ctrltbc
    """
    temp_top: float = 0.0          # surface temperature [°C]
    temp_max: float = 1300.0       # mantle temperature [°C]
    nz: int = 200                  # number of vertical cells
    dt: float = 5e-3               # [Myr]
    slab_age: float = 50.0
    recalculate: int = 0
    # Fixed value for reproducing the benchmarks 
    k: float = 3.1
    rho: float = 3300.0
    cp: float = 1250.0
    dz: float = field(init=False)
    end_time:float = 180
    nt: int = 1 
    right_boundary : str = 'Continental'
    right_age : float = 30.0 # Useful in case the right boundary condition is a oceanic lithosphere. 
    z: NDArray[np.float64] = field(init=False)
    z_right: NDArray[np.float64] = field(init=False)
    temp_1d_right: NDArray[np.float64] = field(init=False)
    temperature_1d: NDArray[np.float64] = field(init=False)
    temperature_2d_field: NDArray[np.float64] = field(init=False)
    t_res_vec: NDArray[np.float64] = field(init=False)
    self_consistent_flag: int = 1

    def update_thermal_bc(self,g_input:GeomInput,ctrl:NumericalControls) -> None:
        """_summary_

        Args:
            g_input (GeomInput): geometrical information

        Raises:
            ValueError: if dt is an absurd number, raise an error. Despite Crank-Nicolson is unconditionally stable, 
            this condition is not necessarely holding up with non-linear thermal properties. 
        """
        if self.dt > 0.1:
            raise ValueError("dt must be in Myr; this timestep blows up the system.")
        self.check_time_variation(ctrl)
        if self.slab_age != self.interval_val[0] and not ctrl.steady_state:
            raise ValueError('The input initial age must be the same of the first entry of the interval of values')

        # Prepare the main vector for computing the right and left boundary condition
        self.dz = g_input.slab_tk / self.nz
        self.z = np.zeros(self.nz, dtype=np.float64)
        self.z_right = np.zeros(self.nz, dtype=np.float64)
        self.nt  = int(self.end_time / self.dt + 1)
        self.temp_1d_right = np.zeros(self.nz, dtype=np.float64)
        self.temperature_1d = np.zeros(self.nz, dtype=np.float64)
        self.temperature_2d_field = np.zeros((self.nt, self.nz), dtype=np.float64)
        self.t_res_vec = np.zeros(self.nt, dtype=np.float64)

    def update_1d_vector_left(self):
        current_age_index = np.where(self.t_res_vec >= self.slab_age)[0][0]
        self.temperature_1d = self.temperature_2d_field[current_age_index,:]
        

# --- #
@dataclass(slots=True)
class CtrlKy(CTRLBC):
    """Control for the kinematic boundary condition 
    Args:
        CTRLBC (_type_): The superclass
    main instance name: ctrlky  
    """
    v_s:  NDArray[np.float64] = field(
        default_factory=lambda: np.array([5.0, 0.0], dtype=np.float64)
    )
    def check_kinematic_bc(self,ctrl:NumericalControls):
        """internal checks of the boundary condition

        Args:
            ctrl (NumericalControls): Main control of the numerical simulation

        Raises:
            ValueError: Consistency check:
            The input initial velocity must be the same of the first entry of the interval of values
        """
        self.check_time_variation(ctrl)
        if self.v_s[0] != self.interval_val[0]:
            raise ValueError('The input initial velocity must be the same of the first entry of the interval of values')

@dataclass(slots=True)
class SimulationControls:
    """Container of simulation controls:
    ctrl: Numerical controls
    ioctrl: I/O controls 
    ctrl_tbc: Thermal Boudary condition controls
    ctrl_ky: Kinematic boundary condition controls
    # name: ctrlsm 
    """
    ctrl: NumericalControls = field(default_factory=NumericalControls)
    ctrl_io: IOControls = field(default_factory=IOControls) 
    ctrl_tbc: CtrlTemperatureBC = field(default_factory=CtrlTemperatureBC)
    ctrl_ky: CtrlKy = field(default_factory=CtrlKy)