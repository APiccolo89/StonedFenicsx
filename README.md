# StonedFEniCSx

A FEniCSx (dolfinx)-based FEM package for simulating the thermal and mechanical evolution of a 2D subduction zone: coupled steady-state/time-dependent thermal, Stokes (velocity–pressure), and lithostatic pressure problems on wedge, slab, and global sub-domains, with temperature- and pressure-dependent rheology and shear heating.

The project started as a Python script built on the [FieldStone](https://cedricthieulot.net/fieldstone.html) educational framework and has since grown into a structured, class-based FEM package.

> Research code developed at the University of Leeds. Solo-maintained; interfaces may still change between branches.

## Physical model

Each outer (Picard) iteration couples three problems over the subduction domain (surface to 600 km depth, split into slab, mantle wedge, and overriding-plate sub-domains — see `docs/Computational_domain.md`):

- **Thermal** — the energy equation (diffusion + SUPG-stabilised advection), optionally including radiogenic heating and frictional (shear) heating along the slab interface.
- **Lithostatic pressure** — a global hydrostatic/lithostatic pressure integration that feeds temperature- and pressure-dependent material properties into the other two problems.
- **Stokes (wedge / slab)** — velocity–pressure momentum + continuity, with kinematic (moving-wall) slab boundary conditions and a Nitsche free-slip condition on the slab's basal interface.

Material properties (density, heat capacity, thermal conductivity, dislocation/diffusion-creep and plastic rheology) are evaluated once per Picard iteration and held "frozen" for that iteration's linear solve. Both steady-state and time-dependent solves are supported; the steady-state path is the one currently validated against benchmarks (see below).

## Requirements

- Python >= 3.10, < 3.14
- [dolfinx](https://github.com/FEniCS/dolfinx) 0.9.x, with PETSc and MPI
- mpi4py, petsc4py
- numpy, scipy, gmsh, meshio, numba, h5py, shapely, pandas, pyyaml

Pinned versions are tracked in `stoned_environment.yml` (conda) and `pyproject.toml`/`pyproject_HPC.toml` (pip).

## Installation

### Local, with conda

```bash
conda env create -f stoned_environment.yml
conda activate stoned_fenicsx
pip install --no-deps -e .
```

### HPC, with a Spack-provided dolfinx

On a cluster, dolfinx/PETSc/MPI are usually best obtained from the module system rather than conda. `pyproject_HPC.toml` targets this setup (dolfinx supplied externally, `pip install -e .` into a venv layered on top of loaded Spack packages). See [`HPC_read.md`](HPC_read.md) for a full worked example on the University of Leeds Aire cluster, including gotchas around gmsh's OSMesa/GLU dependency and h5py/numpy ABI compatibility with the cluster's HDF5.

## Quick start

A simulation is configured with two YAML-parsed inputs — numerical/I-O/thermal/kinematic controls, and per-phase material properties — which drive `stonedfenicsx.stoned_fenicsx`:

```python
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

input_data, ph_in = parse_input("input.yaml")
stoned_fenicsx(input_data, ph_in)
```

`input.yaml` at the repo root is a commented example covering units, numerical controls, shear-heating options, and thermal/kinematic boundary conditions. `stonedfenicsx/stoned_fenicsx.py::test_function` shows a fully scripted example that also overrides material properties in code after parsing. `examples/` contains region-specific driver scripts (`Japan_slab.py`, `Mexico_slab.py`, `Chile_slab.py`, `Tonga_slab copy.py`) built the same way.

Results are written under `Results/<test_name>/` as XDMF/HDF5 fields, plus cached material-property lookups.

Simulations are MPI-parallel; run under `mpirun`/`srun` for multi-rank execution (see the example SLURM script in `HPC_read.md`).

## Running the tests

```bash
pytest tests/
```

The main physical validation is `tests/test_benchmark_shear_heating.py` and `tests/test_benchmark_vankeken.py`, which reproduce reference results from the Van Keken et al. subduction zone benchmark suite (`tests/VanKeken/`) across viscosity/thermal configurations and, for the shear-heating case, several friction-angle (`phi`) values.

## Package layout

```
stonedfenicsx/
├── config/            # Input parsing, non-dimensionalisation (Scal), phase/material database,
│                       # numerical controls, geometry, thermal/kinematic boundary conditions
├── create_mesh/        # gmsh-based mesh generation and slab-surface geometry handling
├── material_property/  # Temperature/pressure-dependent density, heat capacity, conductivity,
│                       # rheology (dislocation/diffusion creep, plasticity)
├── solver_module/       # Problem classes (Global_thermal, Global_pressure, Wedge, Slab),
│                       # PETSc solver wrappers (ScalarSolver, SolverStokes), and the
│                       # outer Picard/time-stepping loop (solution_routine)
├── output.py           # XDMF/HDF5 result writing
└── stoned_fenicsx.py    # Top-level entry point: configure_simulation() then solution_routine()
```

## Documentation

The documentation of the code - which is still in a working progress can be found here: [StonedFEniCSx](https://apiccolo89.github.io/StonedFenicsx/index.html)

# How to use 

The code is organized to always require the definition of an *input_file.yml*. The blue-print of the input file can be found in the main folder of the package. 

The units of measure in the input code are always: **Myr**, **km**, **cm/yr**, **deg** and **degC** for time, length, velocity, angles and temperatures respectively; the only exception is for the *scaling* options; in this case, the unit of measure are **m**, **Pa**, **Pa s** and **deg C** The conversions in SI units and the relative scaling are performed internally during the configuration step of the simulation. 
## Input File 
The input file is divided into 8 subsections: 

- **NumericalControls**: the parameters that control the behaviour of the simulation.
- **ShearHeating**: the set of parameters that controls the internal boundary heat source (the parametrised shear heating along the seismogenetic zone of the subduction plate).
- **InputOutputControl**: the set of parameters that controls where to save the output files, and under which condition to release a timestep. 
- **scaling**: the set of SI units used to scale the parameters of the simulation. 
- **thermal_boundary_condition**: the thermal boundary condition configuration parameters.
- **kinematic_boundary_condition**: the kinematic boundary condition configuration parameters. 
- **Material_properties**: the material property input parameters.
- **geometry**: the geometrical configuration parameters. 

### Numerical Controls 
```
    it_max: 30 # Maximum number of iterations
    tol: 5e-5 # Tolerance of the problem
    relax: 0.9 # Relax factor to update the solution
    g: 9.81 # Module of the gravitational acceleration 
    eta_max: 1.0e26 # Maximum viscosity of the system 
    time_max: 2.0  # Maximum time of the simulation [SI = Myr]
    steady_state: 1 # Steady state flag: 0 -> time-dependent 1-> steady-state simulation 
    decoupling_ctrl: 0 # Flag that activates the decoupling depth 
    model_shear: "NoShear" # [Constant,SelfConsistent,NoShear] # Flag that activates shear heating
    dt: 0.015 # Initial guess for the timestep [SI = Myr]
    stokes_solver_type : "Direct" # Flag that controls if the solver is direct or iterative for the Stokes problem 
    energy_solver_type : "Direct" # Flag that controls if the solver is direct or iterative for the Energy problem
    pressure_dependency: 1 # Flag that activates the pressure-dependency of the material properties (e.g. conductivity, density, thermal expansion)
    iterative_solver_tol : 1e-9 # Relative tolerance of iterative solver
    CFL : 0.8 # Courant Criteria correction factor
    initial_guess : 0  # Initial guess flag
```
**Note**: 

- The iterative solver is still a work in progress; it must be revisited to make it usable. 

- **model_shear**: activate the shear heating boundary conditions. However, *SelfConsistent* and *Constant* require that *decoupling_ctrl* is 1
  - *NoShear*: The shear heating is not active
  - *SelfConsistent*: It requires that a dislocation creep law in the ShearHeating section is defined, together with a friction angle. 
  - *Constant*: It requires the definition of a minimum stress, through which the shear heating is computed. 
- CFL: it corrects the Courant criteria timestep. **remember** Crank-Nicolson is not unconditionally stable with non-linearities. 
- **initial_guess**: This flag is automatically deactivated in case of steady-state simulations. The flag controls if a linear steady-state simulation is run prior the time-dependent case.

### ShearHeating
```
    shear_heating_disl_phi: 5.0 # Friction angle [deg]
    shear_heating_disl_tau_min: 0.0 # Constant stress 
    shear_heating_disl_law: "Wet_Quartzite_2001_Dislocation_creep" # dislocation law for the shear heating
```
**Note**

- The dislocation creep laws available are: 
  - Wet_Plagioclase_Dislocation_creep :cite:p:`rybacki2004deformation`
  - Hirareth_Serpentinite_Dislocation_creep :cite:p:`hilairet2007high`
  - Wet_Quartzite_2001_Dislocation_creep :cite:p:`hirth2001evaluation`
  - Glaucophane_2025_Dislocation_creep :cite:p:`hufford2026blueschist`

### InputOutputControl
```
    test_name: "Output" # Name of the test
    path_test: "../Results" # Main folder
    ts_time : "step" # Flag that controls if the simulation is releasing an output as a function of a fixed amount of timesteps or as a function of a time interval
    ts_out : 20  # The amount of timesteps required to print an output 
    dt_out : 0.5 # Time interval required to print a timestep result. 
```
To run the numerical code, the user has to specify the test name, the path of the folder in which the output is stored. *StonedFEniCSx* automatically creates the parent folder, and the relative test folder. The code creates a **Cached_information** folder in which to store the mesh data, the thermal boundary condition data. The user can decide to always compute the mesh information and the thermal boundary conditions. Especially, whenever the user needs to systematically change the geometry, or the thermal parameters. 

### scaling
```
    length: 600e3 # Scaling of the length
    stress: 1.0e9 # Stress/Pressure scale
    eta: 1.0e21 # Viscosity scale 
    temp: 1333.0 # Temperature scale
```
All the derivative scalings are automatically computed after the configuration of the simulation. For example, time is computed using the stress scale and the viscosity scale. The dimensions must be given in **m**, **Pa**, **Pa s** and **deg C**. 

### thermal_boundary_condition
```
    temp_max: 1300.0 # Maximum temperature (mantle temperature) [SI = deg C]
    temp_top: 0.0 # Surface temperature [SI = deg C]
    constant: 1  # Flag that controls if the age remains constant during the time-dependent solution
    interval_val : [50.0,30] # Interval of values [SI = Myr]
    interval_time: [20,40] # Interval of times in which the change is occuring [SI = Myr]
    nz: 108 # Numerical parameter to compute the right and left side thermal boundary conditions
    end_time: 180.0 # the time in which the half-space cooling model is computed
    dt: 0.005 # the timestep [SI = Myr]
    slab_age: 50.0 # Age of the slab [SI = Myr]
    right_boundary : 'Continental'  # Oceanic  # Type of boundary condition
    right_age: 50.0  # Age of the right boundary condition
    recalculate : 1 # Option to compute on the fly the boundary -> useful if user wants to change thermal properties. 

```
The left and right boundary conditions are computed using a finite difference scheme with Crank-Nicolson. In the case of the left boundary condition, the code computes from 0 to *end_time* a half space cooling model, and then selects the appropriate thermal profile as a function of the *slab_age*. In case of the right boundary condition the user must choose whether or not the boundary represents a continental margin or oceanic plate. In the case of *Oceanic*, the code will compute the same 2D array of thermal profiles; otherwise, in the case of *Continental*, it computes an initial linear geotherm as a function of the geometrical input. Then, it will run a thermal diffusion to reach a quasi-steady state. This is particularly useful when radiogenic heating is active. 
The *recalculate* option forces the computation of the boundary condition at each realization. If it is not active, the code will read the small h5 database saved in the *Cached_information* folder. 

### kinematic_boundary_condition
```
    v_s : [5.0,0.0] # Initial vector for the velocity of the slab [SI=cm/yr]
    constant : 1  # Flag that signals that the velocity of subduction does not change over time
    interval_val : [5.0,1.0] # interval of velocities [SI=cm/yr]
    interval_time : [20,40] # interval of time when the velocity change occurs
```
**Note**: Both in the case of kinematic and thermal boundary condition, the variation of velocity and age over time is linear. Curently these feature have not been tested. 


### Material properties
```
  wedge_mantle: 
      name_diffusion : 'Constant' # Name of diffusion creep law
      name_dislocation : 'Constant' # Name of dislocation creep law
      name_alpha : 'Constant' # Name of the thermal expansivity law
      name_density : 'Constant' # Name of the density law
      name_capacity: 'Constant' # Name of the capacity law
      name_conductivity: 'Constant' # Name of the conductivity law
      eta : 1.0e20  # Viscosity of the phase [SI = Pas]
      rho0 : 3300.0  # Reference density of the phase
      k : 3.0 # Constant conductivity [SI = W/m/K]
      cp : 1250 # Constant Heat capacity [SI = J/kg/K]
      alpha0 : 3e-5 # Constant thermal expansivity [SI = 1/K]
      b_dif : null # Pre-exponential factor of the diffusion creep [SI = Pa^(-1)s^(-1)]
      b_dis : null #Pre-exponential factor of the dislocation creep [SI = Pa^(-n)s^(-1)]
      e_dif : null #Activation energy of the diffusion creep [SI = J/mol]
      e_dis : null #Activation energy of the dislocation creep [SI = J/mol]
      v_dis : null #Activation volume of the dislocation creep [SI = m^3/Pa]
      v_dif : null #Activation volume of the diffusion creep [SI = m^3/Pa]
      n : null # Stress exponent dislocation creep
      radiative_conductivity : 0  # radiative conductivity flag: 0 means that the radiative conductivity for the specific phase is not considered, 1: means that the radiative conductivity is considered for the specific case. 
      radiogenic_heat : 0.0  # radiogenic heat production [SI = W/m^3]

```
The snipet above is an example. The names of the phases stored in the Material_Properties are fixed and bounded to the specific problem. Many phases feature constant viscosity, and the only freedom of the user is deciding the rheology of the **wedge_mantle** and the thermal properties of the other phases. This is a design choice, as the configuration module can be adapted to deal with multiphase problems. 
The user can change the value of diffusion creep and dislocation creep rheologies. However, these value must be given in the unit of measure listed in the input file. Moreover, the user should remember that the pre-exponential factor of dislocation depends on *n*, the stress exponent, so, changing the stress-exponent should be done with extra-care for the dimensional consistency. 

**Diffusion creep laws available**: 

- Hirth_dry_Diffusion_creep: :cite:p:`hirth2003rheology`
- VK_Diffusion_creep: :cite:p:`van2008community`
- Hirth_wet_Diffusion_creep: :cite:p:`hirth2003rheology`
- Constant: it is a flag that tells the code to not use the diffusion creep rheologies in the calculation

**Dislocation creep laws available**:

- Hirth_dry_Dislocation_creep: :cite:p:`hirth2003rheology`
- VK_Dislocation_creep: :cite:p:`van2008community`
- Hirth_wet_Dislocation_creep: :cite:p:`hirth2003rheology`
- Constant: it is a flag that indicates to use the dislocation creep rheology

**Note** if both dislocation and diffusion creep are constant, the code automatically assumes that the model is linear. Thus, it will use  either the default viscosity or the viscosity in the phase.

**Conductivity laws available**: 

- Mantle_Richards_2018: :cite:p:`richards2020structure`
- Crust_Richards_2018: :cite:p:`richards2020structure`
- Constant: flag that tells the code to use the constant conductivity *k* of the phase. 

**Heat capacity laws available**: 

- Mantle_Bernard_1988_FO: :cite:p:`berman1988internally`
- Mantle_Bernard_1988_FA: :cite:p:`berman1988internally`
- Mantle_Bernard_Ar_199x_FO: :cite:p:`berman1996optimized`
- Mantle_Bernard_Ar_199x_FA: :cite:p:`berman1996optimized`
- Mantle_Bernard_1988_FO_FA: :cite:p:`berman1988internally`
- Mantle_Bernard_Ar_199x_FO_FA: :cite:p:`berman1996optimized`
- Oceanic_crust:  :cite:p:`richards2020structure`
- Constant: flag that tells the code to use the constant conductivity *cp* of the phase. 
**Note**: _FO, _FA, FO_FA are flags that indicate which mixture of olivine to use. FO means fosterite, FA means fayalite. FO_FA implies that a mixture of 0.9 FO and 0.1 FA is used for computing the heat capacity.
Thermal expansivity laws available: 

- Mantle: :cite:p:`richards2020structure`
- Oceanic_crust: :cite:p:`richards2020structure`
- Constant: flag that tells the code to use the constant conductivity *alpha* of the phase.

**Density laws available**: 

- PT : pressure and temperature is active (pressure if the pressure dependency is active)
- Constant: flag that tells the code to use only the reference density *rho0*

## geometry 
```
    x: [0.0, 660.0] # Coordinate of X
    y: [-600.0, 0.0] # Coordinate of Y
    van_keken: False # Activate the geometry of Van Keken benchmark
    sub_constant_flag : False # Tells the code to have a slab with a constant bending angle
    slab_tk: 130.0 # Slab Thickness 
    cr: 30.0 # Overriding Crust {if lc==0.0 -> only overriding upper crust}
    ocr: 7.0 # Oceanic crust of the subducting plate 
    lc: 0.3 # lower to upper continental crust ratio
    resolution_normal: 3.0 # resolution far from the singularity point
    resolution_refine: 1.5 # resolution around the slab surface and singularity point
    ns_depth: 50.0 # depth of the no-slip condition 
    lab_d: 100.0 # depth of lithosphere-astenosphere boundary (necessary to compute the continental geotherm)
    decoupling: 80.0 # depth of decoupling
    transition: 10.0 # transition zone thickness (between fully coupled and uncoupled wedge flow regime)
    wz_tk: 0.5 # thickness of the virtual shear zone for computing the shear heating 
    sub_lb: 300.0 # total cumulative length in which the bending angle is evolving (necessary for the CustomRibe)
    sub_dl: 10.0 # length of segment to compute the slab surface
    sub_theta0: 5.0 # initial bending angle 
    sub_theta_max: 45.0 # final bending angle 
    sub_trench: 0.0 # position of the trench
    sub_parabolic_a: 8e-4 # [km^-1] -> curvature of the parabolic slab (necessary for the CustomParabolic)
    slab_type: "CustomParabolic" # [CustomParabolic,CustomRibe,FromFile] 
    sub_path: "Not Defined" # Required for the real geometry of the subducting plate
```
**Note**: CustomParabolic is still under debugging, so, it must not be used. 



## Usage
In the following section, a small example script is provided (see also the tests folder, with the benchmark examples) : 

1) The user must import the following functions: parse_input and stoned_fenicsx (see the snipet below):
```
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx
```
2) The user must indicate the input.yml file to use as a base: 
```
    # Path 2 test
    path_test = os.path.dirname(os.path.realpath(__file__))
    # Path 2 imput fie
    path_input = f"{path_test}/input_tests.yaml"
    # Parse the input: 
    # The input file is required to run a simulation. You can modify  
    # it and parse the input and then call the function for running simulation. 
    # Alternatively, you can generate the input file using it as blue print for the 
    # common property of the simulation, and modify the produced object for personalising 
    # the ensemble of simulations. 
    inp,ph_input = parse_input(path_input)
```
parse_input produces two macro-classes inp (type::Input) and ph_input (type::PhInput). 

These two classes are made of several subclasses: 

```
class Input:
    """Data class containing all the input.
    The class stores all the information parsed from the input.yml file,
    and can be called to be modified in
    other script for configure ensemble of numerical experiments.

    """
    ctrl: NumericalControls = field(default_factory=NumericalControls) # Numerical Controls 
    ctrl_io: IOControls = field(default_factory = IOControls) # InputOutput Controls
    ctrl_tbc: CtrlTemperatureBC = field(default_factory = CtrlTemperatureBC) #thermal_boundary_condition
    ctrl_ky: CtrlKy = field(default_factory=CtrlKy) # kinematic_boundary_condition
    g_input: GeomInput = field(default_factory=GeomInput) # geometry
    sc: Scal = field(default_factory=Scal) # scaling
```
```
class PhInput:
    """Container of the phases"""

    shear_heating_disl_law: str = "WetQuartzite" # from ShearHeating
    shear_heating_disl_tau_min: float = 0.0 # from ShearHeating
    shear_heating_disl_phi: float = 0.0 # from ShearHeating 
    subducting_plate_mantle: Phase = field(init=False)
    oceanic_crust: Phase = field(init=False)
    wedge_mantle: Phase = field(init=False)
    overriding_mantle: Phase = field(init=False)
    overriding_upper_crust: Phase = field(init=False)
    overriding_lower_crust: Phase = field(init=False)
```
To modify the value of each input inside the python script, the user should refer to the input file, and this guide: 
```
    inp.g_input.cr = .0 
    inp.g_input.lc = .0
    inp.g_input.ocr = 6.0 
    inp.g_input.lit_mt = 50.
    inp.g_input.lab_d = 50.
    inp.g_input.decoupling = .0 
    inp.g_input.van_keken = True 
    # Control 
    inp.ctrl.decoupling_ctrl = 0 
    inp.ctrl.steady_state = 1 


```
3) After the input are modified accordingly the user must call stoned_fenicsx function:

**example without modifying the input file
```
    # Path 2 test
    path_test = os.path.dirname(os.path.realpath(__file__))
    # Path 2 imput fie
    path_input = f"{path_test}/input_tests.yaml"
    # Parse the input: 
    # The input file is required to run a simulation. You can modify  
    # it and parse the input and then call the function for running simulation. 
    # Alternatively, you can generate the input file using it as blue print for the 
    # common property of the simulation, and modify the produced object for personalising 
    # the ensemble of simulations. 
    inp,ph_input = parse_input(path_input)
    stoned_fenicsx(inp = inp, ph_in=ph_input)

```
4) After the simulation configuration stage, the code configure the numerical experiments, and proceed with the operation of producing the mesh, the boundary conditions and scale the input parameters accordingly. 

The following snipet represent the configuration script for running the shear_heating tests: 

```
# Import the required path for processing the simulation
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx
from pathlib import Path
import os 
import numpy as np 
from mpi4py import MPI
# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False
#-------------------------------------------------------------------------------
def perform_test(phi=5.0,test_name='phi_5'):
    # Path 2 test
    path_test = os.path.dirname(os.path.realpath(__file__))
    # Path 2 imput fie
    path_input = f"{path_test}/input_tests.yaml"
    # Parse the input: 
    # The input file is required to run a simulation. You can modify  
    # it and parse the input and then call the function for running simulation. 
    # Alternatively, you can generate the input file using it as blue print for the 
    # common property of the simulation, and modify the produced object for personalising 
    # the ensemble of simulations. 
    inp,ph_input = parse_input(path_input)
    # Geometric Input: [inp.g_input.attributes -> change]
    inp.g_input.sub_theta_max = 30
    inp.g_input.cr = .0 
    inp.g_input.lc = .0
    inp.g_input.ocr = 6.0 
    inp.g_input.lit_mt = 50.
    inp.g_input.lab_d = 50.
    inp.g_input.decoupling = 80.0 
    inp.g_input.van_keken = True 
    # Control 
    inp.ctrl.decoupling_ctrl = 0 
    inp.ctrl.steady_state = 1 
    # In this case, for testing the Van Keken benchmark, I opted to create a simple script
    # that has: option viscosity and thermal for testing several potential configuration. 
    
    
    alpha_nameC = 'Constant'
    alpha_nameM = 'Constant'
    density_nameC = 'Constant'
    density_nameM = 'Constant'
    capacity_nameM = 'Constant'
    capacity_nameC = 'Constant'
    conductivity_nameM = 'Constant'
    conductivity_nameC = 'Constant'
    rho0_M = 3300.0
    rho0_C = 3300.0
    radio_flag = 0 
    
    name_diffusion = 'VK_Diffusion_creep'
    name_dislocation = 'VK_Dislocation_creep'     
        
    inp.ctrl.model_shear = 'SelfConsistent'
    inp.ctrl.decoupling_ctrl = 1
    ph_input.shear_heating_disl_phi = phi 
    ph_input.shear_heating_disl_law = "Wet_Quartzite_2001_Dislocation_creep"
    # ph_input contains the compositional phase -> you can modify them. The problem 
    # of kinematic simulations does not give a lot of freedom, and indeed, the possibility 
    # to have different rheologies is a design choiche to allow extension of the code 
    # in the future. Would be easier to start a new branch with more complex dynamic with 
    # config module. 

    # Modify the phase with the new data: 
    ph_input.subducting_plate_mantle.rho0 = rho0_M
    ph_input.subducting_plate_mantle.name_capacity = capacity_nameM
    ph_input.subducting_plate_mantle.name_conductivity = conductivity_nameM
    ph_input.subducting_plate_mantle.name_alpha = alpha_nameM
    ph_input.subducting_plate_mantle.name_density = density_nameM
    ph_input.subducting_plate_mantle.radiative_conductivity = radio_flag


    ph_input.oceanic_crust.rho0 = rho0_C
    ph_input.oceanic_crust.name_capacity = capacity_nameC
    ph_input.oceanic_crust.name_conductivity = conductivity_nameC
    ph_input.oceanic_crust.name_alpha = alpha_nameC
    ph_input.oceanic_crust.name_density = density_nameC
    ph_input.oceanic_crust.radiative_conductivity = radio_flag

    ph_input.wedge_mantle.name_diffusion = name_diffusion
    ph_input.wedge_mantle.name_dislocation = name_dislocation
    ph_input.wedge_mantle.rho0 = rho0_M
    ph_input.wedge_mantle.name_capacity = capacity_nameM 
    ph_input.wedge_mantle.name_conductivity = conductivity_nameM
    ph_input.wedge_mantle.name_alpha = alpha_nameM
    ph_input.wedge_mantle.name_density = density_nameM
    ph_input.wedge_mantle.radiative_conductivity = radio_flag

    ph_input.overriding_mantle.rho0 = rho0_M 
    ph_input.overriding_mantle.name_capacity = capacity_nameM
    ph_input.overriding_mantle.name_conductivity = conductivity_nameM
    ph_input.overriding_mantle.name_alpha = alpha_nameM
    ph_input.overriding_mantle.name_density = density_nameM
    ph_input.overriding_mantle.radiative_conductivity = radio_flag

    ph_input.overriding_upper_crust.rho0 = rho0_C 
    ph_input.overriding_upper_crust.name_capacity = capacity_nameC
    ph_input.overriding_upper_crust.name_conductivity = conductivity_nameC
    ph_input.overriding_upper_crust.name_alpha = alpha_nameC
    ph_input.overriding_upper_crust.name_density = density_nameC
    ph_input.overriding_upper_crust.radiative_conductivity = radio_flag

    ph_input.overriding_lower_crust.rho0 = rho0_C 
    ph_input.overriding_lower_crust.name_capacity = capacity_nameC
    ph_input.overriding_lower_crust.name_conductivity = conductivity_nameC
    ph_input.overriding_lower_crust.name_alpha = alpha_nameC
    ph_input.overriding_lower_crust.name_density = density_nameC
    ph_input.overriding_lower_crust.radiative_conductivity = radio_flag


    # -> Important: where to save and the name of the test. You can fully automatise the creation of new
    # folder. 
    inp.ctrl_io.test_name = f'T_{test_name}'
    inp.ctrl_io.path_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'VanKeken')
    

    # Initialise the input
    # After the user change the required data, and update the input and phase input, he must 
    # call this function, and run the simulation - hopefully, without throwing errors. 
    stoned_fenicsx(inp = inp, ph_in=ph_input)


def test_phi(phi=0.0,test_name='_phi5'):
    # Test Van Keken 
    
    perform_test(phi=phi,test_name=test_name) # IsoViscous

    # Remove folder after completing the test
    if not DEBUG:
        os.remove(f'{os.path.dirname(os.path.realpath(__file__))}/VanKeken')


#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    #test_phi(phi=3.0, test_name='_phi3')
    #test_phi(phi=5.0,test_name='_phi5')
    #test_phi(phi=10.0,test_name='_phi10')
    test_phi(phi=15.0,test_name='_phi15')
#---------------------------------------------------------------------------------

```

**Note**: In case the user wants to use an oceanic plate as overriding plate, the user should use the crustal unit **overriding_upper_crust** to create an oceanic-like crust and set to 0.0 **lc** in the geometry input (or in inp.g_input.lc=0)


## Status

Actively developed on branch `Remove_redundancy` (cleanup following the `Harmonise_zen_branch` refactor to a class-based solver API). Steady-state thermal + Stokes + lithostatic pressure coupling, including shear heating, is validated against the Van Keken et al. benchmark suite. The time-dependent solve path is functional but less exercised by the current test suite than the steady-state path.

## License

Not yet specified.
