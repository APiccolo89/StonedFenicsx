# How to use 

The code is organized to require always the definition of an *input_file.yml*. The blue-print of the input file can be found in the main folder of the package. 

The unit of measure in the input code are always: **Myr**, **km**, **cm/yr**, **deg** and **degC** for time, length, velocity, angles and temperatures respectevely; the only exception is for the *scaling* options, in this case, the unit of measure are **m**, **Pa**, **Pa s** and **deg C** The conversion in SI units and the relative scaling is performed internally during the configuration step of the simulation. 

The input file is divided into 8 subsections: 

- **NumericalControls**: the parameters that control the behaviour of the simulation.
- **ShearHeating**: the set of parameters that controls the internal boundary heat source (the parametrised shear heating along the seismogenetic zone of the subduction plate.)
- **InputOutputControl**: the set of parameters that controls where to save the output files, and under which condition to release a timestep. 
- **scaling**: the set of SI units used to scale the parameters of the simulation. 
- **thermal_boundary_condition**: the thermal boundary condition configuration parameters
- **kinematic_boundary_condition**: the kinematic boundary condition configuration parameters. 
- **Material_properties**: the material property input parameters
- **geometry**: the geometrical configuration parameters. 

### Numerical Controls 
```
    it_max: 30 # Maximum number of iteration 
    tol: 5e-5 # Tollerance of the problem
    relax: 0.9 # Relax factor to update the solution
    g: 9.81 # Module of the gravitational acceleration 
    eta_max: 1.0e26 # Maximum viscosity of the system 
    time_max: 2.0  # Maximum time of the simulation [SI = Myr]
    steady_state: 1 # Steady state flag: 0 -> timedependent 1-> steadystate simulation 
    decoupling_ctrl: 0 # Flag that activate the decoupling depth 
    model_shear: "NoShear" # [Constant,SelfConsistent,NoShear] # Flag that activate shear heating
    dt: 0.015 # Initial guess for the timestep [SI = Myr]
    stokes_solver_type : "Direct" # Flag that controls if the solver is direct or iterative for the stokes problem 
    energy_solver_type : "Direct" # Flag that controls if the solver is direct or iterative for the energy problem
    pressure_dependency: 1 # Flag that activate the pressure-dependency of the material properties (e.g. conductivity, density, thermal expansion)
    iterative_solver_tol : 1e-9 # Relative tollerance iterative solver
    CFL : 0.8 # Courant Criteria correction factor
    initial_guess : 0  # Initial guess flag
```
**Note**: 

- The iterative solver is still a work in progress, it must be revisited to make it usable. 

- **model_shear**: activate the shear heating boundary conditions. However, *SelfConsistent* and *Constant* require that *deocoupling_ctrl* is 1
  - *NoShear*: The shear heating is not active
  - *SelfConsistent*: It requires that a dislocation creep law in the ShearHeating section is defined, together with a friction angle. 
  - *Constant*: It requires the definition of a minimum stress, through which the shear heating is computed. 
- CFL: it corrects the Courant criteria timestep. **remember** Crank-Nicolson is not unconditionally stable with non-linearties. 
- **initial_guess**: This flag is automatically deactivate in case of steady-state simulations. The flag run a steady-state solution with all the material property set to be linear and with default value. 

### ShearHeating
```
    shear_heating_disl_phi: 5.0 # Friction angle [deg]
    shear_heating_disl_tau_min: 0.0 # Constant stress 
    shear_heating_disl_law: "Wet_Quartzite_2001_Dislocation_creep" # dislocation law for the shear heating
```
**Note**

- The dislocation creep law available are: 
  - Wet_Plagioclase_Dislocation_creep :cite:p:`rybacki2004deformation`
  - Hirareth_Serpentinite_Dislocation_creep :cite:p:`hilairet2007high`
  - Wet_Quartzite_2001_Dislocation_creep :cite:p:`hirth2001evaluation`
  - Glaucophane_2025_Dislocation_creep :cite:p:`hufford2026blueschist`

### InputOutputControl
```
    test_name: "Output" # Name of the test
    path_test: "../Results" # Main folder
    ts_time : "step" # Flag that controls if the simulation is releasing an output as a function of a fixed amount of timestep or as a function of a time interval
    ts_out : 20  # The amount of timestep required to print an output 
    dt_out : 0.5 # Time interval required to print a timestep result. 
```
To run the numerical code, the user has to specify the test name, the path of the folder in which the output are stored. *StonedFEniCSx* automatically creates the parent folder, and the relative test folder. Moreover, it creates a **Cached_information** folder where to store the mesh data, the thermal boundary condition data. The user can decide to compute always the mesh information and the thermal boundary conditions. Especially, whenever the user needs to change systematically the geometry, or the thermal parameters. 

### Scal
```
    length: 600e3 # Scaling of the length
    stress: 1.0e9 # Stress/Pressure scale
    eta: 1.0e21 # Viscosity scale 
    temp: 1333.0 # Temperature scale
```
All the derivative scaling are automatically computed after the configuration of the simulation. For example, time is computed using the stress scale and viscosity scale. The dimension must be given in **m**, **Pa**, **Pa s** and **deg C**. 

### thermal_boundary_condition
```
    temp_max: 1300.0 # Maximum temperature (mantle temperature) [SI = deg C]
    temp_top: 0.0 # Surface temperature [SI = deg C]
    constant: 1  # Flag that controls if the age remains constant during the time-dependent solution
    interval_val : [50.0,30] # Interval of value [SI = Myr]
    interval_time: [20,40] # Interval of time in which the change is occuring [SI = Myr]
    nz: 108 # Numerical parameter to compute the right and left side thermal boundary condition
    end_time: 180.0 # the time in which the half-space cooling model is compute
    dt: 0.005 # the timestep [SI = Myr]
    slab_age: 50.0 # Age of the slab [SI = Myr]
    right_boundary : 'Continental'  # Oceanic  # Type of boundary condition
    right_age: 50.0  # Age of the right boundary condition
    recalculate : 1 # Option to compute on the fly the boundary -> useful if user wants to change thermal properties. 

```
The left and right boundary condition are computed using a finite difference scheme with Crank-Nicolson. In the case of the left boundary condition, the code compute from 0 to *end_time* a half space cooling model, and then select the aproriate thermal profile as a function of the *slab_age*. In case of the right boundary condition the user must choose whether or not the boundary represents a continent margin or oceanic plate. In the case of *Oceanic*, the code will compute the same 2D array of thermal profiles; otherwise, in the case of *Continental* it computes an initial linear geotherm as a function of the geometrical input. Then, it will run a thermal diffusion to reach a quasi-steady state. This is particularly useful in the case radiogenic heating is active. 
Recalculate option forces the computation of the boundary condition each realization. If it is not active, the code will read the small h5 database saved in the *Cached_information* folder. 

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
      b_dis : null #Pre-exponential factor of the diffusion creep [SI = Pa^(-n)s^(-1)]
      e_dif : null #Activation energy of the diffusion creep [SI = J/mol]
      e_dis : null #Activation energy of the dislocation creep [SI = J/mol]
      v_dis : null #Activation volume of the dislocation creep [SI = m^3/Pa]
      v_dif : null #Activation volume of the diffusion creep [SI = m^3/Pa]
      n : null # Stress exponent dislocation creep
      radiative_conductivity : 0  # radiative conductivity flag
      radiogenic_heat : 0.0  # radiogenic heat production [SI = W/m^3]

```