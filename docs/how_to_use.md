# How to use 

The code is organized to require always the definition of an *input_file.yml*. The blue-print of the input file can be found in the main folder of the package. 

The unit of measure in the input code are always: **Myr**, **km**, **cm/yr**. The conversion in SI units and the relative scaling is performed internally during the configuration step of the simulation. 

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
```
