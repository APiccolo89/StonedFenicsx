# Config Module — Suggested Docstrings

Docstrings are written in Google style to match the existing codebase convention.
Copy each block directly into the corresponding class or function.

---

## `numerical_control.py`

### `NumericalControls`
```python
"""Solver-level numerical controls for the FEniCSx simulation.

Stores flags and tolerances that govern the iterative solvers,
time-stepping strategy, and physical model switches. This class
contains no domain physics — boundary condition parameters and
material values live in their own dedicated classes.

Instance name convention: ``ctrl``

Attributes:
    it_max (int): Maximum outer Picard iterations per timestep.
    it_inner_max (int): Maximum inner Picard iterations (decoupled mode).
    tol (float): Outer loop convergence tolerance.
    tol_innerpic (float): Inner Picard loop convergence tolerance.
    relax (float): Relaxation factor applied to solution updates (0 < relax <= 1).
    g (float): Gravitational acceleration [m/s²].
    time_max (float): Total simulation duration [Myr].
    dt (float): Timestep size [Myr].
    steady_state (int): If 1, run in steady-state mode; time loop is skipped.
    decoupling_ctrl (int): If 1, Stokes and energy equations are solved
        sequentially (decoupled); if 0, they are solved simultaneously.
    model_shear (int): Shear heating model flag.
        0 = no shear heating, 1 = self-consistent shear heating.
    adiabatic_heating (int): Adiabatic heating flag (to be removed).
    stokes_solver_type (int): Stokes solver backend.
        1 = direct (MUMPS), 0 = iterative (GMRES/AMG).
    energy_solver_type (int): Energy solver backend.
        1 = direct, 0 = iterative.
    iterative_solver_tol (float): Tolerance for iterative linear solvers.
    eta_max (float): Maximum viscosity cap [Pa·s].
    pressure_dependency (int): If 1, material properties depend on pressure
        in addition to temperature.
"""
```

### `IOControls`
```python
"""Input/output path configuration.

Stores the names and filesystem paths required to write simulation
results and cached intermediate data. Call ``generate_io()`` during
configuration to create the directory tree before any output is written.

Instance name convention: ``ctrl_io``

Attributes:
    test_name (str): Name of this specific simulation run.
        Used as the leaf directory name under ``path_save``.
    path_save (str): Root directory where all results are written.
    sname (str): Short label used in output file names.
    ts_out (int): Output frequency in timestep units.
    dt_out (float): Output frequency in simulation time units [Myr].
    path_test (Path): Full path to the run output directory
        (``path_save / test_name``). Set by ``generate_io()``.
    path_cached_information (Path): Path for storing cached mesh and
        boundary condition data across restarts.
"""
```

### `IOControls.generate_io`
```python
"""Create the output directory tree for this simulation run.

Builds ``path_test = path_save / test_name`` and
``path_cached_information = path_save / 'Cached_information'``,
creating any missing parent directories. Must be called during
``configure_simulation`` before any output functions are invoked.

Raises:
    OSError: If the filesystem does not allow directory creation
        at ``path_save``.
"""
```

### `CTRLBC`
```python
"""Base class for time-varying boundary condition controls.

Encapsulates the shared logic for boundary conditions whose value
can either be held constant or varied linearly over a time interval.
Both the thermal (``CtrlTemperatureBC``) and kinematic (``CtrlKy``)
boundary conditions inherit from this class.

The time variation is described by two 2-element arrays:
- ``interval_val``: [initial value, final value]
- ``interval_time``: [start time, end time] [Myr]

If ``constant = 1``, the value does not change and ``interval_val``/
``interval_time`` are ignored at runtime (though ``interval_val[0]``
must still be consistent with the subclass initial value field).

Attributes:
    constant (int): If 1, the boundary condition is constant in time.
        If 0, it varies linearly over ``interval_time``.
    interval_val (ndarray[float64]): Shape (2,). Initial and final
        values of the boundary condition quantity [units depend on subclass].
    interval_time (ndarray[float64]): Shape (2,). Start and end times
        of the linear variation [Myr].
"""
```

### `CTRLBC.check_time_variation`
```python
"""Validate the time-variation settings against the global solver mode.

Raises:
    ValueError: If ``constant = 0`` (time-varying BC) is requested in
        steady-state mode (``ctrl.steady_state = 1``), which is physically
        inconsistent.
    ValueError: If the time interval has zero length
        (``interval_time[0] == interval_time[1]``) in transient mode,
        which would cause division by zero in ``update_vel_age``.

Args:
    ctrl (NumericalControls): Global solver controls providing the
        ``steady_state`` flag.
"""
```

### `CTRLBC.update_vel_age`
```python
"""Return the boundary condition value at time ``t`` via linear interpolation.

Clamps the result to ``interval_val[1]`` once the end of the interval
is reached, so calling this beyond ``interval_time[1]`` is safe.

Usage pattern — update in place without branching::

    ctrl_ky.v_s[0] = ctrl_ky.update_vel_age(t)

Args:
    t (float): Current simulation time [Myr].

Returns:
    float: Interpolated (and clamped) boundary condition value.
"""
```

### `CtrlTemperatureBC`
```python
"""Controls for the 1-D thermal boundary condition solver.

Configures the half-space cooling model used to compute the temperature
profile of the subducting slab at the left boundary. Inherits the
time-variation interface from ``CTRLBC`` so the slab age can evolve
during a transient simulation.

Instance name convention: ``ctrl_tbc``

Note:
    ``slab_age`` must equal ``interval_val[0]``. If they differ at
    configuration time, ``update_thermal_bc`` logs a warning and
    corrects ``interval_val[0]`` automatically.

Attributes:
    temp_top (float): Surface temperature [°C].
    temp_max (float): Far-field mantle temperature [°C].
    nz (int): Number of vertical cells in the 1-D column.
    dt (float): Timestep for the 1-D thermal solver [Myr].
        Must be << 0.1 Myr; Crank–Nicolson is unconditionally stable
        for linear conductivity but not for non-linear models.
    slab_age (float): Initial age of the subducting slab [Myr].
        Convenience alias for ``interval_val[0]``.
    t_res (int): Number of time steps stored in the 2-D temperature array.
    recalculate (int): If 1, recompute the thermal field even if a cached
        version exists.
    k (float): Thermal conductivity [W/m/K].
    rho (float): Density [kg/m³].
    cp (float): Heat capacity [J/kg/K].
    end_time (float): Duration of the 1-D thermal pre-computation [Myr].
    right_boundary (str): Type of right-boundary thermal condition.
        ``'Continental'`` or ``'Oceanic'``.
    right_age (float): Age of the oceanic lithosphere at the right
        boundary [Myr]. Only used when ``right_boundary = 'Oceanic'``.
    self_consistent_flag (int): If 1, use self-consistent thermal
        properties during the 1-D solve.
    dz (float): Vertical cell size [m]. Set by ``update_thermal_bc``.
    z (ndarray): Vertical coordinate array [m]. Set by ``update_thermal_bc``.
    temperature_1d (ndarray): Current 1-D temperature profile [°C].
        Set by ``update_thermal_bc``.
    temperature_2d_field (ndarray): Full time × space temperature history,
        shape (nz, t_res). Set by ``update_thermal_bc``.
    t_res_vec (ndarray): Time vector for the 2-D field [Myr].
        Set by ``update_thermal_bc``.
"""
```

### `CtrlTemperatureBC.update_thermal_bc`
```python
"""Initialise the 1-D thermal solver arrays and validate the configuration.

Must be called once during ``configure_simulation``, after geometry
has been checked and before the thermal solver runs.

Performs:
- Timestep sanity check (``dt > 0.1 Myr`` raises; Crank–Nicolson breaks
  for non-linear conductivity at large steps).
- Time-variation consistency via ``check_time_variation``.
- Auto-correction of ``interval_val[0]`` if it differs from ``slab_age``
  (logs a warning).
- Allocation of ``dz``, ``z``, ``temperature_1d``,
  ``temperature_2d_field``, ``t_res_vec``.

Args:
    g_input (GeomInput): Geometry object providing ``slab_tk``
        (slab thickness [km]).
    ctrl (NumericalControls): Global solver controls.

Raises:
    ValueError: If ``dt > 0.1`` [Myr].
"""
```

### `CtrlKy`
```python
"""Controls for the kinematic boundary condition (slab velocity).

Specifies the convergence velocity of the subducting plate and its
optional time variation. Inherits the time-variation interface from
``CTRLBC`` so the velocity can ramp linearly over a given interval.

Instance name convention: ``ctrl_ky``

Note:
    ``v_s[0]`` must equal ``interval_val[0]``. If they differ at
    configuration time, ``check_kinematic_bc`` logs a warning and
    corrects ``interval_val[0]`` automatically.

Attributes:
    v_s (ndarray[float64]): Shape (2,). Initial and final convergence
        velocities [cm/yr]. ``v_s[0]`` is the starting velocity;
        ``v_s[1]`` is unused in constant mode.
"""
```

### `CtrlKy.check_kinematic_bc`
```python
"""Validate the kinematic boundary condition configuration.

Calls ``check_time_variation`` and checks that ``v_s[0]`` is consistent
with ``interval_val[0]``. If they differ, logs a warning and corrects
``interval_val[0]`` automatically so the time-interpolation in
``update_vel_age`` starts from the correct initial velocity.

Args:
    ctrl (NumericalControls): Global solver controls.
"""
```

### `SimulationControls`
```python
"""Bundle of all simulation control objects passed to the solver.

Assembled by ``configure_simulation`` after the user has finished
modifying the flat ``Input`` class. Acts as a single object that
solver routines can receive instead of four separate arguments,
avoiding long parameter lists in the solver API.

This class is never constructed directly by the user. The user
works with the flat ``Input`` class; ``SimulationControls`` is the
post-configuration, solver-facing representation.

Instance name convention: ``ctrl_sm``

Attributes:
    ctrl (NumericalControls): Solver flags and tolerances.
    ctrl_io (IOControls): I/O path configuration.
    ctrl_tbc (CtrlTemperatureBC): Thermal boundary condition controls
        (fully initialised, arrays allocated).
    ctrl_ky (CtrlKy): Kinematic boundary condition controls (validated).
"""
```

---

## `input_parser.py`

### `Phase`
```python
"""Container for the rheological and thermal parameters of a single material phase.

One ``Phase`` instance represents one material zone in the model
(e.g., subducting plate mantle, oceanic crust, wedge mantle). All
``Phase`` objects are collected inside ``PhInput`` and later used
to build the ``PhaseDataBase`` for runtime material property evaluation.

Optional rheological parameters (activation energies, volumes,
pre-exponential factors) default to ``None`` when the corresponding
flow law is ``'Constant'``.

Attributes:
    name_phase (str): Human-readable label for this phase.
    id_ph (int): Integer phase identifier used by the solver to
        index into the material property database.
    name_diffusion (str): Diffusion creep flow law name.
    name_dislocation (str): Dislocation creep flow law name.
    e_dif (float | None): Activation energy for diffusion creep [J/mol].
    v_dif (float | None): Activation volume for diffusion creep [m³/mol].
    b_dif (float | None): Pre-exponential factor for diffusion creep [Pa⁻¹s⁻¹].
    n (float | None): Stress exponent for dislocation creep.
    e_dis (float | None): Activation energy for dislocation creep [J/mol].
    v_dis (float | None): Activation volume for dislocation creep [m³/mol].
    b_dis (float | None): Pre-exponential factor for dislocation creep [Pa⁻ⁿs⁻¹].
    eta (float): Constant viscosity [Pa·s]. Used when both creep laws
        are ``'Constant'``.
    cp (float): Constant heat capacity [J/kg/K].
    k (float): Constant thermal conductivity [W/m/K].
    rho0 (float): Reference density [kg/m³].
    name_capacity (str): Heat capacity law name.
    name_conductivity (str): Thermal conductivity law name.
    name_alpha (str): Thermal expansivity law name.
    name_density (str): Density law name.
    alpha0 (float): Constant thermal expansivity [K⁻¹].
    radiogenic_heat (float): Radiogenic heat production [W/m³].
    radiative_conductivity (float): Radiative conductivity flag/scale.
        0.0 = off, 1.0 = on.
"""
```

### `PhInput`
```python
"""Temporary container for all material phase definitions.

Holds one ``Phase`` instance per material zone. Populated by
``filling_the_phase_data_base`` from the ``Material_properties``
block of ``input.yaml``. Passed to ``configure_simulation`` where
it is consumed to build the runtime ``PhaseDataBase``.

Like ``Input``, this object is temporary: it exists only between
``parse_input`` and ``configure_simulation`` and should be
discarded afterwards.

Attributes:
    shear_heating_disl_law (str): Dislocation creep law used for
        shear heating computation in the weak zone.
    shear_heating_disl_ch (float): Cohesion for shear heating [Pa].
    shear_heating_disl_phi (float): Friction angle for shear heating [°].
    subducting_plate_mantle (Phase): Lithospheric mantle of the slab.
    oceanic_crust (Phase): Oceanic crustal layer of the slab.
    wedge_mantle (Phase): Mantle wedge above the slab.
    overriding_mantle (Phase): Lithospheric mantle of the overriding plate.
    overriding_upper_crust (Phase): Upper crust of the overriding plate.
    overriding_lower_crust (Phase): Lower crust of the overriding plate.
"""
```

### `Input`
```python
"""Flat, user-facing staging container for simulation configuration.

Produced by ``parse_input`` and intended to be held briefly by the
user script, optionally modified for ensemble runs, then passed to
``configure_simulation``. After configuration this object should be
discarded — the solver never receives it directly.

The flat layout (one attribute per control object) is intentional:
it makes programmatic modification concise::

    inp, ph = parse_input("input.yaml")
    inp.ctrl_ky.v_s = np.array([7.0, 0.0])   # modify for this run
    inp.ctrl_tbc.slab_age = 60.0
    configure_simulation(ph, inp)

Attributes:
    ctrl (NumericalControls): Solver flags and tolerances.
    ctrl_io (IOControls): I/O path configuration.
    ctrl_tbc (CtrlTemperatureBC): Thermal boundary condition controls.
    ctrl_ky (CtrlKy): Kinematic boundary condition controls.
    g_input (GeomInput): Geometry definition.
    sc (Scal): Scaling parameters.
"""
```

### `correct_input`
```python
"""Convert string-valued YAML entries to their integer flag equivalents.

The YAML file uses human-readable strings for certain options
(e.g., ``model_shear: "NoShear"``) while the solver expects
integer flags. This function performs the lookup via module-level
dictionaries and leaves all other values unchanged.

Recognised keys:
- ``model_shear``: ``{"NoShear": 0, "SelfConsistent": 1}``
- ``stokes_solver_type``: ``{"Direct": 1, "Iterative": 0}``
- ``energy_solver_type``: ``{"Direct": 1, "Iterative": 0}``

Args:
    k (str): YAML key.
    v (str): String value from the YAML block.

Returns:
    int | float | str: Converted value, or ``v`` unchanged if ``k``
        is not a recognised string option.
"""
```

### `update_ip_file`
```python
"""Populate a dataclass instance from a YAML dictionary block.

Iterates over the key-value pairs in ``block``, validates that each
key corresponds to a declared field on ``obj``, converts the value to
the correct Python type via ``cast_type``, and sets the attribute.

This function is the single point of contact between the raw YAML
data and the typed dataclass fields. All config objects (``ctrl``,
``ctrl_io``, ``ctrl_tbc``, ``ctrl_ky``, ``g_input``, ``sc``) are
populated through this function.

Args:
    obj (object): Target dataclass instance to populate.
    block (dict): Dictionary of key-value pairs from ``yaml.safe_load``.

Returns:
    object: The same ``obj`` with fields updated in place.

Raises:
    ValueError: If ``block`` contains a key that is not a declared
        field on ``obj.__class__``, with a message indicating the
        unknown key and the class name.
"""
```

### `parse_input`
```python
"""Read and parse a YAML input file into typed configuration objects.

Loads the YAML file, distributes each section to the appropriate
dataclass via ``update_ip_file``, and returns a flat ``Input``
object together with a ``PhInput`` holding the material definitions.

The returned objects are intended for optional programmatic
modification before being passed to ``configure_simulation``.
They should be discarded after configuration is complete.

Args:
    path (str | Path): Path to the YAML input file.

Returns:
    tuple[Input, PhInput]:
        - ``Input``: flat staging container with all simulation controls.
        - ``PhInput``: material phase definitions.

Raises:
    KeyError: If a required top-level section is missing from the YAML
        (e.g., ``NumericalControls``, ``geometry``).
    ValueError: If a YAML key does not match a field in the target class
        (propagated from ``update_ip_file``).
"""
```

### `cast_type`
```python
"""Coerce a value from a YAML parse result to the annotated Python type.

Handles the common mismatches between what ``yaml.safe_load`` produces
(plain Python ints, floats, lists, bools) and what the typed dataclass
fields expect (``np.float64``, ``np.ndarray``, ``bool``, etc.).

Supported target types:
- ``np.float64``: wraps value with ``np.float64()``.
- ``bool``: accepts ``True``/``False``, ``"true"``/``"false"``,
  ``"yes"``/``"no"``, and integer 0/1.
- ``list[T]``: recursively casts each element to subtype ``T``.
- ``tuple[T]``: recursively casts each element to subtype ``T``.
- ``np.ndarray``: converts via ``np.asarray()``.
- Any other type ``T``: calls ``T(v)``.

Args:
    v: Raw value from the YAML parser.
    tp: Target type annotation from ``get_type_hints``.

Returns:
    Value coerced to the target type.
"""
```

### `filling_the_phase_data_base`
```python
"""Populate a ``PhInput`` container from the material properties YAML block.

For each named material zone in ``materialproperties``, constructs a
``Phase`` instance, sets its fields from the YAML dictionary, assigns
the phase id from the zone-to-id mapping, and attaches it to the
corresponding attribute of ``phase_input``.

``None`` values in the YAML (null in YAML) are kept as ``None`` for
optional rheological parameters, except for ``radiogenic_heat`` and
``radiative_conductivity`` which default to ``0.0``.

Args:
    materialproperties (dict): Nested dictionary from
        ``input.yaml["Input"]["Material_properties"]``.
    shheating (dict): Shear heating parameters from
        ``input.yaml["Input"]["Shear_Heating"]``.
    phase_input (PhInput): Empty ``PhInput`` instance to populate.

Returns:
    PhInput: Fully populated phase container.
"""
```

---

## `geometry.py`

### `Domain`
```python
"""Mesh domain container for the global mesh or a named subdomain.

Stores the DOLFINx mesh object together with all associated metadata
needed to transfer fields between the global mesh and its submeshes:
cell/node parent maps, boundary facet tags, cell material tags, and
the phase function space.

Attributes:
    hierarchy (str): ``"parent"`` for the global mesh,
        ``"child"`` for a submesh.
    mesh (dolfinx.mesh.Mesh): The DOLFINx mesh object.
    cell_par (ndarray[int32]): Map from submesh cell indices to
        parent mesh cell indices.
    node_par (ndarray[int32]): Map from submesh node indices to
        parent mesh node indices.
    facets (dolfinx.mesh.MeshTags): Tagged boundary facets
        (trench, free surface, inflow, etc.).
    tagcells (dolfinx.mesh.MeshTags): Tagged cells identifying
        material regions.
    bc_dict (dict): Mapping from boundary condition names to integer tags.
    solph (dolfinx.fem.FunctionSpace): Function space for phase
        indicator fields.
    phase (dolfinx.fem.Function): Material phase indicator defined
        on this domain.
"""
```

### `GeomInput`
```python
"""Geometric parameters defining the subduction zone model.

All lengths in [km] and angles in [degrees] as read from the YAML.
``check_class_consistency`` converts angles to radians in place before
the mesh is generated.

The subducting slab geometry can be either:
- ``slab_type = "Custom"``: parametric geometry controlled by
  ``sub_theta0``, ``sub_theta_max``, ``sub_lb``, etc.
- ``slab_type = "Real"``: geometry read from an external file at
  ``sub_path``.

Attributes:
    x (ndarray): Domain x-extent [x_min, x_max] [km].
    y (ndarray): Domain y-extent [y_min, y_max] [km]. y_min must be < 0.
    slab_tk (float): Slab thickness [km].
    cr (float): Overriding crust thickness [km].
    ocr (float): Oceanic crust thickness [km].
    lit_mt (float): Lithospheric mantle thickness [km].
    lc (float): Lower crust fraction of ``cr`` [dimensionless, 0–1].
        If 0.0, only the upper crust is present.
    ns_depth (float): Total lithosphere thickness (node-search depth) [km].
    decoupling (float): Depth at which the slab and wedge decouple [km].
    resolution_normal (float): Background mesh resolution [km].
    resolution_refine (float): Refined mesh resolution near the slab [km].
    theta_out_slab (float): Outer slab dip angle [degrees].
    theta_in_slab (float): Inner slab dip angle [degrees].
    transition (float): Width of the coupled-to-decoupled transition [km].
    lab_d (float): Lithosphere–asthenosphere boundary depth [km].
    slab_type (str): ``"Custom"`` or ``"Real"``.
    sub_path (str): Path to the slab geometry file (used if
        ``slab_type = "Real"``).
    sub_lb (float): Along-dip length of the slab [km].
    sub_constant_flag (bool): If True, the slab dip angle is constant.
        Required for Van Keken benchmark runs.
    sub_theta0 (float): Initial slab dip angle [degrees → radians after
        ``check_class_consistency``].
    sub_theta_max (float): Maximum slab dip angle [degrees → radians after
        ``check_class_consistency``].
    sub_trench (float): Trench position [km].
    sub_dl (float): Along-dip discretisation length [km].
    wz_tk (float): Weak zone thickness [km].
    van_keken (bool): If True, enforce Van Keken benchmark geometry
        (requires ``sub_constant_flag = True``).
"""
```

### `GeomInput.check_class_consistency`
```python
"""Validate geometry parameters and convert angles to radians.

Called once inside ``configure_simulation`` before mesh generation.
Converts ``sub_theta0`` and ``sub_theta_max`` from degrees to radians
in place so that all downstream geometric computations use SI units.

Raises:
    ValueError: If ``lc`` is outside [0, 1].
    ValueError: If ``slab_type`` is not ``"Custom"`` or ``"Real"``.
    ValueError: If ``van_keken = True`` but ``sub_constant_flag = False``
        (benchmark requires a constant dip angle).
    ValueError: If ``slab_type = "Real"`` but ``sub_path`` is
        ``"Not_Defined"`` or ``None``.
"""
```

### `Mesh`
```python
"""Central container for the FEniCSx mesh and finite element definitions.

Holds the global domain, the three subdomains (subducting plate,
mantle wedge, overriding crust), MPI communicator information, and
the finite element families used for pressure, temperature, and velocity.

Created by ``create_mesh`` during ``configure_simulation``.

Attributes:
    g_input (GeomInput): Geometry parameters used to build the mesh.
    global_domain (Domain): Full computational domain.
    subduction_plate_domain (Domain): Submesh of the subducting plate.
    wedge_domain (Domain): Submesh of the mantle wedge.
    crust_domain (Domain): Submesh of the overriding crust.
    comm (MPI.Intracomm): MPI communicator.
    rank (int): MPI rank of this process.
    element_p (object): Finite element for the pressure field (P1).
    element_pt (object): Finite element for the temperature field (P2).
    element_v (object): Finite element for the velocity field (P2 vector).
"""
```

---

## `scal.py`

### `Scal`
```python
"""Non-dimensionalisation scaling factors for the simulation.

Given four independent characteristic scales (length, temperature,
viscosity, stress), ``compute_the_derivative_scal`` derives all
remaining SI unit scales. Every dimensional quantity in the solver
is divided by its corresponding scale before the equations are solved.

Instance name convention: ``sc``

Attributes:
    length (float): Characteristic length scale [m].
    temp (float): Characteristic temperature scale [K].
    eta (float): Characteristic viscosity scale [Pa·s].
    stress (float): Characteristic stress scale [Pa].
    time (float): Derived time scale [s] = eta / stress.
    mass (float): Derived mass scale [kg].
    ac (float): Derived acceleration scale [m/s²].
    rho (float): Derived density scale [kg/m³].
    force (float): Derived force scale [N].
    energy (float): Derived energy scale [J].
    watt (float): Derived power scale [W].
    strain_rate (float): Derived strain rate scale [1/s].
    k (float): Derived thermal conductivity scale [W/m/K].
    cp (float): Derived heat capacity scale [J/kg/K].
    scale_vel (float): Fixed factor converting cm/yr to m/s.
    scale_myr2sec (float): Fixed factor converting Myr to seconds.
"""
```

### `Scal.compute_the_derivative_scal`
```python
"""Derive all secondary scaling factors from the four primary scales.

Must be called once after the four primary scales (``length``,
``temp``, ``eta``, ``stress``) have been set from the YAML, and
before any scaling function is applied.

All derived scales follow directly from dimensional analysis:
``time = eta / stress``, ``rho = stress · time² / length²``, etc.

Returns:
    Scal: ``self``, allowing chained calls.
"""
```

### `_scaling_material_properties`
```python
"""Non-dimensionalise all fields in a ``PhaseDataBase`` in place.

Divides every dimensional quantity stored in ``pdb`` (viscosities,
conductivities, heat capacities, flow law pre-exponential factors,
densities, radiogenic heat) by the appropriate scale derived from ``sc``.

Must be called once during ``configure_simulation``, after
``compute_the_derivative_scal`` and before the solver starts.

Args:
    pdb (PhaseDataBase): The material property database to scale.
    sc (Scal): Fully initialised scaling object.

Returns:
    PhaseDataBase: The same ``pdb`` with all fields non-dimensionalised.
"""
```

### `scale_parameters`
```python
"""Non-dimensionalise the thermal boundary condition controls in place.

Converts all time quantities (``end_time``, ``dt``, age arrays) from
Myr to the dimensionless time unit, and spatial quantities (``dz``)
and material properties (``k``, ``cp``, ``rho``) from SI to
dimensionless units.

Must be called after ``update_thermal_bc`` has allocated the arrays
and after ``compute_the_derivative_scal`` has populated ``scal``.

Args:
    ctrl_tbc (CtrlTemperatureBC): Thermal BC controls to scale.
    scal (Scal): Fully initialised scaling object.

Returns:
    CtrlTemperatureBC: The same ``ctrl_tbc`` with all fields scaled.
"""
```

### `scaling_control_parameters`
```python
"""Non-dimensionalise the main numerical control parameters in place.

Scales gravity, time-stepping parameters, and solver tolerances that
depend on physical units. Called once during ``configure_simulation``
before the solver is initialised.

Args:
    ctrl (NumericalControls): Solver controls to scale.
    scal (Scal): Fully initialised scaling object.

Returns:
    NumericalControls: The same ``ctrl`` with dimensional fields scaled.
"""
```

### `scaling_mesh`
```python
"""Scale mesh node coordinates to dimensionless units in place.

Divides all nodal coordinates by ``sc.length``. Must be called after
the mesh is created and before any FEniCSx assembly begins.

Args:
    mesh (Mesh): The FEniCSx mesh to scale.
    sc (Scal): Fully initialised scaling object.

Returns:
    Mesh: The same ``mesh`` with scaled coordinates.
"""
```

### `dimensionless_ginput`
```python
"""Scale all geometric input lengths to dimensionless units in place.

Converts every length-valued field in ``g_input`` (domain extents,
thicknesses, resolutions, depths) from km to the dimensionless length
unit. Must be called after ``check_class_consistency`` and before
mesh generation.

Args:
    g_input (GeomInput): Geometry definition to scale.
    sc (Scal): Fully initialised scaling object.

Returns:
    GeomInput: The same ``g_input`` with all lengths scaled.
"""
```

### `scal_time_class`
```python
"""Scale the time-variation arrays of a ``CtrlKy`` to dimensionless units.

Converts velocity arrays from cm/yr to the dimensionless velocity unit
and time arrays from Myr to the dimensionless time unit.

Args:
    ctrl_ky (CtrlKy): Kinematic BC controls to scale.
    sc (Scal): Fully initialised scaling object.

Returns:
    CtrlKy: The same ``ctrl_ky`` with all time/velocity arrays scaled.
"""
```

---

## `simulation_config.py`

### `configure_simulation`
```python
"""Translate a parsed ``Input`` into fully configured runtime objects.

This is the single boundary between the user-facing staging layer
(``Input`` / ``PhInput``) and the solver. It performs, in order:

1. Derive secondary scaling factors (``sc.compute_the_derivative_scal``).
2. Validate geometry (``g_input.check_class_consistency``).
3. Initialise the thermal BC arrays (``ctrl_tbc.update_thermal_bc``).
4. Validate the kinematic BC (``ctrl_ky.check_kinematic_bc``).
5. Create the output directory tree (``ctrl_io.generate_io``).
6. Generate the mesh (``create_mesh``).
7. Build the runtime material database (``generate_phase_database``).
8. Non-dimensionalise all controls, geometry, and material properties.
9. Assemble and return ``SimulationControls``.

The ``Input`` and ``PhInput`` objects should be discarded after this
function returns.

Args:
    ph_in (PhInput): Pre-processed material phase definitions.
        May have been modified by the user script before this call.
    inp (Input): Flat staging container. May have been modified by
        the user script before this call.

Returns:
    tuple[SimulationControls, PhaseDataBase, Mesh]:
        - ``SimulationControls``: fully configured solver control bundle.
        - ``PhaseDataBase``: non-dimensionalised material property database.
        - ``Mesh``: FEniCSx mesh with subdomains and element definitions.
"""
```
