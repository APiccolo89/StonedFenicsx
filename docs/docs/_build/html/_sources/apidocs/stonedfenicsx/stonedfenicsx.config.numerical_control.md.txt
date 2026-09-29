# {py:mod}`stonedfenicsx.config.numerical_control`

```{py:module} stonedfenicsx.config.numerical_control
```

```{autodoc2-docstring} stonedfenicsx.config.numerical_control
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NumericalControls <stonedfenicsx.config.numerical_control.NumericalControls>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls
    :summary:
    ```
* - {py:obj}`IOControls <stonedfenicsx.config.numerical_control.IOControls>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls
    :summary:
    ```
* - {py:obj}`CTRLBC <stonedfenicsx.config.numerical_control.CTRLBC>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC
    :summary:
    ```
* - {py:obj}`CtrlTemperatureBC <stonedfenicsx.config.numerical_control.CtrlTemperatureBC>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC
    :summary:
    ```
* - {py:obj}`CtrlKy <stonedfenicsx.config.numerical_control.CtrlKy>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlKy
    :summary:
    ```
* - {py:obj}`SimulationControls <stonedfenicsx.config.numerical_control.SimulationControls>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`dict_shear_modes <stonedfenicsx.config.numerical_control.dict_shear_modes>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_shear_modes
    :summary:
    ```
* - {py:obj}`dict_solver_type <stonedfenicsx.config.numerical_control.dict_solver_type>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_solver_type
    :summary:
    ```
* - {py:obj}`dict_output_type <stonedfenicsx.config.numerical_control.dict_output_type>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_output_type
    :summary:
    ```
* - {py:obj}`dict_initial_guess <stonedfenicsx.config.numerical_control.dict_initial_guess>`
  - ```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_initial_guess
    :summary:
    ```
````

### API

````{py:data} dict_shear_modes
:canonical: stonedfenicsx.config.numerical_control.dict_shear_modes
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_shear_modes
```

````

````{py:data} dict_solver_type
:canonical: stonedfenicsx.config.numerical_control.dict_solver_type
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_solver_type
```

````

````{py:data} dict_output_type
:canonical: stonedfenicsx.config.numerical_control.dict_output_type
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_output_type
```

````

````{py:data} dict_initial_guess
:canonical: stonedfenicsx.config.numerical_control.dict_initial_guess
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.dict_initial_guess
```

````

`````{py:class} NumericalControls
:canonical: stonedfenicsx.config.numerical_control.NumericalControls

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls
```

````{py:attribute} it_max
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.it_max
:type: int
:value: >
   20

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.it_max
```

````

````{py:attribute} it_inner_max
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.it_inner_max
:type: int
:value: >
   10

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.it_inner_max
```

````

````{py:attribute} tol
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.tol
:type: float
:value: >
   0.0001

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.tol
```

````

````{py:attribute} tol_dtemp
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.tol_dtemp
:type: float
:value: >
   0.05

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.tol_dtemp
```

````

````{py:attribute} relax
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.relax
:type: float
:value: >
   1.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.relax
```

````

````{py:attribute} g
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.g
:type: float
:value: >
   9.81

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.g
```

````

````{py:attribute} time_max
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.time_max
:type: float
:value: >
   30.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.time_max
```

````

````{py:attribute} dt
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.dt
:type: float
:value: >
   500.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.dt
```

````

````{py:attribute} steady_state
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.steady_state
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.steady_state
```

````

````{py:attribute} decoupling_ctrl
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.decoupling_ctrl
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.decoupling_ctrl
```

````

````{py:attribute} model_shear
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.model_shear
:type: str | int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.model_shear
```

````

````{py:attribute} adiabatic_heating
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.adiabatic_heating
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.adiabatic_heating
```

````

````{py:attribute} stokes_solver_type
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.stokes_solver_type
:type: str | int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.stokes_solver_type
```

````

````{py:attribute} energy_solver_type
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.energy_solver_type
:type: str | int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.energy_solver_type
```

````

````{py:attribute} iterative_solver_tol
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.iterative_solver_tol
:type: float
:value: >
   1e-07

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.iterative_solver_tol
```

````

````{py:attribute} eta_max
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.eta_max
:type: float
:value: >
   1e+26

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.eta_max
```

````

````{py:attribute} pressure_dependency
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.pressure_dependency
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.pressure_dependency
```

````

````{py:attribute} time_ini_guess
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.time_ini_guess
:type: float
:value: >
   0.3

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.time_ini_guess
```

````

````{py:attribute} initial_guess
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.initial_guess
:type: str
:value: >
   'None'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.initial_guess
```

````

````{py:attribute} CFL
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.CFL
:type: float
:value: >
   0.8

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.CFL
```

````

````{py:method} update_initial_guess()
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.update_initial_guess

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.update_initial_guess
```

````

````{py:method} convert_string()
:canonical: stonedfenicsx.config.numerical_control.NumericalControls.convert_string

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.NumericalControls.convert_string
```

````

`````

`````{py:class} IOControls
:canonical: stonedfenicsx.config.numerical_control.IOControls

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls
```

````{py:attribute} test_name
:canonical: stonedfenicsx.config.numerical_control.IOControls.test_name
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.test_name
```

````

````{py:attribute} path_save
:canonical: stonedfenicsx.config.numerical_control.IOControls.path_save
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.path_save
```

````

````{py:attribute} sname
:canonical: stonedfenicsx.config.numerical_control.IOControls.sname
:type: str
:value: >
   'MockTest'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.sname
```

````

````{py:attribute} ts_out
:canonical: stonedfenicsx.config.numerical_control.IOControls.ts_out
:type: int
:value: >
   10

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.ts_out
```

````

````{py:attribute} dt_out
:canonical: stonedfenicsx.config.numerical_control.IOControls.dt_out
:type: float
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.dt_out
```

````

````{py:attribute} ts_time
:canonical: stonedfenicsx.config.numerical_control.IOControls.ts_time
:type: str | int
:value: >
   'step'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.ts_time
```

````

````{py:attribute} path_test
:canonical: stonedfenicsx.config.numerical_control.IOControls.path_test
:type: pathlib.Path
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.path_test
```

````

````{py:attribute} path_cached_information
:canonical: stonedfenicsx.config.numerical_control.IOControls.path_cached_information
:type: pathlib.Path
:value: >
   'Path(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.path_cached_information
```

````

````{py:method} generate_io() -> None
:canonical: stonedfenicsx.config.numerical_control.IOControls.generate_io

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.IOControls.generate_io
```

````

`````

`````{py:class} CTRLBC
:canonical: stonedfenicsx.config.numerical_control.CTRLBC

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC
```

````{py:attribute} constant
:canonical: stonedfenicsx.config.numerical_control.CTRLBC.constant
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC.constant
```

````

````{py:attribute} interval_val
:canonical: stonedfenicsx.config.numerical_control.CTRLBC.interval_val
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC.interval_val
```

````

````{py:attribute} interval_time
:canonical: stonedfenicsx.config.numerical_control.CTRLBC.interval_time
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC.interval_time
```

````

````{py:method} check_time_variation(ctrl: stonedfenicsx.config.numerical_control.NumericalControls)
:canonical: stonedfenicsx.config.numerical_control.CTRLBC.check_time_variation

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC.check_time_variation
```

````

````{py:method} update_vel_age(t: float) -> float
:canonical: stonedfenicsx.config.numerical_control.CTRLBC.update_vel_age

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CTRLBC.update_vel_age
```

````

`````

`````{py:class} CtrlTemperatureBC
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC

Bases: {py:obj}`stonedfenicsx.config.numerical_control.CTRLBC`

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC
```

````{py:attribute} temp_top
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temp_top
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temp_top
```

````

````{py:attribute} temp_max
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temp_max
:type: float
:value: >
   1300.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temp_max
```

````

````{py:attribute} nz
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.nz
:type: int
:value: >
   200

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.nz
```

````

````{py:attribute} dt
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.dt
:type: float
:value: >
   0.005

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.dt
```

````

````{py:attribute} slab_age
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.slab_age
:type: float
:value: >
   50.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.slab_age
```

````

````{py:attribute} recalculate
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.recalculate
:type: int
:value: >
   0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.recalculate
```

````

````{py:attribute} k
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.k
:type: float
:value: >
   3.1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.k
```

````

````{py:attribute} rho
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.rho
:type: float
:value: >
   3300.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.rho
```

````

````{py:attribute} cp
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.cp
:type: float
:value: >
   1250.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.cp
```

````

````{py:attribute} dz
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.dz
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.dz
```

````

````{py:attribute} end_time
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.end_time
:type: float
:value: >
   180

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.end_time
```

````

````{py:attribute} nt
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.nt
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.nt
```

````

````{py:attribute} right_boundary
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.right_boundary
:type: str
:value: >
   'Continental'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.right_boundary
```

````

````{py:attribute} right_age
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.right_age
:type: float
:value: >
   30.0

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.right_age
```

````

````{py:attribute} z
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.z
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.z
```

````

````{py:attribute} z_right
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.z_right
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.z_right
```

````

````{py:attribute} temp_1d_right
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temp_1d_right
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temp_1d_right
```

````

````{py:attribute} temperature_1d
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temperature_1d
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temperature_1d
```

````

````{py:attribute} temperature_2d_field
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temperature_2d_field
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.temperature_2d_field
```

````

````{py:attribute} t_res_vec
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.t_res_vec
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.t_res_vec
```

````

````{py:attribute} self_consistent_flag
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.self_consistent_flag
:type: int
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.self_consistent_flag
```

````

````{py:method} update_thermal_bc(g_input: stonedfenicsx.config.geometry.GeomInput, ctrl: stonedfenicsx.config.numerical_control.NumericalControls) -> None
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.update_thermal_bc

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.update_thermal_bc
```

````

````{py:method} update_1d_vector_left()
:canonical: stonedfenicsx.config.numerical_control.CtrlTemperatureBC.update_1d_vector_left

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlTemperatureBC.update_1d_vector_left
```

````

`````

`````{py:class} CtrlKy
:canonical: stonedfenicsx.config.numerical_control.CtrlKy

Bases: {py:obj}`stonedfenicsx.config.numerical_control.CTRLBC`

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlKy
```

````{py:attribute} v_s
:canonical: stonedfenicsx.config.numerical_control.CtrlKy.v_s
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlKy.v_s
```

````

````{py:method} check_kinematic_bc(ctrl: stonedfenicsx.config.numerical_control.NumericalControls)
:canonical: stonedfenicsx.config.numerical_control.CtrlKy.check_kinematic_bc

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.CtrlKy.check_kinematic_bc
```

````

`````

`````{py:class} SimulationControls
:canonical: stonedfenicsx.config.numerical_control.SimulationControls

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls
```

````{py:attribute} g_input
:canonical: stonedfenicsx.config.numerical_control.SimulationControls.g_input
:type: dataclasses.InitVar[stonedfenicsx.config.geometry.GeomInput]
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls.g_input
```

````

````{py:attribute} ctrl
:canonical: stonedfenicsx.config.numerical_control.SimulationControls.ctrl
:type: stonedfenicsx.config.numerical_control.NumericalControls
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls.ctrl
```

````

````{py:attribute} ctrl_io
:canonical: stonedfenicsx.config.numerical_control.SimulationControls.ctrl_io
:type: stonedfenicsx.config.numerical_control.IOControls
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls.ctrl_io
```

````

````{py:attribute} ctrl_tbc
:canonical: stonedfenicsx.config.numerical_control.SimulationControls.ctrl_tbc
:type: stonedfenicsx.config.numerical_control.CtrlTemperatureBC
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls.ctrl_tbc
```

````

````{py:attribute} ctrl_ky
:canonical: stonedfenicsx.config.numerical_control.SimulationControls.ctrl_ky
:type: stonedfenicsx.config.numerical_control.CtrlKy
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls.ctrl_ky
```

````

````{py:attribute} _scaled
:canonical: stonedfenicsx.config.numerical_control.SimulationControls._scaled
:type: bool
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls._scaled
```

````

````{py:method} __post_init__(g_input: stonedfenicsx.config.geometry.GeomInput)
:canonical: stonedfenicsx.config.numerical_control.SimulationControls.__post_init__

```{autodoc2-docstring} stonedfenicsx.config.numerical_control.SimulationControls.__post_init__
```

````

`````
