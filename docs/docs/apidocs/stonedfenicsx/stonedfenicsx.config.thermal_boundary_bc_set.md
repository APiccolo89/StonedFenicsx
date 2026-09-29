# {py:mod}`stonedfenicsx.config.thermal_boundary_bc_set`

```{py:module} stonedfenicsx.config.thermal_boundary_bc_set
```

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`save_data_set <stonedfenicsx.config.thermal_boundary_bc_set.save_data_set>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.save_data_set
    :summary:
    ```
* - {py:obj}`_compute_lithostatic_pressure <stonedfenicsx.config.thermal_boundary_bc_set._compute_lithostatic_pressure>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set._compute_lithostatic_pressure
    :summary:
    ```
* - {py:obj}`compute_cp_k_rho <stonedfenicsx.config.thermal_boundary_bc_set.compute_cp_k_rho>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.compute_cp_k_rho
    :summary:
    ```
* - {py:obj}`build_coefficient_matrix <stonedfenicsx.config.thermal_boundary_bc_set.build_coefficient_matrix>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.build_coefficient_matrix
    :summary:
    ```
* - {py:obj}`fill_phase_properties <stonedfenicsx.config.thermal_boundary_bc_set.fill_phase_properties>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.fill_phase_properties
    :summary:
    ```
* - {py:obj}`compute_half_space_cooling_model_analytical <stonedfenicsx.config.thermal_boundary_bc_set.compute_half_space_cooling_model_analytical>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.compute_half_space_cooling_model_analytical
    :summary:
    ```
* - {py:obj}`initialise_geometry_1d <stonedfenicsx.config.thermal_boundary_bc_set.initialise_geometry_1d>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.initialise_geometry_1d
    :summary:
    ```
* - {py:obj}`solve_temperature_1d_bc <stonedfenicsx.config.thermal_boundary_bc_set.solve_temperature_1d_bc>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.solve_temperature_1d_bc
    :summary:
    ```
* - {py:obj}`compute_thermal_boundary <stonedfenicsx.config.thermal_boundary_bc_set.compute_thermal_boundary>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.compute_thermal_boundary
    :summary:
    ```
* - {py:obj}`configure_thermal_bc <stonedfenicsx.config.thermal_boundary_bc_set.configure_thermal_bc>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.configure_thermal_bc
    :summary:
    ```
* - {py:obj}`configure_boundary_condition <stonedfenicsx.config.thermal_boundary_bc_set.configure_boundary_condition>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.configure_boundary_condition
    :summary:
    ```
* - {py:obj}`test_configure_boundary <stonedfenicsx.config.thermal_boundary_bc_set.test_configure_boundary>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.test_configure_boundary
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_NAME_H5_FILE_TMP <stonedfenicsx.config.thermal_boundary_bc_set._NAME_H5_FILE_TMP>`
  - ```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set._NAME_H5_FILE_TMP
    :summary:
    ```
````

### API

````{py:data} _NAME_H5_FILE_TMP
:canonical: stonedfenicsx.config.thermal_boundary_bc_set._NAME_H5_FILE_TMP
:value: >
   'temporary_file.h5'

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set._NAME_H5_FILE_TMP
```

````

````{py:function} save_data_set(f: h5py.File, buf: any, name: str) -> None
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.save_data_set

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.save_data_set
```
````

````{py:function} _compute_lithostatic_pressure(nz: int, ph: numpy.typing.NDArray[numpy.int32], g: float, dz: float, temp: numpy.typing.NDArray[numpy.float64], pdb: stonedfenicsx.config.phase_db.PhaseDataBase) -> numpy.typing.NDArray[numpy.float64]
:canonical: stonedfenicsx.config.thermal_boundary_bc_set._compute_lithostatic_pressure

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set._compute_lithostatic_pressure
```
````

````{py:function} compute_cp_k_rho(ph: numpy.typing.NDArray[numpy.int32], pdb: stonedfenicsx.config.phase_db.PhaseDataBase, temp: numpy.typing.NDArray[numpy.float64], pres: numpy.typing.NDArray[numpy.float64]) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.compute_cp_k_rho

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.compute_cp_k_rho
```
````

````{py:function} build_coefficient_matrix(pdb: stonedfenicsx.config.phase_db.PhaseDataBase, ph: numpy.typing.NDArray[numpy.int32], temp_old: numpy.typing.NDArray[numpy.float64], temp_guess: numpy.typing.NDArray[numpy.float64], temp_pr: numpy.typing.NDArray[numpy.float64], step: int, lit_p: numpy.typing.NDArray[numpy.float64], temp_min: numpy.float64, temp_max: numpy.float64, nz: int, dt: numpy.float64, dz_m: numpy.float64) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.build_coefficient_matrix

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.build_coefficient_matrix
```
````

````{py:function} fill_phase_properties(g_input: stonedfenicsx.config.geometry.GeomInput, z: numpy.typing.NDArray[numpy.float64], left_right: bool) -> numpy.typing.NDArray[numpy.int32]
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.fill_phase_properties

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.fill_phase_properties
```
````

````{py:function} compute_half_space_cooling_model_analytical(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, z: numpy.typing.NDArray[numpy.float64]) -> None
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.compute_half_space_cooling_model_analytical

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.compute_half_space_cooling_model_analytical
```
````

````{py:function} initialise_geometry_1d(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, g_input: stonedfenicsx.config.geometry.GeomInput, left_right: bool) -> tuple[numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.float64]]
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.initialise_geometry_1d

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.initialise_geometry_1d
```
````

````{py:function} solve_temperature_1d_bc(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, g_input: stonedfenicsx.config.geometry.GeomInput, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, z: numpy.typing.NDArray[numpy.float64], ph: numpy.typing.NDArray[numpy.int32], g: float, left_right: bool)
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.solve_temperature_1d_bc

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.solve_temperature_1d_bc
```
````

````{py:function} compute_thermal_boundary(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, ctrl: stonedfenicsx.config.numerical_control.NumericalControls, ioctrl: stonedfenicsx.config.numerical_control.IOControls, sc: stonedfenicsx.config.scal.Scal, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, g_input: stonedfenicsx.config.geometry.GeomInput, save_data: bool, left_right: bool) -> None
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.compute_thermal_boundary

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.compute_thermal_boundary
```
````

````{py:function} configure_thermal_bc(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, ctrl: stonedfenicsx.config.numerical_control.NumericalControls, ioctrl: stonedfenicsx.config.numerical_control.IOControls, sc: stonedfenicsx.config.scal.Scal, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, g_input: stonedfenicsx.config.geometry.GeomInput, left_right: bool) -> None
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.configure_thermal_bc

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.configure_thermal_bc
```
````

````{py:function} configure_boundary_condition(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, ctrl: stonedfenicsx.config.numerical_control.NumericalControls, ioctrl: stonedfenicsx.config.numerical_control.IOControls, sc: stonedfenicsx.config.scal.Scal, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, g_input: stonedfenicsx.config.geometry.GeomInput) -> None
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.configure_boundary_condition

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.configure_boundary_condition
```
````

````{py:function} test_configure_boundary()
:canonical: stonedfenicsx.config.thermal_boundary_bc_set.test_configure_boundary

```{autodoc2-docstring} stonedfenicsx.config.thermal_boundary_bc_set.test_configure_boundary
```
````
