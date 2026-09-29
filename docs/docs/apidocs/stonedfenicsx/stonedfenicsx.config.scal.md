# {py:mod}`stonedfenicsx.config.scal`

```{py:module} stonedfenicsx.config.scal
```

```{autodoc2-docstring} stonedfenicsx.config.scal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Scal <stonedfenicsx.config.scal.Scal>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.Scal
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`scaling_simulation_physical <stonedfenicsx.config.scal.scaling_simulation_physical>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_simulation_physical
    :summary:
    ```
* - {py:obj}`scaling_material_properties <stonedfenicsx.config.scal.scaling_material_properties>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_material_properties
    :summary:
    ```
* - {py:obj}`scale_parameters <stonedfenicsx.config.scal.scale_parameters>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scale_parameters
    :summary:
    ```
* - {py:obj}`scaling_control_parameters <stonedfenicsx.config.scal.scaling_control_parameters>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_control_parameters
    :summary:
    ```
* - {py:obj}`scaling_mesh <stonedfenicsx.config.scal.scaling_mesh>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_mesh
    :summary:
    ```
* - {py:obj}`dimensionless_ginput <stonedfenicsx.config.scal.dimensionless_ginput>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.dimensionless_ginput
    :summary:
    ```
* - {py:obj}`scale_kinematic_bc <stonedfenicsx.config.scal.scale_kinematic_bc>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scale_kinematic_bc
    :summary:
    ```
* - {py:obj}`scaling_io_controls <stonedfenicsx.config.scal.scaling_io_controls>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_io_controls
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_KELVIN_ <stonedfenicsx.config.scal._KELVIN_>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal._KELVIN_
    :summary:
    ```
* - {py:obj}`_KM_2_M_ <stonedfenicsx.config.scal._KM_2_M_>`
  - ```{autodoc2-docstring} stonedfenicsx.config.scal._KM_2_M_
    :summary:
    ```
````

### API

````{py:data} _KELVIN_
:canonical: stonedfenicsx.config.scal._KELVIN_
:value: >
   273.15

```{autodoc2-docstring} stonedfenicsx.config.scal._KELVIN_
```

````

````{py:data} _KM_2_M_
:canonical: stonedfenicsx.config.scal._KM_2_M_
:value: >
   1000

```{autodoc2-docstring} stonedfenicsx.config.scal._KM_2_M_
```

````

`````{py:class} Scal
:canonical: stonedfenicsx.config.scal.Scal

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal
```

````{py:attribute} length
:canonical: stonedfenicsx.config.scal.Scal.length
:type: float
:value: >
   1

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.length
```

````

````{py:attribute} temp
:canonical: stonedfenicsx.config.scal.Scal.temp
:type: float
:value: >
   1000

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.temp
```

````

````{py:attribute} eta
:canonical: stonedfenicsx.config.scal.Scal.eta
:type: float
:value: >
   1e+24

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.eta
```

````

````{py:attribute} stress
:canonical: stonedfenicsx.config.scal.Scal.stress
:type: float
:value: >
   1000000000.0

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.stress
```

````

````{py:attribute} time
:canonical: stonedfenicsx.config.scal.Scal.time
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.time
```

````

````{py:attribute} mass
:canonical: stonedfenicsx.config.scal.Scal.mass
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.mass
```

````

````{py:attribute} ac
:canonical: stonedfenicsx.config.scal.Scal.ac
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.ac
```

````

````{py:attribute} rho
:canonical: stonedfenicsx.config.scal.Scal.rho
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.rho
```

````

````{py:attribute} force
:canonical: stonedfenicsx.config.scal.Scal.force
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.force
```

````

````{py:attribute} energy
:canonical: stonedfenicsx.config.scal.Scal.energy
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.energy
```

````

````{py:attribute} watt
:canonical: stonedfenicsx.config.scal.Scal.watt
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.watt
```

````

````{py:attribute} strain_rate
:canonical: stonedfenicsx.config.scal.Scal.strain_rate
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.strain_rate
```

````

````{py:attribute} k
:canonical: stonedfenicsx.config.scal.Scal.k
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.k
```

````

````{py:attribute} cp
:canonical: stonedfenicsx.config.scal.Scal.cp
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.cp
```

````

````{py:attribute} scale_vel
:canonical: stonedfenicsx.config.scal.Scal.scale_vel
:type: float
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.scale_vel
```

````

````{py:attribute} scale_myr2sec
:canonical: stonedfenicsx.config.scal.Scal.scale_myr2sec
:type: float
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.scale_myr2sec
```

````

````{py:method} compute_the_derivative_scal()
:canonical: stonedfenicsx.config.scal.Scal.compute_the_derivative_scal

```{autodoc2-docstring} stonedfenicsx.config.scal.Scal.compute_the_derivative_scal
```

````

`````

````{py:function} scaling_simulation_physical(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, mesh: stonedfenicsx.config.geometry.Mesh, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.scaling_simulation_physical

```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_simulation_physical
```
````

````{py:function} scaling_material_properties(pdb: stonedfenicsx.config.phase_db.PhaseDataBase, sc: stonedfenicsx.config.scal.Scal) -> stonedfenicsx.config.phase_db.PhaseDataBase
:canonical: stonedfenicsx.config.scal.scaling_material_properties

```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_material_properties
```
````

````{py:function} scale_parameters(ctrl_tbc: stonedfenicsx.config.numerical_control.CtrlTemperatureBC, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.scale_parameters

```{autodoc2-docstring} stonedfenicsx.config.scal.scale_parameters
```
````

````{py:function} scaling_control_parameters(ctrl: stonedfenicsx.config.numerical_control.NumericalControls, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.scaling_control_parameters

```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_control_parameters
```
````

````{py:function} scaling_mesh(mesh: stonedfenicsx.config.geometry.Mesh, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.scaling_mesh

```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_mesh
```
````

````{py:function} dimensionless_ginput(g_input: stonedfenicsx.config.geometry.GeomInput, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.dimensionless_ginput

```{autodoc2-docstring} stonedfenicsx.config.scal.dimensionless_ginput
```
````

````{py:function} scale_kinematic_bc(ctrl_ky: stonedfenicsx.config.numerical_control.CtrlKy, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.scale_kinematic_bc

```{autodoc2-docstring} stonedfenicsx.config.scal.scale_kinematic_bc
```
````

````{py:function} scaling_io_controls(ctrl_io: stonedfenicsx.config.numerical_control.IOControls, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.config.scal.scaling_io_controls

```{autodoc2-docstring} stonedfenicsx.config.scal.scaling_io_controls
```
````
