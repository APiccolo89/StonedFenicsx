# {py:mod}`stonedfenicsx.material_property.compute_material_property`

```{py:module} stonedfenicsx.material_property.compute_material_property
```

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MATERIALS <stonedfenicsx.material_property.compute_material_property.MATERIALS>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.MATERIALS
    :summary:
    ```
* - {py:obj}`THERMALCACHED <stonedfenicsx.material_property.compute_material_property.THERMALCACHED>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED
    :summary:
    ```
* - {py:obj}`RHEOLOGYCACHED <stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`heat_conductivity_FX <stonedfenicsx.material_property.compute_material_property.heat_conductivity_FX>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.heat_conductivity_FX
    :summary:
    ```
* - {py:obj}`heat_capacity_FX <stonedfenicsx.material_property.compute_material_property.heat_capacity_FX>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.heat_capacity_FX
    :summary:
    ```
* - {py:obj}`compute_radiogenic <stonedfenicsx.material_property.compute_material_property.compute_radiogenic>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.compute_radiogenic
    :summary:
    ```
* - {py:obj}`density_FX <stonedfenicsx.material_property.compute_material_property.density_FX>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.density_FX
    :summary:
    ```
* - {py:obj}`alpha_FX <stonedfenicsx.material_property.compute_material_property.alpha_FX>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.alpha_FX
    :summary:
    ```
* - {py:obj}`compute_viscosity_FX <stonedfenicsx.material_property.compute_material_property.compute_viscosity_FX>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.compute_viscosity_FX
    :summary:
    ```
* - {py:obj}`compute_plastic_strain <stonedfenicsx.material_property.compute_material_property.compute_plastic_strain>`
  - ```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.compute_plastic_strain
    :summary:
    ```
````

### API

`````{py:class} MATERIALS
:canonical: stonedfenicsx.material_property.compute_material_property.MATERIALS

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.MATERIALS
```

````{py:attribute} pdb
:canonical: stonedfenicsx.material_property.compute_material_property.MATERIALS.pdb
:type: dataclasses.InitVar[stonedfenicsx.config.phase_db.PhaseDataBase]
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.MATERIALS.pdb
```

````

````{py:attribute} phase
:canonical: stonedfenicsx.material_property.compute_material_property.MATERIALS.phase
:type: dataclasses.InitVar[dolfinx.fem.Function]
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.MATERIALS.phase
```

````

`````

`````{py:class} THERMALCACHED
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED

Bases: {py:obj}`stonedfenicsx.material_property.compute_material_property.MATERIALS`

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED
```

````{py:attribute} k0
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k0
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k0
```

````

````{py:attribute} rg_cached
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.rg_cached
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.rg_cached
```

````

````{py:attribute} k_a
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_a
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_a
```

````

````{py:attribute} k_b
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_b
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_b
```

````

````{py:attribute} k_c
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_c
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_c
```

````

````{py:attribute} k_d
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_d
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_d
```

````

````{py:attribute} k_e
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_e
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_e
```

````

````{py:attribute} k_f
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_f
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.k_f
```

````

````{py:attribute} c0
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c0
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c0
```

````

````{py:attribute} c1
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c1
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c1
```

````

````{py:attribute} c2
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c2
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c2
```

````

````{py:attribute} c3
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c3
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c3
```

````

````{py:attribute} c4
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c4
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c4
```

````

````{py:attribute} c5
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c5
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.c5
```

````

````{py:attribute} rho0
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.rho0
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.rho0
```

````

````{py:attribute} alpha0
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.alpha0
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.alpha0
```

````

````{py:attribute} alpha1
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.alpha1
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.alpha1
```

````

````{py:attribute} alpha2
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.alpha2
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.alpha2
```

````

````{py:attribute} kb
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.kb
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.kb
```

````

````{py:attribute} option_rho
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.option_rho
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.option_rho
```

````

````{py:attribute} radiogenic
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.radiogenic
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.radiogenic
```

````

````{py:attribute} temp_ref
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.temp_ref
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.temp_ref
```

````

````{py:attribute} gas_constant
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.gas_constant
:type: float
:value: >
   8.3145

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.gas_constant
```

````

````{py:attribute} a_rad
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.a_rad
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.a_rad
```

````

````{py:attribute} b_rad
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.b_rad
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.b_rad
```

````

````{py:attribute} temp_a
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.temp_a
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.temp_a
```

````

````{py:attribute} x_a
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.x_a
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.x_a
```

````

````{py:attribute} temp_b
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.temp_b
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.temp_b
```

````

````{py:attribute} x_b
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.x_b
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.x_b
```

````

````{py:method} __post_init__(pdb: stonedfenicsx.config.phase_db.PhaseDataBase, phase: dolfinx.fem.Function) -> None
:canonical: stonedfenicsx.material_property.compute_material_property.THERMALCACHED.__post_init__

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.THERMALCACHED.__post_init__
```

````

`````

`````{py:class} RHEOLOGYCACHED
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED

Bases: {py:obj}`stonedfenicsx.material_property.compute_material_property.MATERIALS`

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED
```

````{py:attribute} b_dif
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.b_dif
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.b_dif
```

````

````{py:attribute} b_dis
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.b_dis
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.b_dis
```

````

````{py:attribute} n
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.n
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.n
```

````

````{py:attribute} e_dif
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.e_dif
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.e_dif
```

````

````{py:attribute} e_dis
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.e_dis
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.e_dis
```

````

````{py:attribute} v_dif
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.v_dif
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.v_dif
```

````

````{py:attribute} v_dis
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.v_dis
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.v_dis
```

````

````{py:attribute} eta
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.eta
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.eta
```

````

````{py:attribute} eta_def
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.eta_def
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.eta_def
```

````

````{py:attribute} option_eta
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.option_eta
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.option_eta
```

````

````{py:attribute} eta_max
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.eta_max
:type: float
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.eta_max
```

````

````{py:attribute} gas_constant
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.gas_constant
:type: float
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.gas_constant
```

````

````{py:method} __post_init__(pdb: stonedfenicsx.config.phase_db.PhaseDataBase, phase: dolfinx.fem.Function) -> None
:canonical: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.__post_init__

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED.__post_init__
```

````

`````

````{py:function} heat_conductivity_FX(scal_cached: stonedfenicsx.material_property.compute_material_property.THERMALCACHED, T: dolfinx.fem.Function, p: dolfinx.fem.Function, Cp: dolfinx.fem.Expression, rho: dolfinx.fem.Expression) -> dolfinx.fem.Expression
:canonical: stonedfenicsx.material_property.compute_material_property.heat_conductivity_FX

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.heat_conductivity_FX
```
````

````{py:function} heat_capacity_FX(scal_cached: stonedfenicsx.material_property.compute_material_property.THERMALCACHED, T: dolfinx.fem.Function) -> dolfinx.fem.Expression
:canonical: stonedfenicsx.material_property.compute_material_property.heat_capacity_FX

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.heat_capacity_FX
```
````

````{py:function} compute_radiogenic(scal_cached: stonedfenicsx.material_property.compute_material_property.THERMALCACHED, hs: dolfinx.fem.Function) -> dolfinx.fem.Function
:canonical: stonedfenicsx.material_property.compute_material_property.compute_radiogenic

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.compute_radiogenic
```
````

````{py:function} density_FX(scal_cached: stonedfenicsx.material_property.compute_material_property.THERMALCACHED, T: dolfinx.fem.Function, p: dolfinx.fem.Function) -> dolfinx.fem.Expression
:canonical: stonedfenicsx.material_property.compute_material_property.density_FX

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.density_FX
```
````

````{py:function} alpha_FX(scal_cached: stonedfenicsx.material_property.compute_material_property.THERMALCACHED, T: dolfinx.fem.Function, p: dolfinx.fem.Function) -> dolfinx.fem.Expression
:canonical: stonedfenicsx.material_property.compute_material_property.alpha_FX

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.alpha_FX
```
````

````{py:function} compute_viscosity_FX(e: dolfinx.fem.Expression, temp_in: dolfinx.fem.Function, pres_in: dolfinx.fem.Function, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, rg_cached: stonedfenicsx.material_property.compute_material_property.RHEOLOGYCACHED) -> dolfinx.fem.Expression
:canonical: stonedfenicsx.material_property.compute_material_property.compute_viscosity_FX

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.compute_viscosity_FX
```
````

````{py:function} compute_plastic_strain(e_ii: dolfinx.fem.Expression, temp_in: dolfinx.fem.Function, pres_in: dolfinx.fem.Function, pdb: stonedfenicsx.config.phase_db.PhaseDataBase) -> tuple[dolfinx.fem.Expression, dolfinx.fem.Expression, dolfinx.fem.Expression]
:canonical: stonedfenicsx.material_property.compute_material_property.compute_plastic_strain

```{autodoc2-docstring} stonedfenicsx.material_property.compute_material_property.compute_plastic_strain
```
````
