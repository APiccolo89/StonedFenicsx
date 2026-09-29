# {py:mod}`stonedfenicsx.solver_module.solver_utilities`

```{py:module} stonedfenicsx.solver_module.solver_utilities
```

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OUTERITERATION_SOL_VAL <stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_residuum <stonedfenicsx.solver_module.solver_utilities.compute_residuum>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.compute_residuum
    :summary:
    ```
* - {py:obj}`min_max_array <stonedfenicsx.solver_module.solver_utilities.min_max_array>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.min_max_array
    :summary:
    ```
* - {py:obj}`update_solution <stonedfenicsx.solver_module.solver_utilities.update_solution>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.update_solution
    :summary:
    ```
* - {py:obj}`decoupling_function <stonedfenicsx.solver_module.solver_utilities.decoupling_function>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.decoupling_function
    :summary:
    ```
* - {py:obj}`L2_norm_calculation <stonedfenicsx.solver_module.solver_utilities.L2_norm_calculation>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.L2_norm_calculation
    :summary:
    ```
* - {py:obj}`timestep_output <stonedfenicsx.solver_module.solver_utilities.timestep_output>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.timestep_output
    :summary:
    ```
````

### API

`````{py:class} OUTERITERATION_SOL_VAL
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL
```

````{py:attribute} sol
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.sol
:type: dataclasses.InitVar[stonedfenicsx.solver_module.problems_solution.Solution]
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.sol
```

````

````{py:attribute} ctrl
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.ctrl
:type: dataclasses.InitVar[stonedfenicsx.config.numerical_control.NumericalControls]
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.ctrl
```

````

````{py:attribute} ctrl_io
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.ctrl_io
:type: dataclasses.InitVar[stonedfenicsx.config.numerical_control.IOControls]
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.ctrl_io
```

````

````{py:attribute} T
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.T
:type: dolfinx.fem.function
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.T
```

````

````{py:attribute} PL
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.PL
:type: dolfinx.fem.function
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.PL
```

````

````{py:attribute} u
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.u
:type: dolfinx.fem.function
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.u
```

````

````{py:attribute} p
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.p
:type: dolfinx.fem.function
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.p
```

````

````{py:attribute} mom_res_wedge
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.mom_res_wedge
:type: numpy.typing.NDArray[float]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.mom_res_wedge
```

````

````{py:attribute} mom_res_slab
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.mom_res_slab
:type: numpy.typing.NDArray[float]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.mom_res_slab
```

````

````{py:attribute} div_res_slab
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.div_res_slab
:type: numpy.typing.NDArray[float]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.div_res_slab
```

````

````{py:attribute} div_res_wedge
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.div_res_wedge
:type: numpy.typing.NDArray[float]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.div_res_wedge
```

````

````{py:attribute} ene_res_gl
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.ene_res_gl
:type: numpy.typing.NDArray[float]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.ene_res_gl
```

````

````{py:attribute} combined_residual_0
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.combined_residual_0
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.combined_residual_0
```

````

````{py:attribute} res
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.res
:type: numpy.typing.NDArray[float]
:value: >
   1.0

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.res
```

````

````{py:attribute} old_t_max
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.old_t_max
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.old_t_max
```

````

````{py:attribute} old_t_min
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.old_t_min
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.old_t_min
```

````

````{py:method} __post_init__(sol: stonedfenicsx.solver_module.problems_solution.Solution, ctrl: stonedfenicsx.config.numerical_control.NumericalControls, ctrl_io: stonedfenicsx.config.numerical_control.IOControls)
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.__post_init__

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.__post_init__
```

````

````{py:method} update_iteration(sol)
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.update_iteration

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.update_iteration
```

````

````{py:method} check_convergence(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, res_total: float, r_tot_conv: float, dtemp_l1: float, res_alt: float, ts: int, it_outer: int, rmom_wg: float, reseg: float) -> int
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.check_convergence

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.check_convergence
```

````

````{py:method} compute_residuum_outer(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int, sc: stonedfenicsx.config.scal.Scal, tA: float, ts: int, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls) -> tuple[float]
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.compute_residuum_outer

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.compute_residuum_outer
```

````

````{py:method} compute_residuum_outer_initial_guess_diffusion(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int, sc: stonedfenicsx.config.scal.Scal, tA: float, ts: int, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls) -> tuple[float]
:canonical: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.compute_residuum_outer_initial_guess_diffusion

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL.compute_residuum_outer_initial_guess_diffusion
```

````

`````

````{py:function} compute_residuum(a: dolfinx.fem.Function, b: dolfinx.fem.Function) -> float
:canonical: stonedfenicsx.solver_module.solver_utilities.compute_residuum

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.compute_residuum
```
````

````{py:function} min_max_array(a: dolfinx.fem.function.Function, vel=False) -> numpy.typing.NDArray[numpy.float64]
:canonical: stonedfenicsx.solver_module.solver_utilities.min_max_array

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.min_max_array
```
````

````{py:function} update_solution(sk1: dolfinx.fem.function.Function, sk0: dolfinx.fem.function.Function, tol: float) -> None
:canonical: stonedfenicsx.solver_module.solver_utilities.update_solution

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.update_solution
```
````

````{py:function} decoupling_function(z: numpy.ndarray, fun: dolfinx.fem.Function, g_input: stonedfenicsx.config.geometry.GeomInput) -> dolfinx.fem.Function
:canonical: stonedfenicsx.solver_module.solver_utilities.decoupling_function

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.decoupling_function
```
````

````{py:function} L2_norm_calculation(f: dolfinx.fem.Function) -> float
:canonical: stonedfenicsx.solver_module.solver_utilities.L2_norm_calculation

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.L2_norm_calculation
```
````

````{py:function} timestep_output(ctrlio: stonedfenicsx.config.numerical_control.IOControls, ts: int, t: float, time_previous: float, flag_save: bool) -> bool
:canonical: stonedfenicsx.solver_module.solver_utilities.timestep_output

```{autodoc2-docstring} stonedfenicsx.solver_module.solver_utilities.timestep_output
```
````
