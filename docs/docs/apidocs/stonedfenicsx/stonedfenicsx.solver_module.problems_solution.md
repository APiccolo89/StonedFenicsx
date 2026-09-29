# {py:mod}`stonedfenicsx.solver_module.problems_solution`

```{py:module} stonedfenicsx.solver_module.problems_solution
```

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CACHED_FEM_FORM <stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM
    :summary:
    ```
* - {py:obj}`Problem <stonedfenicsx.solver_module.problems_solution.Problem>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem
    :summary:
    ```
* - {py:obj}`Solution <stonedfenicsx.solver_module.problems_solution.Solution>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Solution
    :summary:
    ```
* - {py:obj}`Global_thermal <stonedfenicsx.solver_module.problems_solution.Global_thermal>`
  -
* - {py:obj}`Global_pressure <stonedfenicsx.solver_module.problems_solution.Global_pressure>`
  -
* - {py:obj}`Stokes_Problem <stonedfenicsx.solver_module.problems_solution.Stokes_Problem>`
  -
* - {py:obj}`Wedge <stonedfenicsx.solver_module.problems_solution.Wedge>`
  -
* - {py:obj}`Slab <stonedfenicsx.solver_module.problems_solution.Slab>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Slab
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`debug_boundary_condition <stonedfenicsx.solver_module.problems_solution.debug_boundary_condition>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.debug_boundary_condition
    :summary:
    ```
````

### API

````{py:function} debug_boundary_condition(bc, name)
:canonical: stonedfenicsx.solver_module.problems_solution.debug_boundary_condition

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.debug_boundary_condition
```
````

`````{py:class} CACHED_FEM_FORM
:canonical: stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM
```

````{py:attribute} a
:canonical: stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM.a
:type: dolfinx.fem.Form | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM.a
```

````

````{py:attribute} L
:canonical: stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM.L
:type: dolfinx.fem.Form | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM.L
```

````

````{py:attribute} other_form
:canonical: stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM.other_form
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM.other_form
```

````

`````

`````{py:class} Problem(mesh: stonedfenicsx.config.geometry.Mesh, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, elements: tuple, name: list)
:canonical: stonedfenicsx.solver_module.problems_solution.Problem

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem
```

```{rubric} Initialization
```

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.__init__
```

````{py:attribute} name
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.name
:type: list
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.name
```

````

````{py:attribute} mixed
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.mixed
:type: bool
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.mixed
```

````

````{py:attribute} domain
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.domain
:type: stonedfenicsx.config.geometry.Domain
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.domain
```

````

````{py:attribute} ctrl_sim
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.ctrl_sim
:type: stonedfenicsx.config.numerical_control.SimulationControls
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.ctrl_sim
```

````

````{py:attribute} g_input
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.g_input
:type: stonedfenicsx.config.geometry.GeomInput
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.g_input
```

````

````{py:attribute} pdb
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.pdb
:type: stonedfenicsx.config.phase_db.PhaseDataBase
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.pdb
```

````

````{py:attribute} cached_mat
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.cached_mat
:type: stonedfenicsx.material_property.compute_material_property.MATERIALS
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.cached_mat
```

````

````{py:attribute} FS
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.FS
:type: dolfinx.fem.FunctionSpace
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.FS
```

````

````{py:attribute} F0
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.F0
:type: dolfinx.fem.FunctionSpace | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.F0
```

````

````{py:attribute} F1
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.F1
:type: dolfinx.fem.FunctionSpace | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.F1
```

````

````{py:attribute} trial0
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.trial0
:type: ufl.Argument | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.trial0
```

````

````{py:attribute} trial1
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.trial1
:type: ufl.Argument | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.trial1
```

````

````{py:attribute} test0
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.test0
:type: ufl.Argument | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.test0
```

````

````{py:attribute} test1
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.test1
:type: ufl.Argument | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.test1
```

````

````{py:attribute} typology
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.typology
:type: str | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.typology
```

````

````{py:attribute} dofs
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.dofs
:type: numpy.ndarray | None
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.dofs
```

````

````{py:attribute} bc
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.bc
:type: list
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.bc
```

````

````{py:attribute} ds
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.ds
:type: ufl.measure.Measure
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.ds
```

````

````{py:attribute} dx
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.dx
:type: ufl.measure.Measure
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.dx
```

````

````{py:attribute} solv
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.solv
:type: stonedfenicsx.solver_module.solver.Solvers
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.solv
```

````

````{py:attribute} cached_form
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.cached_form
:type: stonedfenicsx.solver_module.problems_solution.CACHED_FEM_FORM
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.cached_form
```

````

````{py:method} create_cached_material(scalar: bool)
:canonical: stonedfenicsx.solver_module.problems_solution.Problem.create_cached_material

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Problem.create_cached_material
```

````

`````

`````{py:class} Solution()
:canonical: stonedfenicsx.solver_module.problems_solution.Solution

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Solution
```

```{rubric} Initialization
```

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Solution.__init__
```

````{py:method} create_function(PG: stonedfenicsx.solver_module.problems_solution.Problem, PS: stonedfenicsx.solver_module.problems_solution.Problem, PW: stonedfenicsx.solver_module.problems_solution.Problem, elements: list) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Solution.create_function

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Solution.create_function
```

````

`````

`````{py:class} Global_thermal(mesh: stonedfenicsx.config.geometry.Mesh, elements: tuple, name: list, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal

Bases: {py:obj}`stonedfenicsx.solver_module.problems_solution.Problem`

````{py:method} set_linear_picard_TD(p: dolfinx.fem.Function = None, T_k: dolfinx.fem.Function = None, T_O: dolfinx.fem.Function = None, u_global: dolfinx.fem.Function = None, it_outer: int = 0, it_inner: int = 0, ts: int = 0) -> tuple[dolfinx.fem.Form, dolfinx.fem.Form | None]
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.set_linear_picard_TD

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.set_linear_picard_TD
```

````

````{py:method} set_linear_picard_SS(p: dolfinx.fem.Function = None, T_k: dolfinx.fem.Function = None, T_O: dolfinx.fem.Function = None, u_global: dolfinx.fem.Function = None, it_outer: int = 0, it_inner: int = 0, ts: int = 0) -> tuple[dolfinx.fem.Form, dolfinx.fem.Form]
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.set_linear_picard_SS

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.set_linear_picard_SS
```

````

````{py:method} set_form_residual_SS(p: dolfinx.fem.function.Function = None, T: dolfinx.fem.function.Function = None, T_O: dolfinx.fem.function.Function = None, u_global: dolfinx.fem.function.Function = None, it_inner: int = 0, L: dolfinx.fem.Form = None) -> float
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.set_form_residual_SS

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.set_form_residual_SS
```

````

````{py:method} set_form_residual_TD(p: dolfinx.fem.function.Function = None, T: dolfinx.fem.function.Function = None, T_O: dolfinx.fem.function.Function = None, u_global: dolfinx.fem.function.Function = None, it_inner: int = 0, L: dolfinx.fem.Form = None) -> float
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.set_form_residual_TD

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.set_form_residual_TD
```

````

````{py:method} initialise_form(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int, ts: int)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.initialise_form

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.initialise_form
```

````

````{py:method} interpolate_1d_vector_boundary(function_space, z, temp_vec, dofs_intp)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.interpolate_1d_vector_boundary
:staticmethod:

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.interpolate_1d_vector_boundary
```

````

````{py:method} create_bc_temp(u_global: dolfinx.fem.Function, it_outer: int, ts=0) -> list
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.create_bc_temp

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.create_bc_temp
```

````

````{py:method} compute_shear_heating(p: dolfinx.fem.Function, T_k: dolfinx.fem.Function) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_shear_heating

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_shear_heating
```

````

````{py:method} compute_friction_shear_expression(T: dolfinx.fem.function.Function, P: dolfinx.fem.function.Function)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_friction_shear_expression

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_friction_shear_expression
```

````

````{py:method} compute_energy_source()
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_energy_source

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_energy_source
```

````

````{py:method} compute_residual()
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_residual

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_residual
```

````

````{py:method} compute_dt_update_dt(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int, ts: int) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_dt_update_dt

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_dt_update_dt
```

````

````{py:method} Solve_the_Problem(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int = 0, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.Solve_the_Problem

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.Solve_the_Problem
```

````

````{py:method} compute_shear_heating_visualisation(sol: stonedfenicsx.solver_module.problems_solution.Solution, ts: int, it_outer: int) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_shear_heating_visualisation

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.compute_shear_heating_visualisation
```

````

````{py:method} solve_the_linear(sol: stonedfenicsx.solver_module.problems_solution.Solution, a: dolfinx.fem.Form, L: dolfinx.fem.Form, fen_function: dolfinx.fem.Function, isPicard: int = 0, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.solve_the_linear

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.solve_the_linear
```

````

````{py:method} initial_temperature_field() -> dolfinx.fem.Function
:canonical: stonedfenicsx.solver_module.problems_solution.Global_thermal.initial_temperature_field

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_thermal.initial_temperature_field
```

````

`````

`````{py:class} Global_pressure(mesh: stonedfenicsx.config.geometry.Mesh, elements: tuple, name: list, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_pressure

Bases: {py:obj}`stonedfenicsx.solver_module.problems_solution.Problem`

````{py:method} set_problem_bc() -> list[dolfinx.fem.DirichletBC]
:canonical: stonedfenicsx.solver_module.problems_solution.Global_pressure.set_problem_bc

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_pressure.set_problem_bc
```

````

````{py:method} set_linear_picard(p_k: dolfinx.fem.Function, T: dolfinx.fem.Function, it: int = 0)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_pressure.set_linear_picard

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_pressure.set_linear_picard
```

````

````{py:method} Solve_the_Problem(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int = 0, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Global_pressure.Solve_the_Problem

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_pressure.Solve_the_Problem
```

````

````{py:method} solve_the_linear(a: dolfinx.fem.Form, L: dolfinx.fem.Form, function_fen: dolfinx.fem.Function, isPicard: int = 0, it: int = 0, ts: int = 0)
:canonical: stonedfenicsx.solver_module.problems_solution.Global_pressure.solve_the_linear

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Global_pressure.solve_the_linear
```

````

`````

`````{py:class} Stokes_Problem(mesh: stonedfenicsx.config.geometry.Mesh, elements: list, name: list, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, pdb: stonedfenicsx.config.phase_db.PhaseDataBase)
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem

Bases: {py:obj}`stonedfenicsx.solver_module.problems_solution.Problem`

````{py:method} fem_stokes_form(a1, a2, a3, a_p)
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.fem_stokes_form

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.fem_stokes_form
```

````

````{py:method} initialise_fem_form(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int, ts: int)
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.initialise_fem_form

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.initialise_fem_form
```

````

````{py:method} compute_residuum_stokes()
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.compute_residuum_stokes

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.compute_residuum_stokes
```

````

````{py:method} set_linear_picard(vel: dolfinx.fem.function.Function, temp: dolfinx.fem.function.Function, pres_l: dolfinx.fem.function.Function, a_p=None, it: int = 0, ts: int = 0, slab=1) -> tuple[dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form]
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.set_linear_picard

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.set_linear_picard
```

````

````{py:method} compute_moving_wall(facet: str) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.compute_moving_wall

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.compute_moving_wall
```

````

````{py:method} solve_linear_picard(a: dolfinx.fem.Form, a_p0: dolfinx.fem.Form, L: dolfinx.fem.Form, u: dolfinx.fem.Function, p: dolfinx.fem.Function, it_outer: int = 0, ts: int = 0, it_inner=0, slab: int = 0) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.solve_linear_picard

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.solve_linear_picard
```

````

````{py:method} Solve_the_Problem(sol: stonedfenicsx.solver_module.problems_solution.Solution, it_outer: int = 0, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.problems_solution.Stokes_Problem.Solve_the_Problem

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Stokes_Problem.Solve_the_Problem
```

````

`````

`````{py:class} Wedge(mesh: stonedfenicsx.config.geometry.Mesh, elements: list, name: list, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, pdb: stonedfenicsx.config.phase_db.PhaseDataBase)
:canonical: stonedfenicsx.solver_module.problems_solution.Wedge

Bases: {py:obj}`stonedfenicsx.solver_module.problems_solution.Stokes_Problem`

````{py:method} setdirichlecht(V: dolfinx.fem.FunctionSpace, it_outer: int = 0, ts: int = 0) -> list
:canonical: stonedfenicsx.solver_module.problems_solution.Wedge.setdirichlecht

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Wedge.setdirichlecht
```

````

`````

`````{py:class} Slab(mesh: stonedfenicsx.config.geometry.Mesh, elements: list, name: list, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, pdb: stonedfenicsx.config.phase_db.PhaseDataBase)
:canonical: stonedfenicsx.solver_module.problems_solution.Slab

Bases: {py:obj}`stonedfenicsx.solver_module.problems_solution.Stokes_Problem`

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Slab
```

```{rubric} Initialization
```

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Slab.__init__
```

````{py:method} setdirichlecht(Vsubs=None, it_outer: int = 0, ts: int = 0) -> list
:canonical: stonedfenicsx.solver_module.problems_solution.Slab.setdirichlecht

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Slab.setdirichlecht
```

````

````{py:method} compute_nitsche_FS(sol: stonedfenicsx.solver_module.problems_solution.Solution, dS: ufl.measure.Measure, a1: ufl.form.Form, a2: ufl.form.Form, a3: ufl.form.Form, gamma: float, it: int = 0) -> tuple([ufl.form.Form, ufl.form.Form, ufl.form.Form])
:canonical: stonedfenicsx.solver_module.problems_solution.Slab.compute_nitsche_FS

```{autodoc2-docstring} stonedfenicsx.solver_module.problems_solution.Slab.compute_nitsche_FS
```

````

`````
