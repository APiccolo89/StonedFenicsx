# {py:mod}`stonedfenicsx.solver_module.solution_routine`

```{py:module} stonedfenicsx.solver_module.solution_routine
```

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`outerloop_operation_initial_guess <stonedfenicsx.solver_module.solution_routine.outerloop_operation_initial_guess>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.outerloop_operation_initial_guess
    :summary:
    ```
* - {py:obj}`diffusion_initial_guess <stonedfenicsx.solver_module.solution_routine.diffusion_initial_guess>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.diffusion_initial_guess
    :summary:
    ```
* - {py:obj}`initialise_the_simulation <stonedfenicsx.solver_module.solution_routine.initialise_the_simulation>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.initialise_the_simulation
    :summary:
    ```
* - {py:obj}`outerloop_operation <stonedfenicsx.solver_module.solution_routine.outerloop_operation>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.outerloop_operation
    :summary:
    ```
* - {py:obj}`initial_guess_simulation <stonedfenicsx.solver_module.solution_routine.initial_guess_simulation>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.initial_guess_simulation
    :summary:
    ```
* - {py:obj}`time_loop <stonedfenicsx.solver_module.solution_routine.time_loop>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.time_loop
    :summary:
    ```
* - {py:obj}`solution_routine <stonedfenicsx.solver_module.solution_routine.solution_routine>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.solution_routine
    :summary:
    ```
````

### API

````{py:function} outerloop_operation_initial_guess(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, sc: stonedfenicsx.config.scal.Scal, eg: stonedfenicsx.solver_module.problems_solution.Global_thermal, lg: stonedfenicsx.solver_module.problems_solution.Global_pressure, we: stonedfenicsx.solver_module.problems_solution.Wedge, sl: stonedfenicsx.solver_module.problems_solution.Slab, sol: stonedfenicsx.solver_module.problems_solution.Solution, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, outit: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.solution_routine.outerloop_operation_initial_guess

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.outerloop_operation_initial_guess
```
````

````{py:function} diffusion_initial_guess(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, eg: stonedfenicsx.solver_module.problems_solution.Global_thermal, lg: stonedfenicsx.solver_module.problems_solution.Global_pressure, we: stonedfenicsx.solver_module.problems_solution.Wedge, sl: stonedfenicsx.solver_module.problems_solution.Slab, sol: stonedfenicsx.solver_module.problems_solution.Solution, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.solver_module.solution_routine.diffusion_initial_guess

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.diffusion_initial_guess
```
````

````{py:function} initialise_the_simulation(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls = None, pdb: stonedfenicsx.config.phase_db.PhaseDataBase = None, mesh: stonedfenicsx.config.geometry.Mesh = None) -> tuple[stonedfenicsx.solver_module.problems_solution.Solution, stonedfenicsx.solver_module.problems_solution.Global_thermal, stonedfenicsx.solver_module.problems_solution.Global_pressure, stonedfenicsx.solver_module.problems_solution.Wedge, stonedfenicsx.solver_module.problems_solution.Slab]
:canonical: stonedfenicsx.solver_module.solution_routine.initialise_the_simulation

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.initialise_the_simulation
```
````

````{py:function} outerloop_operation(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, sc: stonedfenicsx.config.scal.Scal, eg: stonedfenicsx.solver_module.problems_solution.Global_thermal, lg: stonedfenicsx.solver_module.problems_solution.Global_pressure, we: stonedfenicsx.solver_module.problems_solution.Wedge, sl: stonedfenicsx.solver_module.problems_solution.Slab, sol: stonedfenicsx.solver_module.problems_solution.Solution, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, outit: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.solution_routine.outerloop_operation

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.outerloop_operation
```
````

````{py:function} initial_guess_simulation(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, sc: stonedfenicsx.config.scal.Scal, eg: stonedfenicsx.solver_module.problems_solution.Global_thermal, lg: stonedfenicsx.solver_module.problems_solution.Global_pressure, we: stonedfenicsx.solver_module.problems_solution.Wedge, sl: stonedfenicsx.solver_module.problems_solution.Slab, sol: stonedfenicsx.solver_module.problems_solution.Solution, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, outit: stonedfenicsx.solver_module.solver_utilities.OUTERITERATION_SOL_VAL, ts: int = 0) -> None
:canonical: stonedfenicsx.solver_module.solution_routine.initial_guess_simulation

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.initial_guess_simulation
```
````

````{py:function} time_loop(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, eg: stonedfenicsx.solver_module.problems_solution.Global_thermal, lg: stonedfenicsx.solver_module.problems_solution.Global_pressure, we: stonedfenicsx.solver_module.problems_solution.Wedge, sl: stonedfenicsx.solver_module.problems_solution.Slab, sol: stonedfenicsx.solver_module.problems_solution.Solution, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.solver_module.solution_routine.time_loop

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.time_loop
```
````

````{py:function} solution_routine(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, mesh: stonedfenicsx.config.geometry.Mesh, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.solver_module.solution_routine.solution_routine

```{autodoc2-docstring} stonedfenicsx.solver_module.solution_routine.solution_routine
```
````
