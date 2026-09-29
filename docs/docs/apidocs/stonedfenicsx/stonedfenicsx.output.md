# {py:mod}`stonedfenicsx.output`

```{py:module} stonedfenicsx.output
```

```{autodoc2-docstring} stonedfenicsx.output
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OUTPUT <stonedfenicsx.output.OUTPUT>`
  - ```{autodoc2-docstring} stonedfenicsx.output.OUTPUT
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_benchmark_van_keken <stonedfenicsx.output._benchmark_van_keken>`
  - ```{autodoc2-docstring} stonedfenicsx.output._benchmark_van_keken
    :summary:
    ```
````

### API

`````{py:class} OUTPUT(domain: stonedfenicsx.config.geometry.Domain, ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, sc: stonedfenicsx.config.scal.Scal, pdb: stonedfenicsx.config.phase_db.PhaseDataBase, cach_mat_thermal: stonedfenicsx.material_property.compute_material_property.THERMALCACHED, comm=MPI.COMM_WORLD)
:canonical: stonedfenicsx.output.OUTPUT

```{autodoc2-docstring} stonedfenicsx.output.OUTPUT
```

```{rubric} Initialization
```

```{autodoc2-docstring} stonedfenicsx.output.OUTPUT.__init__
```

````{py:method} print_output(ctrl_sim: stonedfenicsx.config.numerical_control.SimulationControls, sol: stonedfenicsx.solver_module.problems_solution.Solution, sc: stonedfenicsx.config.scal.Scal, it_outer: int = 0, time: float = 0.0, ts: int = 0, debug: int = 0) -> None
:canonical: stonedfenicsx.output.OUTPUT.print_output

```{autodoc2-docstring} stonedfenicsx.output.OUTPUT.print_output
```

````

````{py:method} close()
:canonical: stonedfenicsx.output.OUTPUT.close

```{autodoc2-docstring} stonedfenicsx.output.OUTPUT.close
```

````

`````

````{py:function} _benchmark_van_keken(sol: stonedfenicsx.solver_module.problems_solution.Solution, ctrl_io: stonedfenicsx.config.numerical_control.IOControls, sc: stonedfenicsx.config.scal.Scal) -> None
:canonical: stonedfenicsx.output._benchmark_van_keken

```{autodoc2-docstring} stonedfenicsx.output._benchmark_van_keken
```
````
