# {py:mod}`stonedfenicsx.solver_module.solver`

```{py:module} stonedfenicsx.solver_module.solver
```

```{autodoc2-docstring} stonedfenicsx.solver_module.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Solvers <stonedfenicsx.solver_module.solver.Solvers>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver.Solvers
    :summary:
    ```
* - {py:obj}`ScalarSolver <stonedfenicsx.solver_module.solver.ScalarSolver>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver.ScalarSolver
    :summary:
    ```
* - {py:obj}`SolverStokes <stonedfenicsx.solver_module.solver.SolverStokes>`
  - ```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes
    :summary:
    ```
````

### API

`````{py:class} Solvers
:canonical: stonedfenicsx.solver_module.solver.Solvers

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.Solvers
```

````{py:method} destroy()
:canonical: stonedfenicsx.solver_module.solver.Solvers.destroy

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.Solvers.destroy
```

````

`````

````{py:class} ScalarSolver(a, L, bcs, COMM, direct=0)
:canonical: stonedfenicsx.solver_module.solver.ScalarSolver

Bases: {py:obj}`stonedfenicsx.solver_module.solver.Solvers`

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.ScalarSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.ScalarSolver.__init__
```

````

`````{py:class} SolverStokes(a, a_p, L, COMM, nl, bcs, F0, F1, ctrl, J=None, r=None, it=0, ts=0, slab=0)
:canonical: stonedfenicsx.solver_module.solver.SolverStokes

Bases: {py:obj}`stonedfenicsx.solver_module.solver.Solvers`

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes
```

```{rubric} Initialization
```

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes.__init__
```

````{py:method} set_direct_solver(a, a_p, L, COMM, nl, bcs, F0, F1, ctrl, J=None, r=None, it=0, ts=0)
:canonical: stonedfenicsx.solver_module.solver.SolverStokes.set_direct_solver

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes.set_direct_solver
```

````

````{py:method} set_iterative_solver(a, a_p, L, COMM, nl, bcs, F0, F1, ctrl, J=None, r=None, it=0, ts=0)
:canonical: stonedfenicsx.solver_module.solver.SolverStokes.set_iterative_solver

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes.set_iterative_solver
```

````

````{py:method} set_block_operator(a, a_p, bcs, L, F0, F1)
:canonical: stonedfenicsx.solver_module.solver.SolverStokes.set_block_operator

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes.set_block_operator
```

````

````{py:method} update_block_operator(a, a_p, bcs, L, F0, F1)
:canonical: stonedfenicsx.solver_module.solver.SolverStokes.update_block_operator

```{autodoc2-docstring} stonedfenicsx.solver_module.solver.SolverStokes.update_block_operator
```

````

`````
