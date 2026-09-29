# {py:mod}`stonedfenicsx.utils`

```{py:module} stonedfenicsx.utils
```

```{autodoc2-docstring} stonedfenicsx.utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`check_race_condition <stonedfenicsx.utils.check_race_condition>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.check_race_condition
    :summary:
    ```
* - {py:obj}`timing_function <stonedfenicsx.utils.timing_function>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.timing_function
    :summary:
    ```
* - {py:obj}`time_the_time <stonedfenicsx.utils.time_the_time>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.time_the_time
    :summary:
    ```
* - {py:obj}`print_ph <stonedfenicsx.utils.print_ph>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.print_ph
    :summary:
    ```
* - {py:obj}`interpolate_from_sub_to_main <stonedfenicsx.utils.interpolate_from_sub_to_main>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.interpolate_from_sub_to_main
    :summary:
    ```
* - {py:obj}`gather_vector <stonedfenicsx.utils.gather_vector>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.gather_vector
    :summary:
    ```
* - {py:obj}`gather_coordinates <stonedfenicsx.utils.gather_coordinates>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.gather_coordinates
    :summary:
    ```
* - {py:obj}`compute_strain_rate <stonedfenicsx.utils.compute_strain_rate>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.compute_strain_rate
    :summary:
    ```
* - {py:obj}`compute_eii <stonedfenicsx.utils.compute_eii>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.compute_eii
    :summary:
    ```
* - {py:obj}`evaluate_material_property <stonedfenicsx.utils.evaluate_material_property>`
  - ```{autodoc2-docstring} stonedfenicsx.utils.evaluate_material_property
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DEBUG_ <stonedfenicsx.utils._DEBUG_>`
  - ```{autodoc2-docstring} stonedfenicsx.utils._DEBUG_
    :summary:
    ```
````

### API

````{py:data} _DEBUG_
:canonical: stonedfenicsx.utils._DEBUG_
:value: >
   False

```{autodoc2-docstring} stonedfenicsx.utils._DEBUG_
```

````

````{py:function} check_race_condition(ioctrl: stonedfenicsx.config.numerical_control.IOControls, file: str) -> bool
:canonical: stonedfenicsx.utils.check_race_condition

```{autodoc2-docstring} stonedfenicsx.utils.check_race_condition
```
````

````{py:function} timing_function(fun: collections.abc.Callable) -> collections.abc.Callable
:canonical: stonedfenicsx.utils.timing_function

```{autodoc2-docstring} stonedfenicsx.utils.timing_function
```
````

````{py:function} time_the_time(delta_time: float) -> float
:canonical: stonedfenicsx.utils.time_the_time

```{autodoc2-docstring} stonedfenicsx.utils.time_the_time
```
````

````{py:function} print_ph(string: str) -> int
:canonical: stonedfenicsx.utils.print_ph

```{autodoc2-docstring} stonedfenicsx.utils.print_ph
```
````

````{py:function} interpolate_from_sub_to_main(u_dest: dolfinx.fem.Function, u_start: dolfinx.fem.Function, cells: numpy.ndarray, parent2child: int = 0) -> None
:canonical: stonedfenicsx.utils.interpolate_from_sub_to_main

```{autodoc2-docstring} stonedfenicsx.utils.interpolate_from_sub_to_main
```
````

````{py:function} gather_vector(v)
:canonical: stonedfenicsx.utils.gather_vector

```{autodoc2-docstring} stonedfenicsx.utils.gather_vector
```
````

````{py:function} gather_coordinates(V)
:canonical: stonedfenicsx.utils.gather_coordinates

```{autodoc2-docstring} stonedfenicsx.utils.gather_coordinates
```
````

````{py:function} compute_strain_rate(u)
:canonical: stonedfenicsx.utils.compute_strain_rate

```{autodoc2-docstring} stonedfenicsx.utils.compute_strain_rate
```
````

````{py:function} compute_eii(e)
:canonical: stonedfenicsx.utils.compute_eii

```{autodoc2-docstring} stonedfenicsx.utils.compute_eii
```
````

````{py:function} evaluate_material_property(expression: dolfinx.fem.Expression, function_space: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Function
:canonical: stonedfenicsx.utils.evaluate_material_property

```{autodoc2-docstring} stonedfenicsx.utils.evaluate_material_property
```
````
