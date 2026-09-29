# {py:mod}`stonedfenicsx.create_mesh.read_slab_surface`

```{py:module} stonedfenicsx.create_mesh.read_slab_surface
```

```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_Savitzky_Golay <stonedfenicsx.create_mesh.read_slab_surface.apply_Savitzky_Golay>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.apply_Savitzky_Golay
    :summary:
    ```
* - {py:obj}`function_bending_data <stonedfenicsx.create_mesh.read_slab_surface.function_bending_data>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.function_bending_data
    :summary:
    ```
* - {py:obj}`curve_fitting <stonedfenicsx.create_mesh.read_slab_surface.curve_fitting>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.curve_fitting
    :summary:
    ```
* - {py:obj}`compute_bending_angle_length <stonedfenicsx.create_mesh.read_slab_surface.compute_bending_angle_length>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.compute_bending_angle_length
    :summary:
    ```
* - {py:obj}`read_file_slab <stonedfenicsx.create_mesh.read_slab_surface.read_file_slab>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.read_file_slab
    :summary:
    ```
````

### API

````{py:function} apply_Savitzky_Golay(yd, order, cf)
:canonical: stonedfenicsx.create_mesh.read_slab_surface.apply_Savitzky_Golay

```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.apply_Savitzky_Golay
```
````

````{py:function} function_bending_data(ell: numpy.typing.NDArray[float], theta: numpy.typing.NDArray[float]) -> float
:canonical: stonedfenicsx.create_mesh.read_slab_surface.function_bending_data

```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.function_bending_data
```
````

````{py:function} curve_fitting(xd: numpy.typing.NDArray[float], yd: numpy.typing.NDArray[float], path: str, name: str, max_depth: float) -> numpy.typing.NDArray[float]
:canonical: stonedfenicsx.create_mesh.read_slab_surface.curve_fitting

```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.curve_fitting
```
````

````{py:function} compute_bending_angle_length(x: numpy.typing.NDArray[float], z: numpy.typing.NDArray[float]) -> tuple[numpy.typing.NDArray[float], numpy.typing.NDArray[float]]
:canonical: stonedfenicsx.create_mesh.read_slab_surface.compute_bending_angle_length

```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.compute_bending_angle_length
```
````

````{py:function} read_file_slab(file_path: str, max_depth: float) -> tuple[numpy.typing.NDArray[float], numpy.typing.NDArray[float]]
:canonical: stonedfenicsx.create_mesh.read_slab_surface.read_file_slab

```{autodoc2-docstring} stonedfenicsx.create_mesh.read_slab_surface.read_file_slab
```
````
