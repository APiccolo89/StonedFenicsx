# {py:mod}`stonedfenicsx.create_mesh.create_mesh`

```{py:module} stonedfenicsx.create_mesh.create_mesh
```

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_differs <stonedfenicsx.create_mesh.create_mesh._differs>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh._differs
    :summary:
    ```
* - {py:obj}`compare_data <stonedfenicsx.create_mesh.create_mesh.compare_data>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.compare_data
    :summary:
    ```
* - {py:obj}`write_mesh_data <stonedfenicsx.create_mesh.create_mesh.write_mesh_data>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.write_mesh_data
    :summary:
    ```
* - {py:obj}`create_mesh <stonedfenicsx.create_mesh.create_mesh.create_mesh>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_mesh
    :summary:
    ```
* - {py:obj}`create_gmesh <stonedfenicsx.create_mesh.create_mesh.create_gmesh>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_gmesh
    :summary:
    ```
* - {py:obj}`create_domain_subduction_plate <stonedfenicsx.create_mesh.create_mesh.create_domain_subduction_plate>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_domain_subduction_plate
    :summary:
    ```
* - {py:obj}`create_domain_wedge <stonedfenicsx.create_mesh.create_mesh.create_domain_wedge>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_domain_wedge
    :summary:
    ```
* - {py:obj}`create_domain_crust <stonedfenicsx.create_mesh.create_mesh.create_domain_crust>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_domain_crust
    :summary:
    ```
* - {py:obj}`create_physical_line <stonedfenicsx.create_mesh.create_mesh.create_physical_line>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_physical_line
    :summary:
    ```
* - {py:obj}`create_gmsh <stonedfenicsx.create_mesh.create_mesh.create_gmsh>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_gmsh
    :summary:
    ```
* - {py:obj}`create_mesh_fenicsx <stonedfenicsx.create_mesh.create_mesh.create_mesh_fenicsx>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_mesh_fenicsx
    :summary:
    ```
* - {py:obj}`extract_facet_boundary <stonedfenicsx.create_mesh.create_mesh.extract_facet_boundary>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.extract_facet_boundary
    :summary:
    ```
* - {py:obj}`create_subdomain <stonedfenicsx.create_mesh.create_mesh.create_subdomain>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_subdomain
    :summary:
    ```
* - {py:obj}`read_mesh <stonedfenicsx.create_mesh.create_mesh.read_mesh>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.read_mesh
    :summary:
    ```
* - {py:obj}`create_mesh_object <stonedfenicsx.create_mesh.create_mesh.create_mesh_object>`
  - ```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_mesh_object
    :summary:
    ```
````

### API

````{py:function} _differs(cached, current) -> bool
:canonical: stonedfenicsx.create_mesh.create_mesh._differs

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh._differs
```
````

````{py:function} compare_data(g_input: stonedfenicsx.config.geometry.GeomInput, ioctrl: stonedfenicsx.config.numerical_control.IOControls) -> None
:canonical: stonedfenicsx.create_mesh.create_mesh.compare_data

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.compare_data
```
````

````{py:function} write_mesh_data(g_input: stonedfenicsx.config.geometry.GeomInput, ioctrl: stonedfenicsx.config.numerical_control.IOControls) -> None
:canonical: stonedfenicsx.create_mesh.create_mesh.write_mesh_data

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.write_mesh_data
```
````

````{py:function} create_mesh(ioctrl: stonedfenicsx.config.numerical_control.IOControls, g_input: stonedfenicsx.config.geometry.GeomInput, ctrl: stonedfenicsx.config.numerical_control.NumericalControls) -> stonedfenicsx.config.geometry.Mesh
:canonical: stonedfenicsx.create_mesh.create_mesh.create_mesh

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_mesh
```
````

````{py:function} create_gmesh(ioctrl: stonedfenicsx.config.numerical_control.IOControls, g_input: stonedfenicsx.config.geometry.GeomInput)
:canonical: stonedfenicsx.create_mesh.create_mesh.create_gmesh

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_gmesh
```
````

````{py:function} create_domain_subduction_plate(mesh_model: gmsh.model, CP: stonedfenicsx.create_mesh.aux_create_mesh.Class_Points, LC: stonedfenicsx.create_mesh.aux_create_mesh.Class_Line, g_input: stonedfenicsx.config.geometry.GeomInput) -> gmsh.model
:canonical: stonedfenicsx.create_mesh.create_mesh.create_domain_subduction_plate

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_domain_subduction_plate
```
````

````{py:function} create_domain_wedge(mesh_model, CP, LC, g_input)
:canonical: stonedfenicsx.create_mesh.create_mesh.create_domain_wedge

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_domain_wedge
```
````

````{py:function} create_domain_crust(mesh_model, CP, LC, g_input)
:canonical: stonedfenicsx.create_mesh.create_mesh.create_domain_crust

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_domain_crust
```
````

````{py:function} create_physical_line(CP: stonedfenicsx.create_mesh.aux_create_mesh.Class_Points, LC: stonedfenicsx.create_mesh.aux_create_mesh.Class_Line, g_input: stonedfenicsx.config.geometry.GeomInput, mesh_model: gmsh.model) -> gmsh.model
:canonical: stonedfenicsx.create_mesh.create_mesh.create_physical_line

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_physical_line
```
````

````{py:function} create_gmsh(sx: numpy.ndarray, sy: numpy.ndarray, bsx: numpy.ndarray, bsy: numpy.ndarray, oc_cx: numpy.ndarray, oc_cy: numpy.ndarray, g_input: stonedfenicsx.config.geometry.GeomInput) -> gmsh.model
:canonical: stonedfenicsx.create_mesh.create_mesh.create_gmsh

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_gmsh
```
````

````{py:function} create_mesh_fenicsx(mesh: meshio._mesh.Mesh, cell_type: str, prune_z: bool = False) -> dolfinx.mesh.Mesh
:canonical: stonedfenicsx.create_mesh.create_mesh.create_mesh_fenicsx

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_mesh_fenicsx
```
````

````{py:function} extract_facet_boundary(Mesh: dolfinx.mesh.Mesh, Mfacet_tag: dolfinx.mesh.MeshTags, submesh: dolfinx.mesh.Mesh, sm_vertex_maps: numpy.ndarray, boundary: list, m_id: int) -> tuple([np.ndarray, np.ndarray])
:canonical: stonedfenicsx.create_mesh.create_mesh.extract_facet_boundary

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.extract_facet_boundary
```
````

````{py:function} create_subdomain(mesh: dolfinx.mesh.Mesh, mesh_tag: dolfinx.mesh.MeshTags, facet_tag: dolfinx.mesh.MeshTags, phase_set: list, name: str, phase: dolfinx.fem.function.Function, ioctrl: stonedfenicsx.config.numerical_control.IOControls) -> stonedfenicsx.config.geometry.Domain
:canonical: stonedfenicsx.create_mesh.create_mesh.create_subdomain

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_subdomain
```
````

````{py:function} read_mesh(ioctrl: stonedfenicsx.config.numerical_control.IOControls, redo_mesh: bool) -> tuple([dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags])
:canonical: stonedfenicsx.create_mesh.create_mesh.read_mesh

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.read_mesh
```
````

````{py:function} create_mesh_object(ioctrl: stonedfenicsx.config.numerical_control.IOControls, g_input: stonedfenicsx.config.geometry.GeomInput) -> stonedfenicsx.config.geometry.Mesh
:canonical: stonedfenicsx.create_mesh.create_mesh.create_mesh_object

```{autodoc2-docstring} stonedfenicsx.create_mesh.create_mesh.create_mesh_object
```
````
