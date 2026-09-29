# {py:mod}`stonedfenicsx.config.geometry`

```{py:module} stonedfenicsx.config.geometry
```

```{autodoc2-docstring} stonedfenicsx.config.geometry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Domain <stonedfenicsx.config.geometry.Domain>`
  - ```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain
    :summary:
    ```
* - {py:obj}`GeomInput <stonedfenicsx.config.geometry.GeomInput>`
  - ```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput
    :summary:
    ```
* - {py:obj}`Mesh <stonedfenicsx.config.geometry.Mesh>`
  - ```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_ELEMENT_P <stonedfenicsx.config.geometry._ELEMENT_P>`
  - ```{autodoc2-docstring} stonedfenicsx.config.geometry._ELEMENT_P
    :summary:
    ```
* - {py:obj}`_ELEMENT_PT <stonedfenicsx.config.geometry._ELEMENT_PT>`
  - ```{autodoc2-docstring} stonedfenicsx.config.geometry._ELEMENT_PT
    :summary:
    ```
* - {py:obj}`_ELEMENT_V <stonedfenicsx.config.geometry._ELEMENT_V>`
  - ```{autodoc2-docstring} stonedfenicsx.config.geometry._ELEMENT_V
    :summary:
    ```
````

### API

````{py:data} _ELEMENT_P
:canonical: stonedfenicsx.config.geometry._ELEMENT_P
:value: >
   'element(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry._ELEMENT_P
```

````

````{py:data} _ELEMENT_PT
:canonical: stonedfenicsx.config.geometry._ELEMENT_PT
:value: >
   'element(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry._ELEMENT_PT
```

````

````{py:data} _ELEMENT_V
:canonical: stonedfenicsx.config.geometry._ELEMENT_V
:value: >
   'element(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry._ELEMENT_V
```

````

`````{py:class} Domain
:canonical: stonedfenicsx.config.geometry.Domain

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain
```

````{py:attribute} hierarchy
:canonical: stonedfenicsx.config.geometry.Domain.hierarchy
:type: str
:value: >
   'Parent'

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.hierarchy
```

````

````{py:attribute} mesh
:canonical: stonedfenicsx.config.geometry.Domain.mesh
:type: dolfinx.mesh.Mesh
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.mesh
```

````

````{py:attribute} cell_par
:canonical: stonedfenicsx.config.geometry.Domain.cell_par
:type: numpy.typing.NDArray[numpy.int32]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.cell_par
```

````

````{py:attribute} node_par
:canonical: stonedfenicsx.config.geometry.Domain.node_par
:type: numpy.typing.NDArray[numpy.int32]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.node_par
```

````

````{py:attribute} facets
:canonical: stonedfenicsx.config.geometry.Domain.facets
:type: dolfinx.mesh.MeshTags
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.facets
```

````

````{py:attribute} tagcells
:canonical: stonedfenicsx.config.geometry.Domain.tagcells
:type: dolfinx.mesh.MeshTags
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.tagcells
```

````

````{py:attribute} bc_dict
:canonical: stonedfenicsx.config.geometry.Domain.bc_dict
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.bc_dict
```

````

````{py:attribute} solph
:canonical: stonedfenicsx.config.geometry.Domain.solph
:type: dolfinx.fem.FunctionSpace
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.solph
```

````

````{py:attribute} phase
:canonical: stonedfenicsx.config.geometry.Domain.phase
:type: dolfinx.fem.Function
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.phase
```

````

````{py:attribute} comm
:canonical: stonedfenicsx.config.geometry.Domain.comm
:type: mpi4py.MPI.Intracomm
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.comm
```

````

````{py:attribute} name
:canonical: stonedfenicsx.config.geometry.Domain.name
:type: str
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Domain.name
```

````

`````

`````{py:class} GeomInput
:canonical: stonedfenicsx.config.geometry.GeomInput

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput
```

````{py:attribute} x
:canonical: stonedfenicsx.config.geometry.GeomInput.x
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.x
```

````

````{py:attribute} y
:canonical: stonedfenicsx.config.geometry.GeomInput.y
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.y
```

````

````{py:attribute} redo_mesh
:canonical: stonedfenicsx.config.geometry.GeomInput.redo_mesh
:type: bool
:value: >
   True

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.redo_mesh
```

````

````{py:attribute} slab_tk
:canonical: stonedfenicsx.config.geometry.GeomInput.slab_tk
:type: float
:value: >
   130.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.slab_tk
```

````

````{py:attribute} cr
:canonical: stonedfenicsx.config.geometry.GeomInput.cr
:type: float
:value: >
   30.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.cr
```

````

````{py:attribute} ocr
:canonical: stonedfenicsx.config.geometry.GeomInput.ocr
:type: float
:value: >
   7.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.ocr
```

````

````{py:attribute} lit_mt
:canonical: stonedfenicsx.config.geometry.GeomInput.lit_mt
:type: float
:value: >
   20.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.lit_mt
```

````

````{py:attribute} lc
:canonical: stonedfenicsx.config.geometry.GeomInput.lc
:type: float
:value: >
   0.3

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.lc
```

````

````{py:attribute} ns_depth
:canonical: stonedfenicsx.config.geometry.GeomInput.ns_depth
:type: float
:value: >
   50.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.ns_depth
```

````

````{py:attribute} decoupling
:canonical: stonedfenicsx.config.geometry.GeomInput.decoupling
:type: float
:value: >
   80.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.decoupling
```

````

````{py:attribute} resolution_normal
:canonical: stonedfenicsx.config.geometry.GeomInput.resolution_normal
:type: float
:value: >
   2.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.resolution_normal
```

````

````{py:attribute} resolution_refine
:canonical: stonedfenicsx.config.geometry.GeomInput.resolution_refine
:type: float
:value: >
   2.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.resolution_refine
```

````

````{py:attribute} theta_out_slab
:canonical: stonedfenicsx.config.geometry.GeomInput.theta_out_slab
:type: float
:value: >
   45.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.theta_out_slab
```

````

````{py:attribute} theta_in_slab
:canonical: stonedfenicsx.config.geometry.GeomInput.theta_in_slab
:type: float
:value: >
   10.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.theta_in_slab
```

````

````{py:attribute} transition
:canonical: stonedfenicsx.config.geometry.GeomInput.transition
:type: float
:value: >
   10.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.transition
```

````

````{py:attribute} lab_d
:canonical: stonedfenicsx.config.geometry.GeomInput.lab_d
:type: float
:value: >
   100.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.lab_d
```

````

````{py:attribute} slab_type
:canonical: stonedfenicsx.config.geometry.GeomInput.slab_type
:type: str
:value: >
   'CustomParabolic'

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.slab_type
```

````

````{py:attribute} sub_path
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_path
:type: str
:value: >
   'Not_Defined'

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_path
```

````

````{py:attribute} sub_parabolic_a
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_parabolic_a
:type: float
:value: >
   0.0005

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_parabolic_a
```

````

````{py:attribute} sub_lb
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_lb
:type: float
:value: >
   300.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_lb
```

````

````{py:attribute} sub_constant_flag
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_constant_flag
:type: bool
:value: >
   False

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_constant_flag
```

````

````{py:attribute} sub_theta0
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_theta0
:type: float
:value: >
   5.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_theta0
```

````

````{py:attribute} sub_theta_max
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_theta_max
:type: float
:value: >
   45.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_theta_max
```

````

````{py:attribute} sub_trench
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_trench
:type: float
:value: >
   0.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_trench
```

````

````{py:attribute} sub_dl
:canonical: stonedfenicsx.config.geometry.GeomInput.sub_dl
:type: float
:value: >
   1.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.sub_dl
```

````

````{py:attribute} wz_tk
:canonical: stonedfenicsx.config.geometry.GeomInput.wz_tk
:type: float
:value: >
   2.0

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.wz_tk
```

````

````{py:attribute} van_keken
:canonical: stonedfenicsx.config.geometry.GeomInput.van_keken
:type: bool
:value: >
   True

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.van_keken
```

````

````{py:attribute} model_full
:canonical: stonedfenicsx.config.geometry.GeomInput.model_full
:type: bool
:value: >
   False

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.model_full
```

````

````{py:method} check_class_consistency()
:canonical: stonedfenicsx.config.geometry.GeomInput.check_class_consistency

```{autodoc2-docstring} stonedfenicsx.config.geometry.GeomInput.check_class_consistency
```

````

`````

`````{py:class} Mesh
:canonical: stonedfenicsx.config.geometry.Mesh

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh
```

````{py:attribute} g_input
:canonical: stonedfenicsx.config.geometry.Mesh.g_input
:type: stonedfenicsx.config.geometry.GeomInput
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.g_input
```

````

````{py:attribute} global_domain
:canonical: stonedfenicsx.config.geometry.Mesh.global_domain
:type: stonedfenicsx.config.geometry.Domain
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.global_domain
```

````

````{py:attribute} subduction_plate_domain
:canonical: stonedfenicsx.config.geometry.Mesh.subduction_plate_domain
:type: stonedfenicsx.config.geometry.Domain
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.subduction_plate_domain
```

````

````{py:attribute} wedge_domain
:canonical: stonedfenicsx.config.geometry.Mesh.wedge_domain
:type: stonedfenicsx.config.geometry.Domain
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.wedge_domain
```

````

````{py:attribute} crust_domain
:canonical: stonedfenicsx.config.geometry.Mesh.crust_domain
:type: stonedfenicsx.config.geometry.Domain
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.crust_domain
```

````

````{py:attribute} comm
:canonical: stonedfenicsx.config.geometry.Mesh.comm
:type: mpi4py.MPI.Intracomm
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.comm
```

````

````{py:attribute} rank
:canonical: stonedfenicsx.config.geometry.Mesh.rank
:type: int
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.rank
```

````

````{py:attribute} element_p
:canonical: stonedfenicsx.config.geometry.Mesh.element_p
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.element_p
```

````

````{py:attribute} element_pt
:canonical: stonedfenicsx.config.geometry.Mesh.element_pt
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.element_pt
```

````

````{py:attribute} element_v
:canonical: stonedfenicsx.config.geometry.Mesh.element_v
:value: >
   None

```{autodoc2-docstring} stonedfenicsx.config.geometry.Mesh.element_v
```

````

`````
