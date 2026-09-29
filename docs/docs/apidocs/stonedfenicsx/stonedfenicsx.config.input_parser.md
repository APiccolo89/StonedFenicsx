# {py:mod}`stonedfenicsx.config.input_parser`

```{py:module} stonedfenicsx.config.input_parser
```

```{autodoc2-docstring} stonedfenicsx.config.input_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Input <stonedfenicsx.config.input_parser.Input>`
  - ```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_input <stonedfenicsx.config.input_parser.parse_input>`
  - ```{autodoc2-docstring} stonedfenicsx.config.input_parser.parse_input
    :summary:
    ```
* - {py:obj}`filling_the_phase_data_base <stonedfenicsx.config.input_parser.filling_the_phase_data_base>`
  - ```{autodoc2-docstring} stonedfenicsx.config.input_parser.filling_the_phase_data_base
    :summary:
    ```
* - {py:obj}`test_function <stonedfenicsx.config.input_parser.test_function>`
  - ```{autodoc2-docstring} stonedfenicsx.config.input_parser.test_function
    :summary:
    ```
````

### API

`````{py:class} Input
:canonical: stonedfenicsx.config.input_parser.Input

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input
```

````{py:attribute} ctrl
:canonical: stonedfenicsx.config.input_parser.Input.ctrl
:type: stonedfenicsx.config.numerical_control.NumericalControls
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input.ctrl
```

````

````{py:attribute} ctrl_io
:canonical: stonedfenicsx.config.input_parser.Input.ctrl_io
:type: stonedfenicsx.config.numerical_control.IOControls
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input.ctrl_io
```

````

````{py:attribute} ctrl_tbc
:canonical: stonedfenicsx.config.input_parser.Input.ctrl_tbc
:type: stonedfenicsx.config.numerical_control.CtrlTemperatureBC
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input.ctrl_tbc
```

````

````{py:attribute} ctrl_ky
:canonical: stonedfenicsx.config.input_parser.Input.ctrl_ky
:type: stonedfenicsx.config.numerical_control.CtrlKy
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input.ctrl_ky
```

````

````{py:attribute} g_input
:canonical: stonedfenicsx.config.input_parser.Input.g_input
:type: stonedfenicsx.config.geometry.GeomInput
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input.g_input
```

````

````{py:attribute} sc
:canonical: stonedfenicsx.config.input_parser.Input.sc
:type: stonedfenicsx.config.scal.Scal
:value: >
   'field(...)'

```{autodoc2-docstring} stonedfenicsx.config.input_parser.Input.sc
```

````

`````

````{py:function} parse_input(path: str) -> tuple[stonedfenicsx.config.input_parser.Input, stonedfenicsx.config.phase_db.PhInput]
:canonical: stonedfenicsx.config.input_parser.parse_input

```{autodoc2-docstring} stonedfenicsx.config.input_parser.parse_input
```
````

````{py:function} filling_the_phase_data_base(materialproperties: dict, shheating: dict, phase_input: stonedfenicsx.config.phase_db.PhInput) -> stonedfenicsx.config.phase_db.PhInput
:canonical: stonedfenicsx.config.input_parser.filling_the_phase_data_base

```{autodoc2-docstring} stonedfenicsx.config.input_parser.filling_the_phase_data_base
```
````

````{py:function} test_function()
:canonical: stonedfenicsx.config.input_parser.test_function

```{autodoc2-docstring} stonedfenicsx.config.input_parser.test_function
```
````
