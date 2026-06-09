# Config module — code review

**Date:** 2026-06-09  
**Branch:** `Refractoring_towards_a_better_version`  
**Scope:** `stonedfenicsx/config/`

---

## Bugs (will fail at runtime)

### `geometry.py:104` — wrong attribute name in `__post_init__`

```python
# current (broken)
raise ValueError(f'sub_type must be "Custom" or "Real", got {self.sub_type!r}')
# fix
raise ValueError(f'slab_type must be "Custom" or "Real", got {self.slab_type!r}')
```

`self.sub_type` does not exist; the field is `slab_type`. This raises `AttributeError` on every `GeomInput()` construction.

---

### `scal.py` — attribute names don't match `Scal` fields

`Scal` stores fields with these names:

| Correct field | Wrong name used in scaling functions |
|---|---|
| `temp` | `Temp` |
| `length` | `L` |
| `time` | `T` |
| `mass` | `M` |
| `energy` | `Energy` |
| `watt` | `Watt` |
| `cp` | `Cp` |
| `scale_myr2sec` | `scale_Myr2sec` |

All four standalone functions — `_scaling_material_properties`, `scale_parameters`, `scaling_control_parameters`, `scal_time_class` — use the wrong names and will raise `AttributeError` at runtime.

---

### `scal.py:183` — two typos in `dimensionless_ginput`

```python
# line 183 — typo
g_input.resolution_normal /= sc.lenght   # should be sc.length
# line 185 — wrong attribute name
g_input.trans /= sc.length               # should be g_input.transition
```

---

### `input_parser.py:335` — wrong variable in `check_bool`

```python
# current (broken): uses outer `v` instead of local `vbuf`
if isinstance(vbuf, int) or isinstance(v, float):
# fix
if isinstance(vbuf, int) or isinstance(vbuf, float):
```

This silently reads the outer scope variable and converts the wrong value.

---

### `right_boundary.py` — file is broken / incomplete

Multiple issues make this file non-executable:

- `Assembly_matrix` / `Assembly_vector` (capitalised) are called but only `assembly_matrix` (lowercase) is defined, and `Assembly_vector` doesn't exist at all.
- `phases` is used in `solve_1D_steady_state_diffusion` but never defined.
- `res` is tested in a `while` loop before being assigned.
- Line 200 is an incomplete assignment: `Pdb =` (syntax error).
- Imports use old module paths: `stonedfenicsx.numerical_control`, `stonedfenicsx.scal`.

**Recommended action:** move to a feature branch or remove until the implementation is ready.

---

### `thermal_structure_ocean.py` — two runtime errors

1. **Line 45** — `start = timing.time()` runs at import time; `timing` is never defined (probably meant `import time` → `time.time()`). The module fails on import.
2. **`update_age_lhs` (line 679)** — references `t` and `temperature` which are local variables inside `compute_ocean_plate_temperature`. They don't exist in this function's scope, causing `NameError`.

---

## Type annotation errors

| Location | Current | Should be |
|---|---|---|
| `input_parser.py:237` | `-> int` | `-> tuple[Input, PhInput]` |
| `input_parser.py:308` | `v: any, tp: any` | `v: Any, tp: Any` (from `typing`) |
| `numerical_control.py:82` | `ndarray[np.int32]`, default `1` | `int`, or give it a proper array default |
| `numerical_control.py:98-99` | `float`, default `None` | `float \| None` |
| `numerical_control.py:100-103` | typed `float`, default `np.array(...)` | `ndarray[np.float64]` |
| `numerical_control.py:45` | typed `Path`, default `'Cached_information'` | default should be `Path('Cached_information')` |

---

## Design / structural issues

### `PhInput` phases are `field(init=False)` with no `__post_init__`

```python
subducting_plate_mantle: Phase = field(init=False)
oceanic_crust: Phase = field(init=False)
# ... etc
```

Accessing any of these before `filling_the_phase_data_base` is called raises `AttributeError`. Either assign `Phase()` as the default, or use `Optional[Phase] = None` to make the unset state explicit.

---

### `dict_phase_id` recreated on every call

In `filling_the_phase_data_base` (`input_parser.py:375`), the `dict_phase_id` mapping is rebuilt every invocation. It is a pure constant; place it at module level alongside `dict_options` and `dict_stokes`.

---

### `Phase.id` shadows the builtin `id`

Rename to `phase_id` to avoid confusion and linter warnings.

---

### `time_dependent_evolution` violates PEP 8

Class names must be `UpperCamelCase`. Rename to `TimeDependentEvolution`. The old name can be kept as an alias (`TimeDependentEvolution = time_dependent_evolution`) temporarily if other modules reference it.

---

### `scal.py` stale imports

```python
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input   # old path & old name
from stonedfenicsx.numerical_control import time_dependent_evolution  # old path
```

Both should now be imported from `stonedfenicsx.config`.

---

### Wildcard imports in `right_boundary.py` and `thermal_structure_ocean.py`

```python
from stonedfenicsx.package_import import *
```

This hides all dependencies and makes it impossible to know what names are in scope. Replace with explicit imports.

---

## Minor / style

- **Incomplete docstrings** — `cast_type`, `compute_the_derivative_scal`, and `update_ip_file` still have `_summary_` placeholder text.
- **Stale `Mesh` docstring** (`geometry.py:119-133`) — refers to `domainA/B/C/G`; actual field names are `subduction_plate_domain`, `wedge_domain`, `crust_domain`, `global_domain`.
- **Mixed Italian/English inline comments** in `geometry.py` (`# gradi, invariato`, `# adimensionale`). Pick one language for the codebase.
- **`correct_input` style** — prefer `k in (...)` over chained `or`:

  ```python
  # current
  elif k == "stokes_solver_type" or k == "energy_solver_type":
  # preferred
  elif k in ("stokes_solver_type", "energy_solver_type"):
  ```

---

## Priority order

1. `geometry.py:104` `slab_type` bug — breaks every run immediately.
2. `scal.py` attribute name mismatches — breaks all scaling.
3. `thermal_structure_ocean.py` import-time crash (`timing`).
4. `input_parser.py:335` `check_bool` wrong variable.
5. Type annotation fixes — won't crash but will confuse `beartype` and static checkers.
6. `right_boundary.py` — isolate on a branch.
7. Structural issues — no urgency but improve robustness.
