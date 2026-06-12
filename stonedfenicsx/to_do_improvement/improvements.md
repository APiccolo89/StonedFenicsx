# Pending Improvements

## 1. Fix: `path_test` in `input.yaml`
`InputOutputControl.path_test` should be `path_save`. `path_test` is a computed
field set by `generate_io()` as `path_save / test_name`. Setting it directly from
the YAML leaves `path_save = ""` and silently breaks directory creation.

## 2. Fix: `dt` unit inconsistency
`NumericalControls.dt` defaults to `500.0` (implicitly yr) but the YAML sets
`dt: 0.015 # Myr`. Decide on one unit and enforce it consistently across the
default, the YAML comment, and the class docstring.

## 3. Fix: `filling_the_phase_data_base` bypasses `update_ip_file`
The phase-filling loop uses raw `setattr` directly, skipping the unknown-key guard
and type casting in `update_ip_file`. Unknown YAML keys in material sections
silently succeed. Route through `update_ip_file` or add the same guard explicitly.

## 4. Design: Auto-correct + warn for `slab_age` / `interval_val[0]`
Rationale: most simulations are constant velocity; enforcing `interval_val[0]` as
the primary field creates poor UX for the common case. Users are distracted and
will forget to align the fields.

Replace the hard `ValueError` in `update_thermal_bc` (and the equivalent check in
`check_kinematic_bc` for `v_s[0]` / `interval_val[0]`) with a logged warning that
auto-corrects:

```python
if self.interval_val[0] != self.slab_age:
    logger.warning(
        "interval_val[0]=%.1f != slab_age=%.1f — correcting interval_val[0]",
        self.interval_val[0], self.slab_age,
    )
    self.interval_val[0] = self.slab_age
```

Same pattern for `CtrlKy`: if `v_s[0] != interval_val[0]`, correct and warn.

## 5. Design: Replace `print_ph` with structured logging

### Problem
Currently the codebase mixes `print`, `print_ph`, and bare `print('Warning: ...')`
strings. `print_ph` is MPI-aware (rank 0 only) but provides no log levels, no file
output, and no way for users to suppress noise during ensemble runs.

### Solution
Add `stonedfenicsx/config/log.py`:

```python
import logging
from mpi4py import MPI

class _MPIRank0Filter(logging.Filter):
    def filter(self, record):
        return MPI.COMM_WORLD.rank == 0

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """Call once from the user's run script."""
    logger = logging.getLogger("stonedfenicsx")
    logger.setLevel(level)
    fmt = logging.Formatter("%(levelname)-8s | %(name)s | %(message)s")
    mpi_filter = _MPIRank0Filter()

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.addFilter(mpi_filter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        fh.addFilter(mpi_filter)
        logger.addHandler(fh)
```

In every module, top of file:
```python
from stonedfenicsx.config.log import get_logger
logger = get_logger(__name__)
```

Migration:
- `print_ph("...")` → `logger.info("...")`
- `print('Warning: ...')` → `logger.warning("...")`
- `timing_function` bare `print` → `logger.info`

User run script:
```python
from stonedfenicsx.config.log import setup_logging
setup_logging(level=logging.INFO, log_file="run_001.log")
```

For ensemble runs: pass `logging.WARNING` to suppress progress noise.
For debugging: pass `logging.DEBUG`.

### Rules
- The package never configures the root logger — only `logging.getLogger("stonedfenicsx")`.
- Use `logger.warning()` not `warnings.warn()` — MPI-safe via the filter.
- `timing_function` in `utils.py` routes through `logger.info` instead of bare `print`.

## 6. Cleanup: Minor items
- `update_thermal_bc` docstring: replace `_summary_` placeholder.
- `adiabatic_heating: int = 1 # REMOVE (?)`: make a decision and act on it.
- `cast_type` signature: `any` → `Any` (from `typing`).
- `SimulationControls` docstring: typo `Boudary` → `Boundary`.
- `int` flags (`steady_state`, `decoupling_ctrl`, `pressure_dependency`): consider
  `bool` — `cast_type` already handles the coercion, and `bool` rejects `99` or `-1`.

## 7. Next: Implement `configure_simulation`
The stub in `Stoned_fenicx.py` needs to become the real translation boundary:
- Scale parameters via `sc`
- Call `ctrl_tbc.update_thermal_bc(g_input, ctrl)`
- Build mesh
- Assemble `SimulationControls` from the flat `Input`
- Return `(SimulationControls, PhaseDataBase, Mesh)`
