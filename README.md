# StonedFEniCSx

A FEniCSx (dolfinx)-based FEM package for simulating the thermal and mechanical evolution of a 2D subduction zone: coupled steady-state/time-dependent thermal, Stokes (velocity–pressure), and lithostatic pressure problems on wedge, slab, and global sub-domains, with temperature- and pressure-dependent rheology and shear heating.

The project started as a Python script built on the [FieldStone](https://cedricthieulot.net/fieldstone.html) educational framework and has since grown into a structured, class-based FEM package.

> Research code developed at the University of Leeds. Solo-maintained; interfaces may still change between branches.

## Physical model

Each outer (Picard) iteration couples three problems over the subduction domain (surface to 600 km depth, split into slab, mantle wedge, and overriding-plate sub-domains — see `docs/Computational_domain.md`):

- **Thermal** — the energy equation (diffusion + SUPG-stabilised advection), optionally including radiogenic heating and frictional (shear) heating along the slab interface.
- **Lithostatic pressure** — a global hydrostatic/lithostatic pressure integration that feeds temperature- and pressure-dependent material properties into the other two problems.
- **Stokes (wedge / slab)** — velocity–pressure momentum + continuity, with kinematic (moving-wall) slab boundary conditions and a Nitsche free-slip condition on the slab's basal interface.

Material properties (density, heat capacity, thermal conductivity, dislocation/diffusion-creep and plastic rheology) are evaluated once per Picard iteration and held "frozen" for that iteration's linear solve. Both steady-state and time-dependent solves are supported; the steady-state path is the one currently validated against benchmarks (see below).

## Requirements

- Python >= 3.10, < 3.14
- [dolfinx](https://github.com/FEniCS/dolfinx) 0.9.x, with PETSc and MPI
- mpi4py, petsc4py
- numpy, scipy, gmsh, meshio, numba, h5py, shapely, pandas, pyyaml

Pinned versions are tracked in `stoned_environment.yml` (conda) and `pyproject.toml`/`pyproject_HPC.toml` (pip).

## Installation

### Local, with conda

```bash
conda env create -f stoned_environment.yml
conda activate stoned_fenicsx
pip install --no-deps -e .
```

### HPC, with a Spack-provided dolfinx

On a cluster, dolfinx/PETSc/MPI are usually best obtained from the module system rather than conda. `pyproject_HPC.toml` targets this setup (dolfinx supplied externally, `pip install -e .` into a venv layered on top of loaded Spack packages). See [`HPC_read.md`](HPC_read.md) for a full worked example on the University of Leeds Aire cluster, including gotchas around gmsh's OSMesa/GLU dependency and h5py/numpy ABI compatibility with the cluster's HDF5.

## Quick start

A simulation is configured with two YAML-parsed inputs — numerical/I-O/thermal/kinematic controls, and per-phase material properties — which drive `stonedfenicsx.stoned_fenicsx`:

```python
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

input_data, ph_in = parse_input("input.yaml")
stoned_fenicsx(input_data, ph_in)
```

`input.yaml` at the repo root is a commented example covering units, numerical controls, shear-heating options, and thermal/kinematic boundary conditions. `stonedfenicsx/stoned_fenicsx.py::test_function` shows a fully scripted example that also overrides material properties in code after parsing. `examples/` contains region-specific driver scripts (`Japan_slab.py`, `Mexico_slab.py`, `Chile_slab.py`, `Tonga_slab copy.py`) built the same way.

Results are written under `Results/<test_name>/` as XDMF/HDF5 fields, plus cached material-property lookups.

Simulations are MPI-parallel; run under `mpirun`/`srun` for multi-rank execution (see the example SLURM script in `HPC_read.md`).

## Running the tests

```bash
pytest tests/
```

The main physical validation is `tests/test_benchmark_shear_heating.py` and `tests/test_benchmark_vankeken.py`, which reproduce reference results from the Van Keken et al. subduction zone benchmark suite (`tests/VanKeken/`) across viscosity/thermal configurations and, for the shear-heating case, several friction-angle (`phi`) values.

## Package layout

```
stonedfenicsx/
├── config/            # Input parsing, non-dimensionalisation (Scal), phase/material database,
│                       # numerical controls, geometry, thermal/kinematic boundary conditions
├── create_mesh/        # gmsh-based mesh generation and slab-surface geometry handling
├── material_property/  # Temperature/pressure-dependent density, heat capacity, conductivity,
│                       # rheology (dislocation/diffusion creep, plasticity)
├── solver_module/       # Problem classes (Global_thermal, Global_pressure, Wedge, Slab),
│                       # PETSc solver wrappers (ScalarSolver, SolverStokes), and the
│                       # outer Picard/time-stepping loop (solution_routine)
├── output.py           # XDMF/HDF5 result writing
└── stoned_fenicsx.py    # Top-level entry point: configure_simulation() then solution_routine()
```

## Documentation

- [`HPC_read.md`](HPC_read.md) — HPC (Aire/Spack) installation notes.
- `docs/` — Sphinx documentation (installation, computational domain, material properties); build with `sphinx-build docs docs/build`.
- `stonedfenicsx/config/thermal_boundary_flow.md` — call-flow diagram for the thermal boundary condition module.

## Status

Actively developed on branch `Remove_redundancy` (cleanup following the `Harmonise_zen_branch` refactor to a class-based solver API). Steady-state thermal + Stokes + lithostatic pressure coupling, including shear heating, is validated against the Van Keken et al. benchmark suite. The time-dependent solve path is functional but less exercised by the current test suite than the steady-state path.

## License

Not yet specified.
