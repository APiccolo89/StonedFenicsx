<!-- StonedFenicsx documentation master file -->

# StonedFEniCSx

Documentation of the FEM package **StonedFEniCSx**.
The purpose of the library is to compute the thermal evolution of a subducting plate using the FEniCSx library.  
The project initially started as a Python script (using FieldStone educational framework [xFieldstone](https://github.com/irisvanzelst/xFieldstone)) and then evolved into a more structured FEM project. A few of the older routines from  [xFieldstone](https://github.com/irisvanzelst/xFieldstone) have been rewritten and updated. 

The main repository page is [StonedFEniCSx](https://github.com/APiccolo89/StonedFenicsx)

The code is underdevelopment and the documentation will be modified in the following weeks. 


The package is organised in six modules: 
```
Stonedfenics/
├── config/                  — configure the simulations
├── create_mesh/             — creates the mesh via gmsh and meshio
├── material_properties/     — computes material properties
├── solver_module/           — solution routines
├── stoned_fenicsx.py        — top-level entry point
├── output.py                — handles output printing
└── utils.py                 — general-purpose functions (timing, print_ph, ...)
```
The intended workflow is to use the configuration module to configure the simulation and then use the top-level function **stonedfenicsx** to run the simulation. The code has a specific purpose: the production of subduction kinematic model. However, the code has been designed to be adapted for other problems, especially the routines that configure the simulation and handle the material properties. 

The code has benefitted of several additional sources: 

- [FEniCS-SZ](https://cianwilson.github.io/fenics-sz/notebooks/0_index.html)- for the overall organisation of the problems - 
- [FEniCS-discourse](https://fenicsproject.discourse.group/) - especially the tutorials and the discussion concerning how to set solvers, how to optimise the routines 
- [FieldStone](https://cedrict.github.io/) - a useful and pedagogical introduction to finite element - 

The documentation is organized in a such way that the code's component are connected to the relative method. It is a mix of scientific-technical documentation. The guide will introduce a few examples that will be part of manuscript in preparation. The package will link to the repository of these experiments to reproduce the results. 


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

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
how_to_use
Computational_domain
Material_properties
Examples
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`