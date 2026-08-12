<!-- StonedFenicsx documentation master file -->

# StonedFEniCSx documentation

Documentation of the FEM package **StonedFEniCSx**.
The purpose of the library is to compute the thermal evolution of a subducting plate using the FEniCSx library.  
The project initially started as a Python script (using FieldStone educational framework [xFieldstone](https://github.com/irisvanzelst/xFieldstone)) and then evolved into a more structured FEM project. A few of the older routines from  [xFieldstone](https://github.com/irisvanzelst/xFieldstone) have been rewritten and updated. 

The package is organised in six modules: 
```
**stonedfenicsx**/
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




```{toctree}
:maxdepth: 2
:caption: Contents:

installation
how_to_use
Computational_domain
Material_properties
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`