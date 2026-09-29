# Installation and Quick Start
## Installation
### Installation with Conda/Miniconda
```bash
git clone https://github.com/APiccolo89/StonedFenicsx.git # Clone the repository
cd StonedFenicsx # Go to the folder of the repository
conda env create -f stonedenvironment.yml # Create the enviroment ** optional --name personalised_name (choose what do you like)
conda activate stonedfenicsx # or your personalised name
```
>[!TIP]
> Conda can be slow for installing all the dependency. The solution that has been found is the following [conda-lib-mamba]('https://conda.github.io/conda-libmamba-solver/'). So, 
> in the case in which the installation of the depencency tooks ages, you can install it:
> ```bash
>  conda install -n base conda-libmamba-solver
>  conda config --set solver libmamba
>```
> However, consider to read a bit how to install before doing it.

## Quick start

A simulation is configured with two YAML-parsed inputs — numerical/I-O/thermal/kinematic controls, and per-phase material properties — which drive `stonedfenicsx.stoned_fenicsx`:

```python
from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx
input_data, ph_in = parse_input("input.yaml")
stoned_fenicsx(input_data, ph_in)
```

`input.yaml` at the repo root is a commented example covering units, numerical controls, shear-heating options, and thermal/kinematic boundary conditions. `stonedfenicsx/stoned_fenicsx.py::test_function` shows a fully scripted example that also overrides material properties in code after parsing. `examples/data` contains region-specific driver datasets. Soon, there will be simple scripts to run simulations for specific regions.

Results are written under `Results/<test_name>/` as XDMF/HDF5 fields.

Simulations are MPI-parallel; run under `mpirun`/`srun` for multi-rank execution.

## Running the tests

```bash
pytest tests/
```

The main physical validation is and `tests/test_benchmark_vankeken.py`, which reproduce reference results from the Van Keken et al. subduction zone benchmark suite (`tests/VanKeken/`) across viscosity/thermal configurations.