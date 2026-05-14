# Introduction
This short readme is a personal collection of solutions valid for the Aire cluster at the University of Leeds. Its purpose is to provide insights to colleagues who want to use my package in an HPC environment. This is a practical guide to understanding the required steps and the practical problems that anyone might encounter when installing and running FEniCSx-based software in an HPC environment. 

HPC systems and conda are often not a good fit, particularly when working with MPI-dependent scientific libraries such as FEniCSx, PETSc, h5py, and meshio. I wrote this short guide to document how I installed `StonedFenicsx` using Spack on Aire (the University of Leeds HPC cluster).

The main purpose is not to provide a universal installation recipe, but rather a practical template that can be adapted depending on local HPC policies, available modules, and compiler/MPI configurations.

There is an additional complication: gmsh requires a Mesa/GLU library. At the time of writing this guide, this library is only accessible on the login node, not on compute nodes.

---

# Installation Steps

## 1st: Install FEniCSx using Spack

On Aire, I first loaded Spack:

```bash
module load spack/0.23
```

Then installed Dolfinx:

```bash
spack install py-fenics-dolfinx
```

> **Note:** Depending on the local Spack repository, the package name may differ slightly (`py-fenics-dolfinx` vs `py-fenicsx-dolfinx`).

## 2nd: Install `py-h5py` and `py-numpy`

This step was necessary because `h5py` must be compiled consistently against the correct HDF5 (and compatible NumPy ABI) available on the system.

```bash
spack install py-h5py ^hdf5
spack install py-numpy
```

## 3rd: Install `py-petsc4py` with MUMPS support

My code uses MUMPS as the direct solver. The PETSc version automatically installed as a dependency of Dolfinx may not include MUMPS, which can produce errors such as:

```text
Could not locate solver type mumps for factorization type LU
```

To avoid this:

```bash
spack install py-petsc4py ^petsc+mumps+scalapack+metis
```

### Important:
Spack may now contain **multiple PETSc / petsc4py installations**.

To inspect them:

```bash
spack find -ldv py-petsc4py
```

Example output:

```bash
32aqyca py-petsc4py@3.22.1+mpi build_system=python_pip
c35jere py-petsc4py@3.22.1+mpi build_system=python_pip
```

You must identify and remove the unwanted build (typically the one installed automatically without MUMPS). Usually, the version not compiled with MUMPS is listed last (2nd line).

For example:

```bash
spack uninstall --force /c35jere
```

> **Important:** The hash will differ on every system. Always verify before uninstalling.

## 4th: Create a dedicated Python virtual environment

Navigate to the `StonedFenicsx` source folder.

### 4a: Create the environment

```bash
python -m venv {path_to_environment}/stoned_fenicsx
```

### 4b: Activate it

```bash
source {path_to_environment}/stoned_fenicsx/bin/activate
```

## 5th: Install `StonedFenicsx`

In most cases, the repository contains multiple `pyproject.toml` files:

- `pyproject_local.toml`
- `pyproject_HPC.toml`

Rename the HPC version:

```bash
mv pyproject_HPC.toml pyproject.toml
```

Then install:

```bash
pip install -e .
```


---

# Running the software on the HPC

Before running:

1. Load Spack  
2. Load all required Spack packages  
3. Activate the Python virtual environment  
4. Run the software  

# Example SLURM Script

Below is a minimal working example. Adjust paths, memory, and package names as needed.

```bash
#!/bin/bash
#SBATCH --job-name=JapanTests
#SBATCH --array=0-1%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH --output=out/Mexico_%A_%a.txt
#SBATCH --error=err/Mexico_%A_%a.txt


module purge

module load python/3.13.0

module load spack/0.23
unset LD_PRELOAD
export LD_LIBRARY_PATH="/mnt/scratch/wlnw570/libs/glu:$LD_LIBRARY_PATH"

spack load py-fenics-dolfinx
spack load py-h5py
spack load py-numpy
spack load py-petsc4py@3.22.1

source /mnt/scratch/wlnw570/stoned_fenicsx_env/bin/activate


mpirun -n 1 python -u Mexico_slab.py --case_index $SLURM_ARRAY_TASK_ID
```

# Final Remarks

- Avoid mixing **conda** and **Spack** unless you explicitly understand your library paths.  
- Use `module purge` before loading dependencies.  
- PETSc + MUMPS compatibility is often the most critical step.  
- When in doubt, inspect loaded libraries:

```bash
spack find -ldv
spack loaded
```

This workflow is not necessarily elegant, but it has proven robust enough for large-scale FEM workflows on Aire. Note that the setup process can take considerable time.

**Important Note on gmsh:** There is a known issue where gmsh requires the OSMesa/GLU library, which is typically only available on the login node, not on compute nodes. A temporary workaround is to copy the library to a local folder, unset `LD_PRELOAD`, and point `LD_LIBRARY_PATH` to the new location (as shown in the example script above). This is not an ideal solution, and I am seeking official guidance from the Aire support team on a more robust approach.  