#!/bin/bash
#SBATCH --job-name=MPI_job
#SBATCH --time=30:00:01
#SBATCH --ntasks=1            # Number of MPI processes
#SBATCH --nodes=1              # Number of nodes
#SBATCH --mem-per-cpu=5G


module purge
module load miniforge/24.7.1
module load spack/0.23
source /opt/apps/pkg/interpreters/miniforge/24.7.1/bin/activate
conda activate /users/wlnw570/.conda/envs/stoned_fenicsx

export CONDA_PREFIX=/users/wlnw570/.conda/envs/stoned_fenicsx
export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1


# User specific environment and startup programs
module load miniforge/24.7.1
# Run your application
conda activate stoned_fenicsx



mpirun -n 1 python -u Tonga_slab.py --steady_state 1 


