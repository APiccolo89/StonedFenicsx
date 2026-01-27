#!/bin/bash
#SBATCH --job-name=MPI_job
#SBATCH --time=30:00:01
#SBATCH --ntasks=1            # Number of MPI processes
#SBATCH --nodes=1              # Number of nodes
#SBATCH --mem-per-cpu=10G



# User specific environment and startup programs
module load miniforge/24.7.1
# Run your application
conda activate fenicsx-env

mpirun -n 1 python3 -u script_running.py  


