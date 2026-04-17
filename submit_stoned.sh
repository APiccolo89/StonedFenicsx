#!/bin/bash
#SBATCH --job-name=MPI_test
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G

module purge
module load miniforge/24.7.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stoned_fenicsx

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_pmix=^slurm
export OMPI_MCA_plm=isolated

echo "=== dolfinx test ==="
python -c "from mpi4py import MPI; from dolfinx.mesh import create_unit_square; mesh = create_unit_square(MPI.COMM_WORLD, 4, 4); print('OK')"