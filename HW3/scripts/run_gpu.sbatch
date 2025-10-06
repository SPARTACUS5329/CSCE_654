#!/bin/bash
#SBATCH -A m5101              # Replace with your NERSC project account
#SBATCH -C gpu                # GPU partition
#SBATCH -q regular            # regular queue (or debug for short test)
#SBATCH -t 00:20:00           # Walltime
#SBATCH -N 4                  # Number of nodes
#SBATCH -n 16                 # MPI ranks (must be a perfect square)
#SBATCH -c 8                  # CPU cores per rank
#SBATCH --gpus-per-task=1     # 1 GPU per rank

# Load modules
module load PrgEnv-gnu
module load cudatoolkit

# Build GPU version
make

# Set OpenMP threads for host-side work
export OMP_NUM_THREADS=8

# Run (example: 16 ranks : 4x4 process grid)
srun -n 16 -c 8 ./summa --N 1000 --gpu --verify
