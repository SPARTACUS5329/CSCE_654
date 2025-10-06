#!/bin/bash
#SBATCH -A m5101              # Replace with your NERSC project account
#SBATCH -C cpu                # CPU partition
#SBATCH -q debug              # debug for short test
#SBATCH -t 00:20:00           # Walltime
#SBATCH -N 4                  # Number of nodes
#SBATCH -n 64                 # MPI ranks (must be a perfect square)
#SBATCH -c 8                  # CPU cores per rank

# Load modules
module load PrgEnv-gnu

# Build CPU-only version
make cpu

# Set OpenMP threads
export OMP_NUM_THREADS=8

# Run (example: 64 ranks : 8x8 process grid)
srun -n 64 -c 8 ./summa --N 1000 --verify
