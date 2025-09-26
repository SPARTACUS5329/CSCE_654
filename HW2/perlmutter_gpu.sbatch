#!/bin/bash

#SBATCH --job-name=hw2_kovilur_aditya
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=result.out

module purge
module load cudatoolkit

make clean && make -f Makefile_Perlmutter
srun ./cuda_hw --task 0

# ./cuda_hw --task 0 // DGEMM naive
# ./cuda_hw --task 1 // DGEMM tiled
# ./cuda_hw --task 2 // DGEMM cuBLAS
# ./cuda_hw --task 3 // Attention with DGEMM naive
# ./cuda_hw --task 4 // Attention with DGEMM tiled
# ./cuda_hw --task 5 // Attention with cuBLAS

## Change these
srun -n 1 -G 1 ./cuda_hw --task 0 --M 2048 --N 2048 --K 2048
srun -n 1 -G 1 ./cuda_hw --task 1 --M 2048 --N 2048 --K 2048 --TILE 32
srun -n 1 -G 1 ./cuda_hw --task 2 --M 2048 --N 2048 --K 2048

srun -n 1 -G 1 ./cuda_hw --task 3 --L 1024 --D 256
srun -n 1 -G 1 ./cuda_hw --task 4 --L 1024 --D 256
srun -n 1 -G 1 ./cuda_hw --task 5 --L 1024 --D 256
