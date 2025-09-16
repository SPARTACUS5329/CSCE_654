#!/bin/bash

#SBATCH --job-name=hw1_kovilur_aditya
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=result.out

module load PrgEnv-intel

make all USE_MKL=1
./hw_driver

