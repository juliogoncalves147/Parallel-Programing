#!/bin/bash
#SBATCH --time=2:00
#SBATCH --partition=cpar
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=compute-134-102
perf stat -r 3 mpirun -np 4 ./bin/k_means 10000000 32 4