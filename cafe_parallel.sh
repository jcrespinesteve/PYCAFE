#!/bin/bash

#SBATCH --array=0-2 # The indices you are passing to your script, inclusive, e.g. 0-2 means we run 3 Jobs
#SBATCH --partition=big 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task 64 
#SBATCH --mem-per-cpu 16G 
#SBATCH --job-name=CAFE_P
#SBATCH --time 16:00:00 
#SBATCH -o out-%A-%a.out 
#SBATCH -e err-%A-%a.out 

echo Running Task ID $SLURM_ARRAY_TASK_ID

python -u run_cafe_parallel.py $SLURM_ARRAY_TASK_ID

exit
