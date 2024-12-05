#!/bin/bash

#SBATCH --array=0-2 # The indices you are passing to your script, inclusive, e.g. 0-2 means we run 3 Jobs
#SBATCH --partition=big # Normal Jobs should use partition work
#SBATCH --nodes=1 # Nodes per Job
#SBATCH --ntasks=1 # Tasks per Job
#SBATCH --cpus-per-task 64 # CPUs per Task, Increase this if your computation uses multiprocessing
#SBATCH --mem-per-cpu 16G # Memory per CPU, Increase this if your computation is memory heavy
#SBATCH --job-name=CAFE_p # Give a meaningful Job name
#SBATCH --time 16:00:00 # Maximum run-time
#SBATCH -o out-%A-%a.out # Logs are saved in the folder slurmlogs
#SBATCH -e err-%A-%a.out # Logs are saved in the folder slurmlogs

echo Running Task ID $SLURM_ARRAY_TASK_ID

python -u CAFE_parallel.py $SLURM_ARRAY_TASK_ID

exit
