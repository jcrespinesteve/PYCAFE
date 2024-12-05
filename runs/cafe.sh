#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=big
#SBATCH --mem-per-cpu 30G
#SBATCH --job-name=CAFE
#SBATCH --time=17:00:00
#SBATCH --output=cafe.out
#SBATCH --error=cafe.err

python run_cafe.py
