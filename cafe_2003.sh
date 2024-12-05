#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=big
#SBATCH --mem-per-cpu 30G
#SBATCH --job-name=CAFE_03
#SBATCH --time=17:00:00
#SBATCH --output=2003.out
#SBATCH --error=2003.err

python CAFE_2003.py
