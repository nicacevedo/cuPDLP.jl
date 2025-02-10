#!/bin/bash
#SBATCH --job-name=IR_M_test # job name
#SBATCH --partition=sched_mit_sloan_gpu_r8 # partition
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=1 # cpu
#SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --output=IR_M_test.out
#SBATCH --error=IR_M_test.err
#SBATCH -t 0-01:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set

julia scripts/test.jl
# julia --project -e 'import Pkg; Pkg.instantiate()' # install dependencies




