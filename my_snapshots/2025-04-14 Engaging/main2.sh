#!/bin/bash
#SBATCH --job-name=MR_Bch_T # job name
#SBATCH --partition=sched_mit_sloan_gpu_r8 # partition
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=1 # cpu
#SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=64GB
#SBATCH --output=MR_Bch_T.out
#SBATCH --error=MR_Bch_T.err
#SBATCH -t 0-24:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set
# SBATCH --exclusive # exclusiveness of the node (?)

julia scripts/test_copy3.jl
# julia --project -e 'import Pkg; Pkg.instantiate()' # install dependencies