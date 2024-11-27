# !/bin/bash
# SBATCH --job-name=IR_batch # job name
# SBATCH --partition=sched_mit_sloan_gpu # partition
# SBATCH --ntasks 1 # number of tasks
# SBATCH --cpus-per-task=1 # cpu
# SBATCH --mem-per-cpu=64GB # memory per cpu
# SBATCH --output=IR_batch.out
# SBATCH --error=IR_batch.err
# SBATCH -t 0-24:00:00 # time format is day-hours:minutes:seconds
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set

# julia --project -e 'import Pkg; Pkg.instantiate()' # install dependencies

smallInstances=("hgms30" "var-smallemery-m6j6" "nursesched-medium-hint03" "neos-2629914-sudost"
 "neos-4966258-blicks" "neos-3755335-nizao" "neos-3695882-vesdre" "neos6"
 "cmflsp40-24-10-7" "triptim7" "neos-2746589-doon" "reblock420"
 "neos-872648" "neos-4760493-puerua" "fhnw-schedule-pairb200" "sct1"
 "t1717" "iis-hc-cov" "gmut-75-50" "t1722" "ex1010-pi"
 "neos-5221106-oparau" "neos-1354092" "neos-827175" "radiationm40-10-02"
 "nw04" "neos-4359986-taipa" "neos-960392" "map18" "neos-932721"
 "gmut-76-40")

echo "SMALL INSTANCES"
for i in $(seq 1 2); 
do
    echo "Solving the instance: ${smallInstances[i-1]} w/ max_iter=0"; 
    julia --project scripts/iterative_refinement.jl --instance_path="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/${smallInstances[i-1]}.mps.gz" --output_directory="./output/MIPLIB_batch/small_instances" --iter_tolerance=1e-3 --obj_tolerance=1e-8 --time_sec_limit=3600 --max_iter=0
    echo "Solving the instance: ${smallInstances[i-1]} w/ max_iter=inf"; 
    julia --project scripts/iterative_refinement.jl --instance_path="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/${smallInstances[i-1]}.mps.gz" --output_directory="./output/MIPLIB_batch/small_instances" --iter_tolerance=1e-3 --obj_tolerance=1e-8 --time_sec_limit=3600 --max_iter=1000
done

# julia +1.10.4 --project scripts/auxiliary_ir.jl --instance_path="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/hgms30" --output_directory="./output/MIPLIB_batch/small_instances" --iter_tolerance=1e-3 --obj_tolerance=1e-8 --time_sec_limit=3600 --max_iter=0


# mediumInstances=("ds-big" "neos-4647030-tutaki" "neos-5129192-manaia" "graph40-80-1rand"
#  "neos-5049753-cuanza" "seqsolve1" "neos-5102383-irwell" "bab6"
#  "neos-5123665-limmat" "shs1014" "shs1042" "wnq-n100-mw99-14"
#  "fhnw-binschedule0" "fhnw-binschedule1" "neos-4321076-ruwer"
#  "physiciansched3-3" "neos-5079731-flyers" "neos-3322547-alsek"
#  "neos-4647027-thurso" "ns1644855" "datt256" "kosova1"
#  "neos-4533806-waima" "neos-4647032-veleka" "z26" "neos-5118851-kowhai"
#  "neos-4972437-bojana" "hgms62" "in" "zeil")

# echo "MEDIUM INSTANCES"
# for i in $(seq 1 30); 
# do
#     echo "Solving the instance: ${mediumInstances[i-1]} w/ max_iter=0"; 
#     julia --project scripts/iterative_refinement.jl --instance_path="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/${mediumInstances[i-1]}.mps.gz" --output_directory="./output/MIPLIB_batch/medium_instances" --iter_tolerance=1e-3 --obj_tolerance=1e-8 --time_sec_limit=3600 --max_iter=0
#     echo "Solving the instance: ${mediumInstances[i-1]} w/ max_iter=inf"; 
#     julia --project scripts/iterative_refinement.jl --instance_path="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/${mediumInstances[i-1]}.mps.gz" --output_directory="./output/MIPLIB_batch/medium_instances" --iter_tolerance=1e-3 --obj_tolerance=1e-8 --time_sec_limit=3600 --max_iter=1000
# done



