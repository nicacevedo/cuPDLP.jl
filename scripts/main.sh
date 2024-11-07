#!/bin/bash
#SBATCH --job-name=cuPDLP_test # job name
#SBATCH --partition=sched_any # partition
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=2 # cpu
#SBATCH --mem-per-cpu=1GB # memory per cpu
#SBATCH --output=cuPDLP_test.out
#SBATCH --error=cuPDLP_test.err
#SBATCH -t 0-00:10:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni.acevedo.villena@gmail.com
# SBATCH --array=1-22%10

module load julia/1.6.5
julia --project -e 'import Pkg; Pkg.instantiate()'

# julia --project scripts/solve.jl \
# --instance_path=INSTANCE_PATH --output_directory=OUTPUT_DIRECTORY \
# --tolerance=TOLERANCE --time_sec_limit=TIME_SEC_LIMIT


#julia --project scripts/solve.jl --instance_path="C:/Users/niace/Documents/Github/MIT/cuPDLP.jl/instance/mps1.mps" --output_directory="./output" --tolerance=0.0001 --time_sec_limit=300



# Copy the following files, with ".mps.gz" extension,
# from a ssh server to the local machine
rmatr200-p10
momentum1
neos-1171448
piperout-d20
ns1856153
ns930473
graph20-80-1rand
supportcase37
2club200v15p5scn
ci-s4
bab5
dws008-03
brazil3
neos-827175
cmflsp50-24-8-8
leo1
supportcase40
30n20b8
ger50-17-trans-dfn-3t
ex1010-pi
mzzv11
fiball
pb-grow22
lr1dr04vc05v17a-t360
sorrell7
opm2-z8-s0
neos-1354092
neos-4359986-taipa
neos-3372571-onahau
t1722

# Copy the following files, with ".mps.gz" extension,
# from a ssh server to the local machine


scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/rmatr200-p10.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/rmatr200-p10.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/momentum1.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/momentum1.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/neos-1171448.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/neos-1171448.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/piperout-d20.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/piperout-d20.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/ns1856153.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/ns1856153.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/ns930473.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/ns930473.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/graph20-80-1rand.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/graph20-80-1rand.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/supportcase37.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/supportcase37.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/2club200v15p5scn.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/2club200v15p5scn.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/ci-s4.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/ci-s4.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/bab5.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/bab5.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/dws008-03.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/dws008-03.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/brazil3.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/brazil3.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/neos-827175.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/neos-827175.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/cmflsp50-24-8-8.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/cmflsp50-24-8-8.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/leo1.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/leo1.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/supportcase40.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/supportcase40.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/30n20b8.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/30n20b8.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/ger50-17-trans-dfn-3t.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/ger50-17-trans-dfn-3t.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/ex1010-pi.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/ex1010-pi.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/mzzv11.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/mzzv11.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/fiball.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/fiball.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/pb-grow22.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/pb-grow22.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/lr1dr04vc05v17a-t360.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/lr1dr04vc05v17a-t360.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/sorrell7.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/sorrell7.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/opm2-z8-s0.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/opm2-z8-s0.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/neos-1354092.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/neos-1354092.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/neos-4359986-taipa.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/neos-4359986-taipa.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/neos-3372571-onahau.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/neos-3372571-onahau.mps.gz
scp /nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/t1722.mps.gz /home/nacevedo/RA/cuPDLP.jl/MIPLIB/t1722.mps.gz