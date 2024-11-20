#!/bin/bash
#SBATCH --job-name=hubbard-spin-kell-aggregator # create a short name for your job
#SBATCH --output=slurm-%A.%a.out                # stdout file
#SBATCH --error=slurm-%A.%a.err                 # stderr file
#SBATCH --nodes=1                               # node count
#SBATCH --ntasks=1                              # total number of tasks across all nodes
#SBATCH --cpus-per-task=6                       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                        # memory per cpu-core (4G is default)
#SBATCH --time=05:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --array=0-4                             # job array with index values (both ends inclusive)
#SBATCH --mail-type=all                         # send email on job start, end and fault
#SBATCH --mail-user=jonas.kell@uni-a.de         # send email address
#SBATCH --partition=epyc                        # Select a partiton (epyc, epyc-mem or epyc-gpu)
 
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)
 
# Always assume your application might be multithreaded.
# Safeguard to limit number of threads to number of requested CPU cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Loading modules here
module purge
module load miniforge/24.7.1
conda activate aggregatorenv

# Execute Script
srun python batch-run-aggregator.py $SLURM_ARRAY_TASK_ID