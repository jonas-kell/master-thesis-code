# HPC - High Performance Computing in Augsburg

## Connection to `LiCCA` or `ALCC`

```cmd
ssh -i ~/.ssh/hpcaugsburg kelljona@licca-li-01.rz.uni-augsburg.de

ssh -i ~/.ssh/hpcaugsburg kelljona@alcc129.rz.uni-augsburg.de
```

## Run the test-job

Copy all the stuff to the login-node or git-clone the repo.

```cmd
# ONCE (or if scratch-disk is erased)
module load anaconda/2024.02
conda create --channel conda-forge -n myenv python=3.12.4 numpy
conda activate myenv
conda deactivate

# RUN (in tests folder)
sbatch job.slurm
```

## Run the experiment-job

Copy all the stuff to the login-node or git-clone the repo.

```cmd
# ONCE (or if scratch-disk is erased)
module load anaconda/2024.02
conda create --channel conda-forge -n experimentenv python=3.12.4 numpy
conda activate experimentenv
conda deactivate

# RUN (in experiments folder)
sbatch job.slurm

# Cleanup outputs
rm *.err
rm *.out
```

## Check the Job Status

```cmd
squeue -u $USER
```

Cancel all your jobs

```cmd
scancel -u $USER
```
