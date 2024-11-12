# HPC - High Performance Computing in Augsburg

https://openproject-tp3.physik.uni-augsburg.de/projects/group/wiki/supercomputing
(Login required)

https://collab.dvb.bayern/display/UniARZHPCKB/Augsburg+Linux+Compute+Cluster
https://collab.dvb.bayern/display/UniARZHPCKB/Linux+Compute+Cluster+Augsburg

## Connection to `LiCCA` or `ALCC`

```shell
ssh -i ~/.ssh/hpcaugsburg kelljona@licca-li-01.rz.uni-augsburg.de

ssh -i ~/.ssh/hpcaugsburg kelljona@alcc129.rz.uni-augsburg.de
```

## Run the test-job

Copy all the stuff to the login-node or git-clone the repo.

```shell
# ONCE (or if scratch-disk is erased)
module load miniforge/24.7.1
conda create --channel conda-forge -n myenv python=3.12.4
conda activate myenv
conda deactivate

# RUN (in tests folder)
sbatch job.slurm
```

## Run the experiment-job

Copy all the stuff to the login-node or git-clone the repo.

```shell
# ONCE (or if scratch-disk is erased)
module load miniforge/24.7.1
conda create --channel conda-forge -n experimentenv python=3.12.4 numpy scipy
conda activate experimentenv
conda deactivate

# RUN (in experiments folder)
sbatch job.slurm

# Cleanup outputs
rm *.err
rm *.out
```

## Run the aggregator-job

Copy all the stuff to the login-node or git-clone the repo.

```shell
# ONCE (or if scratch-disk is erased)
module load miniforge/24.7.1
conda create --channel conda-forge -n aggregatorenv python=3.12.4 numpy scipy
conda activate aggregatorenv
conda deactivate

# RUN (in aggregator folder)
sbatch job.slurm

# Cleanup outputs
rm *.err
rm *.out
```

## Check the Job Status

```shell
squeue -u $USER
```

Cancel all your jobs

```shell
scancel -u $USER
```

## Remove Venv

```shell
module load miniforge/24.7.1
conda remove -n ENV_NAME --all
conda deactivate
```
