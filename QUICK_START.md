# Quick Start Instructions

## Commands to run the computation

Running with most default settings.

```shell
cd computation-scripts
python3 script.py
```

## How to set arguments

The program is controlled via setting command line arguments to the `script.py`.
Quite a few settings need to be adapted to get correct behavior.
Therefor specific experiments can be started with the `aggregator.py` script.

The arguments itself can be seen in th respective scripts and should for the most part be self-explanatory.

```shell
cd calculation-helpers
python3 aggregator.py
```

## Running on a server

There is a pre-configuration, for batching aggregator-runs/other experiments with different parameters.

See [THIS README](./hpc-augsburg/README.md) for how to run on the `University of Augsburg` High-Performance-Computing-Clusters.
