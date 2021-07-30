This is the repository for the papers [Warm-Starting AC Optimal Power Flow with GraphNeural Networks](https://www.climatechange.ai/papers/neurips2019/1/paper.pdf) and [Applying Graph Neural Networks on HeterogeneousNodes and Edge Features](https://grlearning.github.io/papers/6.pdf).

It is the final repository of a research project; accordingly, it is not well-documented and might contain errors that have been introduced during clean-up.

![Texas Dataset Visualization](ACTIVSg2000_presentation.jpg?raw=true "Texas Dataset Visualization")

# Usage

## Creating Samples

Creating samples using ACOPF takes a while (that's the reason for this paper, after all). Accordingly, we pre-compute them: Use 

```python -u utils/create_data.py --case ACTIVSg200```

to create new casefiles for the Illinois dataset and `ACTIVSg2000` for the Texas dataset.

WARNING: This assumes you have your files in `/experiment/data` and want your results in `/experiment/results`. That's easily doable using singularity containers; if not, change them.

This works in parallel, with as many workers as you'd like: It first loads the power network and the casefiles, then uses the `runs.pickly_sync` file to synchronize runs between workers, with a primitive lock mechanism. The file contains a dict in which, for each scenario ID, the key is either "running" or "done" depending on their current state.

I generally created the cases using 40 workers in parallel, on 2 CPUs each.

## Training The Models

To train models, you'll use the `train_models.py` script. Call it with 

```
python -u gnn_acopf/experimental/train_models.py --dataset ACTIVSg200
```

or `ACTIVSg2000`, depending on your casefile. Run it with `--local` and it assumes your data can be found in `../../data`, your results in `../../experiment/results`, and your checkpoints should be stored in `../../experiment/checkpoints`. If not, they'll all be stored in subfolders of `/experiment/`.

It trains a single model (which needs to be defined in code; the models are commented out in `train_models.py`) for 500 epochs, then prints out training, validation, and test results. It gives you mse, mae, and maes for buses, generators, and branches. By default, it uses the paper's best model.


## Evaluate GNN/ACOPF Combination

Once a model has been trained, you can evaluate it using the `evaluate_opf.py` script. Call it with 

```
python3 -u gnn_acopf/experimental/evaluate_opf.py --dataset ACTIVSg200
```

or `ACTIVSg2000` respectively. This evaluates ACOPF, Model -> ACOPF, DCOPF, DCOPF -> ACOPF and DCOPF for one specific model, which you can set in code. You'll also have to explicitly set the model type; by default it's using the best model from the paper.

Similar to creating the samples, this uses a simple synchronization mechanic to synchronize results between as many processes as you'll start to parallelize the run. I've used 40 of those. It writes our the results to `results.csv`.

## Creating Visualizations

To create the visualizations found in the paper, two different methods exist: 

`produce_evaluation` creates a number of different statistics based on the csv results file from `evaluate_opf.py`. These include mean and 95 percentile runtime and the number of cases solved, allowing you to produce Table 1 and Fig. 1 from the paper.

`plot_map` allows you to produce the maps in the paper's appendix. Call it as `python plot_map.py` and choose in code which of the maps you want to produce (ACTIVSg200, or ACTIVSg2000 both in a presentation and a paper mode).


# Installation on Local

The following assumes that you have pyenv including the pyenv virtualenv extension installed.


## Install Julia

For the `powermodels.jl` integration, we'll need a Julia integration. To install an appropriate Julia version:

1. Choose a directory for it to be installed; in this example, we'll use `~/workspace/clones/powermodels-julia`; assign it to a variable: `export JULIADIR=~/workspace/clones/powermodels-julia`
2. Create it: `mkdir -p $JULIADIR`, then change to it `cd $JULIADIR`
3. Download and untar it: `curl https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.4-linux-x86_64.tar.gz | tar xz`
4. Make sure it's on the path: `export PATH=$JULIADIR/julia-1.0.4/bin:$PATH` and `export JULIA_DEPOT_PATH=$JULIADIR/.julia/` and `JULIA_LOAD_PATH=$JULIADIR/.julia/`.
5. Install the required packages within Julia: `julia -e 'using Pkg; Pkg.activate("/opt/julia/.julia"); Pkg.add(["Pkg", "Ipopt", "PowerModels", "JSON", "SCS", "Cbc", "JuMP", "Libdl", "REPL"])'`


## Install Python/Julia Bridge

1. Create a new virtualenv: `pyenv virtualenv 3.7.2 powermodels`; acivate it: `pyenv activate powermodels`.


2. Install PyCall for julia: `julia -e 'ENV["PYTHON"]="python"; import Pkg; Pkg.activate("/workspace/clones/powermodels-julia/.julia"); Pkg.add("PyCall"); Pkg.build("PyCall"); using PyCall'`
3. Install the julia package for python: `pip install julia`


## Install Python

1. Install standard packages: `pip install numpy==1.20.2 ruamel.yaml==0.16.1 portalocker==1.5.1 pyyaml==5.1.2 seaborn==0.9.0 ipython==7.7.0 scikit-image==0.15.0 dill==0.3.0 beautifultable==0.7.0 requests==2.22.0 pandas==1.2.3 cartopy owslib`
2. Install torch requirements: `pip install torch==1.2.0 torchvision==0.4.0 Pillow==6.1.0`
3. Install pytorch-geometric: `pip install torch-cluster==1.4.4 torch-geometric==1.3.1 torch-scatter==1.3.1 torch-sparse==0.4.0 torch-spline-conv==1.1.0`

# Installation in a Singularity container

The original experiments have been made using singularity containers. To create a singularity container with the required software, just call 

```
sudo singularity build powermodels.sif powermodels.singularity
```

# Tests

Tests can be run using `pytest` in the `tests` subdirectory. Originally, they ran within a CI.
