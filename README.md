# Combined Ranking and Regression for Algorithm Selection
This repository contains the code for "Combined Ranking and Regression for Algorithm Selection" (CoRRAS). 

## Installation
### Requirements
CoRRAS runs with Python 3.6.8. Most of the dependencies can be installed using the requirements.txt file and pip
`pip install -r requirements.txt`. The ASlibScenario package can be downloaded and installed from the [original repository](https://github.com/mlindauer/ASlibScenario). 

## Reproducing Experimental Results
Before trying to run the experiments, make sure to download the benchmark In order to reproduce the experimental results, the python scripts in the "experiments" folder can be used. The following table illustrates which script corresponds to which approach:

| Script    | Approach           |
|-----------|--------------------|
| experiment_lin_pl.py    | PL-LM, PL-QM       |
| experiment_nn_pl.py   | PL-NN              |
| experiment_lin_hinge.py    | Hinge-LM, Hinge-QM |
| experiment_nn_hinge.py | Hinge-NN           |

The scripts take 6 input arguments. The first two arguments are used to parallelize the evaluation procedure over multiple jobs. The first argument is the total number of jobs, the second is the index of the current job. The remaining 4 arguments specify connection to the database which will be used to store the resutls. We assume a MySQL server with version 5.7.9 running. The scripts create one database table per scenario. For example, to run the full set of experiments in a single job, the call may look as follows

`python3 experiment_lin_pl.py 1 0 db_url db_user db_password db_name`

After running the experiments, the raw predictions are stored in the specified database. The scripts provided in the "evaluation" folder can then be used to evaluate the resutls in terms of several performance measures such as Kendall's Tau or the penalized average runtime. The scripts take 4 arguments, namely the database credentials. The evaluations will be stored in a csv file locally in the "evaluations" folder. There will be one csv file per database table, containing the names of the problem instances of the respective scenario, the configuration and the achieved performances. Following the above example, the call may look as follows

`python3 experiment_lin_pl.py 1 0 db_url db_user db_password db_name`

These results can be plotted using the script in the folder "plots" in order to reproduce the papers plots. The papers table can be reproduced using the "table_ki2020.py" script.