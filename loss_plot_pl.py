import os 

import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from itertools import product

# measures
from scipy.stats import kendalltau, describe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Corras.Evaluation.evaluation import ndcg_at_k, compute_relevance_scores_unit_interval 

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

scenario_names = ["SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND"]
scenario_path = "./aslib_data-aslib-v4.0/"
result_path = "./results/results-pl/"
figures_path = "./figures_loss_hist_pl/"
lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# epsilon_values = [0.2,0.4,0.6,0.8,1]
split = 4
seed = 15

# scenarios = ["MIP-2016", "CSP-2010", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND"]
# lambda_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
#     0.7, 0.8, 0.9, 0.95, 0.9999, 1.0]
scenarios = ["SAT11-INDU"]
lambda_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 0.95, 0.9999, 1.0]
max_pairs_per_instance = 5
maxiter = 100
seeds = [15]
use_quadratic_transform_values = [False]
use_max_inverse_transform_values = ["none", "max_cutoff", "max_par10"]
scale_target_to_unit_interval_values = [True, False]

splits = [4]
params = [scenarios, lambda_values, splits, seeds, use_quadratic_transform_values, use_max_inverse_transform_values, scale_target_to_unit_interval_values]


param_product = list(product(*params))

# /home/jonas/Documents/CoRRAS/loss-hists/loss-hist-hinge-SAT11-RAND-1.0-0.6-4-15.csv

for scenario_name, lambda_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval in param_product:
    params_string = "-".join([scenario_name,
                              str(lambda_value), str(split), str(seed), str(use_quadratic_transform), str(use_max_inverse_transform), str(scale_target_to_unit_interval)])

    filename = "pl_log_linear" + "-" + params_string + ".csv"
    loss_filename = "pl_log_linear" + "-" + params_string + "-losses.csv"
    filepath = result_path + filename
    loss_filepath = result_path + loss_filename    
    figure_file = params_string + "losses" + ".pdf"
    df = None
    if not os.path.isfile(loss_filepath):
        print("File for " + params_string + " not found!")
        continue
    df = pd.read_csv(loss_filepath)
    print(df)
    df = df.rename(columns={"iter":"iteration"})
    # df = df.rename(columns={"NLL":"PL-NLL"})
    df["$\lambda$ NLL"] = lambda_value * df["NLL"]
    df["$(1 - \lambda)$ MSE"] = (1 - lambda_value) * df["MSE"]
    df["TOTAL_LOSS"] = df["$\lambda$ NLL"] + df["$(1 - \lambda)$ MSE"]
    df = df.melt(id_vars=["iteration"])
    print(df.head())
    text = "$\lambda$ = " + str(lambda_value) + ", "
    text += "split = " + str(split) + ", "
    text += "seed = " + str(seed) + ", "
    text += "quadratic transform: " + str(use_quadratic_transform) + ",\n"
    text += "max inverse transform: " + str(use_max_inverse_transform) + ", "
    text += "scale target to unit interval: " + str(scale_target_to_unit_interval) + ", "

    plt.clf()
    # plt.tight_layout()
    plt.annotate(text,(0,0), (0,-40), xycoords="axes fraction", textcoords="offset points", va="top")
    lp = sns.lineplot(x="iteration", y="value", hue="variable", data=df)
    plt.title(scenario_name)
    plt.savefig(figures_path+figure_file, bbox_inches="tight")