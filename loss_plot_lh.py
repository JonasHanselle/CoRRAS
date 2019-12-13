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
scenario_names = ["CSP-2010"]
scenario_path = "./aslib_data-aslib-v4.0/"
result_path = "./losses-lh/"
figures_path = "./figures_loss_hist_lh/"
lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                 0.7, 0.8, 0.9]
epsilon_values = [0, 0.0001, 0.001, 0.01, 0.1,
                  0.2]
splits = [1]
seeds = [15]

use_quadratic_transform_values = [False, True]
use_max_inverse_transform_values = ["none", "max_cutoff", "max_par10"]
scale_target_to_unit_interval_values = [True, False]

params = [scenario_names, lambda_values, epsilon_values, splits, seeds, use_quadratic_transform_values, use_max_inverse_transform_values, scale_target_to_unit_interval_values]
param_product = list(product(*params))

for scenario_name, lambda_value, epsilon_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval in param_product:
    params_string = "-".join([scenario_name,
                              str(lambda_value), str(epsilon_value), str(split), str(seed), str(use_quadratic_transform), str(use_max_inverse_transform), str(scale_target_to_unit_interval)])

    filename = "linear_hinge" + "-" + params_string + ".csv"
    loss_filename = "linear_hinge" + "-" + params_string + "-losses.csv"
    filepath = result_path + filename
    loss_filepath = result_path + loss_filename    
    print(loss_filepath)
    figure_file = params_string + "losses" + ".pdf"
    try:
        df = pd.read_csv(loss_filepath)
    except:
        print("File " + loss_filename + " not found")
        continue
    df = df.rename(columns={"call":"iteration"})
    df["$\lambda$ MSE"] = lambda_value * df["MEAN_SQUARED_ERROR"]
    df["$(1 - \lambda)$ SQH"] = (1 - lambda_value) * df["SQUARED_HINGE"]
    df["TOTAL_LOSS"] = df["$\lambda$ MSE"] + df["$(1 - \lambda)$ SQH"]
    df = df.melt(id_vars=["iteration"])
    print(df.head())
    text = "$\lambda$ = " + str(lambda_value) + ", "
    text += "$\epsilon$ = " + str(epsilon_value) + ", "
    text += "split = " + str(split) + ", "
    text += "seed = " + str(seed) + ",\n"
    text += "quadratic transform: " + str(use_quadratic_transform) + ", "
    text += "max inverse transform: " + str(use_max_inverse_transform) + ", "
    text += "scale target to unit interval: " + str(scale_target_to_unit_interval) + ", "
    plt.clf()
    # plt.tight_layout()
    plt.annotate(text,(0,0), (0,-40), xycoords="axes fraction", textcoords="offset points", va="top")
    lp = sns.lineplot(x="iteration", y="value", hue="variable", data=df)
    plt.title(scenario_name)
    plt.savefig(figures_path+figure_file, bbox_inches="tight")