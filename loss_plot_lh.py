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

sns.set_style("whitegrid")

scenario_names = ["SAT11-INDU"]
scenarios = ["SAT11-INDU"]
scenario_path = "./aslib_data-aslib-v4.0/"
result_path = "./losses-lh/"
figures_path = "./figures_loss_hist_pl/"
figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/plots/hinge/losses"
lambda_values = [0.6]
epsilon_values = [1.0]
split = 4
seed = 15

max_pairs_per_instance = 5
maxiter = 100
seeds = [15]
use_quadratic_transform_values = [False, True]
use_max_inverse_transform_values = ["max_cutoff"]
scale_target_to_unit_interval_values = [True]
skip_censored_values = [False]
use_weighted_samples_values = [False]
regulerization_params_values = [0.001]

splits = [4]
params = [
    scenarios, lambda_values, epsilon_values, splits, seeds,
    use_quadratic_transform_values, use_max_inverse_transform_values,
    scale_target_to_unit_interval_values, skip_censored_values,
    regulerization_params_values, use_weighted_samples_values
]

param_product = list(product(*params))

# /home/jonas/Documents/CoRRAS/loss-hists/loss-hist-hinge-SAT11-RAND-1.0-0.6-4-15.csv
fig, axes = plt.subplots(1, 2)

for index, (scenario_name, lambda_value, epsilon_value, split, seed,
            use_quadratic_transform, use_max_inverse_transform,
            scale_target_to_unit_interval, skip_censored, regulerization_param,
            use_weighted_samples) in enumerate(param_product):
    params_string = "-".join([
        scenario_name,
        str(lambda_value),
        str(epsilon_value),
        str(split),
        str(seed),
        str(use_quadratic_transform),
        str(use_max_inverse_transform),
        str(skip_censored),
        str(regulerization_param),
        str(scale_target_to_unit_interval),
        str(use_weighted_samples)
    ])

    filename = "linear_hinge" + "-" + params_string + ".csv"
    loss_filename = "linear_hinge" + "-" + params_string + "-losses.csv"
    filepath = result_path + filename
    loss_filepath = result_path + loss_filename
    figure_file = params_string.replace(".", "_") + "-losses" + ".pdf"
    df = None
    if not os.path.isfile(loss_filepath):
        print("File for " + params_string + " not found!")
        continue
    df = pd.read_csv(loss_filepath)
    print(df)
    df = df.rename(columns={"call": "iteration"})
    # df = df.rename(columns={"NLL":"PL-NLL"})
    # df["SHL"] = df["SQUARED_HINGE"]
    # df["MSE"] = df["MEAN_SQUARED_ERROR"]

    df.rename(columns={
        "SQUARED_HINGE": "SHL",
        "MEAN_SQUARED_ERROR": "MSE"
    },
              inplace=True)

    df["$\lambda$ SHL"] = lambda_value * df["SHL"]
    df["$(1 - \lambda)$ MSE"] = (1 - lambda_value) * df["MSE"]
    df["TOTAL_LOSS"] = df["$\lambda$ SHL"] + df["$(1 - \lambda)$ MSE"]
    df = df.melt(id_vars=["iteration"])
    print(df.head())
    text = "$\lambda$ = " + str(lambda_value) + ", "
    text += "split = " + str(split) + ", "
    text += "seed = " + str(seed) + ", "
    text += "quadratic transform: " + str(use_quadratic_transform) + ",\n"
    text += "max inverse transform: " + str(use_max_inverse_transform) + ", "
    text += "scale target to unit interval: " + \
        str(scale_target_to_unit_interval) + ", "

    # plt.clf()
    # plt.tight_layout()
    # plt.annotate(text,(0,0), (0,-40), xycoords="axes fraction", textcoords="offset points", va="top")
    lp = sns.lineplot(x="iteration",
                      y="value",
                      hue="variable",
                      data=df,
                      ax=axes[index],
                      legend=None)
    axes[index].set_xlabel("Iteration")
    axes[index].set_ylabel("Value")
    if index == 0:
        axes[index].set_title(scenario_name + " LM")
    else:
        axes[index].set_title(scenario_name + " QM")
    # if index == 1:
labels = ["MSE", "SHL", "$\\lambda$SHL", "$(1-\\lambda)$MSE", "Total Loss"]
plt.annotate("", (0, 0), (0, -40),
             xycoords="axes fraction",
             textcoords="offset points",
             va="top")
fig.set_size_inches(8, 3.3)
legend = fig.legend(list(axes),
                    labels=labels,
                    loc="lower center",
                    ncol=len(labels),
                    bbox_to_anchor=(0.45, -0.00))
plt.subplots_adjust(bottom=0.25)
# plt.show()
plt.savefig(figures_path + figure_file, bbox_inches="tight")
os.system("pdfcrop " + figures_path + figure_file + " " + figures_path +
          figure_file)
