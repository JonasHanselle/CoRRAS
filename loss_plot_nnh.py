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

scenario_path = "./aslib_data-aslib-v4.0/"
result_path = "./losses-nnh-new/"
figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/plots/hinge_nn/losses"

scenarios = ["SAT11-INDU"]
# scenarios = ["CSP-2010", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND"]
lambda_values = [0.4]
epsilon_values = [1.0]
max_pairs_per_instance = 5
seeds = [15]

learning_rates = [0.001]
batch_sizes = [128]
es_patiences = [64]
es_intervals = [8]
es_val_ratios = [0.3]
layer_sizes_vals = [[32]]
activation_functions = ["sigmoid"]
use_weighted_samples_values = [False, True]

splits = [4]
params = [
    scenarios, lambda_values, epsilon_values, splits, seeds, learning_rates,
    es_intervals, es_patiences, es_val_ratios, batch_sizes, layer_sizes_vals,
    activation_functions, use_weighted_samples_values
]

param_product = list(product(*params))

fig, axes = plt.subplots(1, 2)

for index, (scenario_name, lambda_value, epsilon_value, split, seed,
            learning_rate, es_interval, es_patience, es_val_ratio, batch_size,
            layer_size, activation_function,
            use_weighted_samples) in enumerate(param_product):
    params_string = "-".join([
        scenario_name,
        str(lambda_value),
        str(epsilon_value),
        str(split),
        str(seed),
        str(learning_rate),
        str(es_interval),
        str(es_patience),
        str(es_val_ratio),
        str(batch_size),
        str(layer_size),
        str(activation_function),
        str(use_weighted_samples)
    ])

    filename = "nn_hinge" + "-" + params_string + ".csv"
    loss_filename = "nn_hinge" + "-" + params_string + "-losses.csv"
    filepath = result_path + filename
    loss_filepath = result_path + loss_filename
    print(loss_filepath)
    figure_file = params_string.replace(".", "_") + "-losses" + ".pdf"
    try:
        df = pd.read_csv(loss_filepath)
    except:
        print(
            "loss file name",
            "nn_hinge-SAT11-INDU-0.6-1.0-4-15-0.001-8-64-0.3-128-losses.csv")
        print("File " + loss_filename + " not found")
        continue
    df.rename(columns={"SQH": "SHL"}, inplace=True)
    df["$\lambda$ MSE"] = lambda_value * df["MSE"]
    df["$(1 - \lambda)$ SHL"] = (1 - lambda_value) * df["SHL"]
    df["TOTAL_LOSS"] = df["$\lambda$ MSE"] + df["$(1 - \lambda)$ SHL"]
    df = df.melt(id_vars=["iter"])
    # print(df.head())
    # text = "$\lambda$ = " + str(lambda_value) + ", "
    # text += "split = " + str(split) + ", "
    # text += "seed = " + str(seed) + ",\n"
    # text += "learning rate = " + str(learning_rate) + ", "
    # text += "es patience = " + str(es_patience) + ", "
    # text += "es interval = " + \
    #     str(es_interval) + ", "
    # text += "es val raio = " + str(es_val_ratio) + ",\n"
    # text += "batch size = " + str(batch_size)
    # plt.tight_layout()
    # plt.annotate(text, (0, 0), (0, -40), xycoords="axes fraction",
    #              textcoords="offset points", va="top")
    lp = sns.lineplot(x="iter",
                      y="value",
                      hue="variable",
                      data=df,
                      legend=False,
                      ax=axes[index])
    loss_filename_es = "nn_hinge" + "-" + params_string + "-es-val.csv"
    filepath_es = result_path + loss_filename_es
    es_filepath = result_path + loss_filename_es
    try:
        df_es = pd.read_csv(es_filepath)
    except:
        print("File " + loss_filename + " not found")
        continue
    df_es = df_es.melt(id_vars=["es_call"])
    df_es["es_call"] = es_interval * df_es["es_call"]
    lp = sns.lineplot(x="es_call",
                      y="value",
                      hue="variable",
                      data=df_es,
                      legend=False,
                      markers='o',
                      markersize=8,
                      ax=axes[index],
                      palette=["brown"])
    if index == 0:
        axes[index].set_title(scenario_name + " Unweighted")
    else:
        axes[index].set_title(scenario_name + " Weighted")
    axes[index].set_ylabel("Value")
    axes[index].set_xlabel("Epoch")
    # plt.show()
    # plt.savefig(figures_path+figure_file, bbox_inches="tight")
    # filename = "nn_pl" + "-" + params_string + ".csv"

labels = [
    "MSE", "SHL", "$(1-\\lambda)$MSE", "$\\lambda$SHL", "Total Loss",
    "Val Loss"
]
plt.annotate("", (0, 0), (0, -40),
             xycoords="axes fraction",
             textcoords="offset points",
             va="top")
fig.set_size_inches(8, 3.3)
legend = fig.legend([axes],
                    labels=labels,
                    loc="lower center",
                    ncol=len(labels) // 2,
                    bbox_to_anchor=(0.45, -0.00))
plt.subplots_adjust(bottom=0.3)
# plt.figure(fig.number)
# plt.show()
plt.savefig(figures_path + figure_file, bbox_inches="tight")

# print(df.head())
# text = "$\lambda$ = " + str(lambda_value) + ", "