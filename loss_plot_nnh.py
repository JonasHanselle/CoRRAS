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
figures_path = "./figures_loss_hist_nnh/"

scenarios = ["MIP-2016", "CSP-2010", "SAT11-HAND"]
# scenarios = ["CSP-2010", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND"]
lambda_values = [0.0, 0.3, 0.5, 0.7, 0.9]
epsilon_values = [0, 0.01, 0.1,
                  0.2, 1]
max_pairs_per_instance = 5
seeds = [15]

learning_rates = [0.01, 0.001]
batch_sizes = [128]
es_patiences = [64]
es_intervals = [8]
es_val_ratios = [0.3]

splits = [4]

params = [scenarios, lambda_values, epsilon_values, splits, seeds, learning_rates,
          batch_sizes, es_patiences, es_intervals, es_val_ratios]

param_product = list(product(*params))

for scenario_name, lambda_value, epsilon_value, split, seed, learning_rate, batch_size, es_patience, es_interval, es_val_ratio in param_product:
    params_string = "-".join([scenario_name,
                              str(lambda_value), str(epsilon_value), str(split), str(seed), str(learning_rate), str(es_interval), str(es_patience), str(es_val_ratio), str(batch_size)])

    filename = "nn_hinge" + "-" + params_string + ".csv"
    loss_filename = "nn_hinge" + "-" + params_string + "-losses.csv"
    filepath = result_path + filename
    loss_filepath = result_path + loss_filename
    print(loss_filepath)
    figure_file = params_string + "-losses" + ".pdf"
    try:
        df = pd.read_csv(loss_filepath)
    except:
        print("File " + loss_filename + " not found")
        continue
    df["$\lambda$ MSE"] = lambda_value * df["MSE"]
    df["$(1 - \lambda)$ SQH"] = (1 - lambda_value) * df["SQH"]
    df["TOTAL_LOSS"] = df["$\lambda$ MSE"] + df["$(1 - \lambda)$ SQH"]
    df = df.melt(id_vars=["iter"])
    print(df.head())
    text = "$\lambda$ = " + str(lambda_value) + ", "
    text += "$\epsilon$ = " + str(epsilon_value) + ", "
    text += "split = " + str(split) + ", "
    text += "seed = " + str(seed) + ",\n"
    text += "learning rate = " + str(learning_rate) + ", "
    text += "es patience = " + str(es_patience) + ", "
    text += "es interval = " + \
        str(es_interval) + ", "
    text += "es val raio = " + str(es_val_ratio) + ",\n"
    text += "batch size = " + str(batch_size)
    plt.clf()
    # plt.tight_layout()
    plt.annotate(text, (0, 0), (0, -40), xycoords="axes fraction",
                 textcoords="offset points", va="top")
    lp = sns.lineplot(x="iter", y="value", hue="variable", data=df)
    plt.title(scenario_name)
    plt.savefig(figures_path+figure_file, bbox_inches="tight")
    filename = "nn_hinge" + "-" + params_string + ".csv"
    loss_filename = "nn_hinge" + "-" + params_string + "-es-val.csv"
    filepath = result_path + filename
    loss_filepath = result_path + loss_filename
    print(loss_filepath)
    figure_file = params_string + "es-val" + ".pdf"
    try:
        df = pd.read_csv(loss_filepath)
    except:
        print("File " + loss_filename + " not found")
        continue

    df = df.melt(id_vars=["es_call"])
    print(df.head())
    text = "$\lambda$ = " + str(lambda_value) + ", "
    text += "$\epsilon$ = " + str(epsilon_value) + ", "
    text += "split = " + str(split) + ", "
    text += "seed = " + str(seed) + ",\n"
    text += "learning rate = " + str(learning_rate) + ", "
    text += "es patience = " + str(es_patience) + ", "
    text += "es interval = " + \
        str(es_interval) + ", "
    text += "es val raio = " + str(es_val_ratio) + ",\n"
    text += "batch size = " + str(batch_size)
    plt.show()
    plt.clf()
    # plt.tight_layout()
    plt.annotate(text, (0, 0), (0, -40), xycoords="axes fraction",
                 textcoords="offset points", va="top")
    lp = sns.lineplot(x="es_call", y="value", hue="variable", data=df)
    plt.title(scenario_name)
    # plt.savefig(figures_path+figure_file, bbox_inches="tight")
