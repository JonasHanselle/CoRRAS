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
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

sns.set_style("whitegrid")

scenario_path = "./aslib_data-aslib-v4.0/"
evaluations_path = "./evaluations/"

figures_path = "../Masters_Thesis/KI2020/figures/plots/"
scenario_names = [
    "SAT11-RAND",
    "CSP-2010",
    "SAT11-INDU",
    "MIP-2016",
    "SAT11-HAND",
    "CPMP-2015",
]

lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lambda_values = [0.5]
epsilon_values = [1.0]
max_pairs_per_instance = 5
maxiter = 1000
use_quadratic_transform_values = [False, True]
use_max_inverse_transform_values = ["None"]
scale_target_to_unit_interval_values = [True]
skip_censored_values = [False]
regulerization_params_values = [0.001]
use_weighted_samples_values = [False]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

params = [
    scenario_names, use_max_inverse_transform_values,
    scale_target_to_unit_interval_values, skip_censored_values,
    regulerization_params_values, use_weighted_samples_values
]

name_map = {
    "ndcg": "NDCG",
    "tau_corr": "Kendall $\\tau$",
    "tau_p": "Kendall $\\tau$ p-value",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "par10": "PAR10",
    "abs_distance_to_vbs": "MP",
    "success_rate": "SR"
}

param_product = list(product(*params))

measures = ["tau_corr", "par10"]


for measure in measures:
    plt.clf()
    fig, axes = plt.subplots(2, 3)
    for index, (scenario_name, use_max_inverse_transform,
                scale_target_to_unit_interval, skip_censored,
                regulerization_param,
                use_weighted_samples) in enumerate(param_product):

        ax = axes[index % 2, index // 2]
        df_baseline_lr = None
        df_baseline_rf = None
        df_baseline_label_ranking = None
        try:
            df_baseline_lr = pd.read_csv(
                evaluations_path + "baseline-evaluation-linear-regression" +
                scenario_name + ".csv")
            df_baseline_rf = pd.read_csv(evaluations_path +
                                         "baseline-evaluation-random_forest" +
                                         scenario_name + ".csv")
            df_baseline_label_ranking = pd.read_csv(evaluations_path +
                                                    "baseline-label-ranking-" +
                                                    scenario_name + ".csv")
            print("df baseline label ", len(df_baseline_label_ranking))
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")

        params_string = "-".join([
            scenario_name,
            str(use_max_inverse_transform),
            str(scale_target_to_unit_interval)
        ])

        try:
            corras_linhinge = pd.read_csv(evaluations_path +
                                          "ki2020-linhinge-" + scenario_name +
                                          ".csv")
            corras_plnn = pd.read_csv(evaluations_path + "ki2020-plnet-" +
                                      scenario_name + ".csv")
            corras_linpl = pd.read_csv(evaluations_path + "ki2020_linpl-" +
                                       scenario_name + ".csv")
            corras_hinge_nn = corras = pd.read_csv(evaluations_path +
                                                   "ki2020-nnh-" +
                                                   scenario_name + ".csv")
            print(
                f"lengths {len(corras_linhinge)} {len(corras_hinge_nn)} {len(corras_plnn)} {len(corras_linpl)}"
            )

            corras_linhinge = corras_linhinge.groupby(
                ["lambda", "seed",
                 "quadratic_transform"]).agg("mean").reset_index()

            corras_plnn = corras_plnn.groupby(["lambda", "seed"
                                               ]).agg("mean").reset_index()
            corras_hinge_nn = corras_hinge_nn.groupby(
                ["lambda", "seed"]).agg("mean").reset_index()

            corras_linpl = corras_linpl.groupby(
                ["lambda", "seed",
                 "quadratic_transform"]).agg("mean").reset_index()

            corras_linhinge["Approach"] = "Hinge-LM"
            print(corras_linhinge.head())
            corras_linpl["Approach"] = "PL-LM"
            corras_plnn["Approach"] = "PL-NN"
            corras_hinge_nn["Approach"] = "Hinge-NN"

            print(corras_linhinge.columns)
            corras_linhinge.loc[corras_linhinge["quadratic_transform"] == True,
                                "Approach"] = "Hinge-QM"
            corras_linpl.loc[corras_linpl["quadratic_transform"] == True,
                             "Approach"] = "PL-QM"
            print(corras_linpl.loc[corras_linpl["Approach"] == "PL-QM"])


            frames = [
                corras_hinge_nn[[
                    "Approach",
                    "lambda",
                    "mae",
                    "par10",
                    "tau_corr",
                ]], corras_linpl[[
                    "Approach",
                    "lambda",
                    "mae",
                    "par10",
                    "tau_corr",
                ]], corras_plnn[[
                    "Approach",
                    "lambda",
                    "mae",
                    "par10",
                    "tau_corr",
                ]], corras_linhinge[[
                    "Approach",
                    "lambda",
                    "mae",
                    "par10",
                    "tau_corr",
                ]]
            ]

            df_all = pd.concat(frames)

        except Exception as ex:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation dataaaaaaaa!" + str(ex))
            continue

        labels = [
            "PL-LM", "PL-QM", "PL-NN", "Hinge-LM", "Hinge-QM", "Hinge-NN"
        ]
        if measure in ["mae", "mse", "rmse"]:
            ax.set_yscale("log")
            df_all = df_all.loc[(df_all["lambda"] <= 0.99)]

            lp = sns.lineplot(x="lambda",
                              y=measure,
                              marker="o",
                              markersize=6,
                              data=df_all,
                              hue="Approach",
                              ax=ax,
                              legend="full",
                              ci=None,
                              hue_order=labels)
        else:
            lp = sns.lineplot(x="lambda",
                              y=measure,
                              marker="o",
                              markersize=6,
                              data=df_all,
                              hue="Approach",
                              ax=ax,
                              legend="full",
                              ci=None,
                              hue_order=labels)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.set_title(scenario_name)
        ax.set_ylabel(name_map[measure])
        ax.set_xlabel("$\\lambda$")
    fig.set_size_inches(10.5, 5.5)
    plt.subplots_adjust(bottom=0.14, wspace=0.35, hspace=0.42)

    legend = fig.legend(list(axes), labels=labels, loc="lower center", ncol=6)
    plt.show()
