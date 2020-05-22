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
evaluations_path = "./evaluations/"

figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/plots/pl_nn/"


scenarios = [
    # "CPMP-2015",
    "MIP-2016",
    # "CSP-2010",
    # "SAT11-HAND",
    # "SAT11-INDU",
    # "SAT11-RAND",
    # "CSP-Minizinc-Time-2016",
    # "MAXSAT-WPMS-2016",
    # "MAXSAT-PMS-2016",
    # "QBF-2016"
]

# scenarios = ["CPMP-2015", "SAT11-RAND", "MIP-2016", "QBF-2016", "MAXSAT-WPMS-2016", "MAXSAT-PMS-2016"]

lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lambda_values = [0.5,1.0]
max_pairs_per_instance = 5
maxiter = 1000
seeds = [15]

learning_rates = [0.001]
batch_sizes = [128]
es_patiences = [8]
es_intervals = [8]
es_val_ratios = [0.3]
layer_sizes_vals = ["[32]"]
activation_functions = ["sigmoid"]
use_max_inverse_transform_values = ["max_cutoff"]
scale_target_to_unit_interval_values = [True]
use_weighted_samples_values = [False]

splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
splits = [1, 2]

params = [
    scenarios, learning_rates, seeds, batch_sizes, es_patiences, es_intervals,
    es_val_ratios, layer_sizes_vals, activation_functions
]

param_product = list(product(*params))
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

measures = [
    "tau_corr", "ndcg", "mae", "mse", "par10", "abs_distance_to_vbs",
    "success_rate", "rmse"
]

for measure in measures:
    plt.clf()
    fig, axes = plt.subplots(1, 6)
    print(len(param_product))
    for index, (scenario_name, learning_rate, seed, batch_size, es_patience,
                es_interval, es_val_ratio, layer_sizes,
                activation_function) in enumerate(param_product):

        ax = axes[index]
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
            str(learning_rate),
            str(batch_size),
            str(es_patience),
            str(es_interval),
            str(es_val_ratio)
        ])

        # continue
        try:
            # df_corras = pd.read_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")
            corras = pd.read_csv(evaluations_path + "corras-pl-nn-" +
                                 scenario_name + "-ki.csv")
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")
            continue
        print(corras.head())
        current_frame = corras.loc[
            (corras["seed"] == seed)
            & (corras["learning_rate"] == learning_rate)
            & (corras["batch_size"] == batch_size) &
            (corras["es_patience"] == es_patience) &
            (corras["es_interval"] == es_interval) &
            (corras["layer_sizes"] == layer_sizes) &
            (corras["activation_function"] == activation_function)]

        if(len(current_frame) != 11*len(df_baseline_lr)):
            print(f"LEN OF {scenario_name} IS {len(current_frame)} BUT SHOULD BE {len(df_baseline_lr)}")

        if measure == "success_rate":
            val_rf = df_baseline_rf["run_status"].value_counts(
                normalize=True)["ok"]
            val_lr = df_baseline_lr["run_status"].value_counts(
                normalize=True)["ok"]
            val_label_ranking = df_baseline_label_ranking[
                "run_status"].value_counts(normalize=True)["ok"]
            lambdas = list(current_frame["lambda"].unique())
            results = []
            for lambd in lambdas:
                for use_weighted_samples in [True, False]:
                    lambd_frame = current_frame.loc[
                        (corras["lambda"] == lambd)
                        & (corras["use_weighted_samples"] ==
                           use_weighted_samples)]
                    try:
                        print(lambd_frame["run_status"].value_counts(
                            normalize=False))
                        results.append([
                            lambd, use_weighted_samples,
                            lambd_frame["run_status"].value_counts(
                                normalize=True)["ok"]
                        ])
                    except:
                        results.append([lambd, use_weighted_samples, 0.0])

            results_frame = pd.DataFrame(
                data=results,
                columns=["lambda", "use_weighted_samples", "success_rate"])

            print(results_frame)
            lp = sns.lineplot(x="lambda",
                              y=measure,
                              marker="o",
                              markersize=8,
                              hue="use_weighted_samples",
                              data=results_frame,
                              ax=ax,
                              legend=None,
                              ci=None)
            lp.axes.axhline(val_rf, c="g", ls="--", label="rf-baseline-mean")
            lp.axes.axhline(val_lr, c="m", ls="--", label="lr-baseline-mean")
            if measure not in ["rmse", "mse", "mae"]:
                lp.axes.axhline(val_label_ranking,
                                c="brown",
                                ls="--",
                                label="label-ranking-baseline-mean")
            ax.set_title(scenario_name)
            ax.set_ylabel(name_map[measure])
            ax.set_xlabel("$\\lambda$")
            # fig.set_size_inches(10.5, 3.0)
            # fig.tight_layout()
            # labels = ["PL-GLM", "PL-QM", "Random Forest", "Linear Regression"]
            # legend = fig.legend(list(axes), labels=labels, loc="lower center", ncol=len(
            #     labels), bbox_to_anchor=(0.5, -0.02))
            # plt.savefig(fname=figures_path + "-".join(scenarios) + "-" + params_string.replace(
            #     ".", "_") + "-" + measure + ".pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
            continue
        current_frame["rmse"] = current_frame["mse"].pow(1. / 2)
        print(current_frame.head())
        df_baseline_rf["rmse"] = df_baseline_rf["mse"].pow(1. / 2)
        df_baseline_lr["rmse"] = df_baseline_lr["mse"].pow(1. / 2)
        df_baseline_label_ranking["rmse"] = df_baseline_label_ranking[
            "mse"].pow(1. / 2)

        if measure in ["mae", "mse", "rmse"]:
            ax.set_yscale("log")
            current_frame = current_frame.loc[(current_frame["lambda"] <=
                                               0.99)]

        if measure in ["par10", "abs_distance_to_vbs"]:
            lp = sns.lineplot(x="lambda",
                              y=measure,
                              marker="o",
                              markersize=8,
                              hue="use_weighted_samples",
                              data=current_frame,
                              ax=ax,
                              legend=None,
                              ci=None)
        else:
            lp = sns.lineplot(x="lambda",
                              y=measure,
                              marker="o",
                              markersize=8,
                              hue="use_weighted_samples",
                              data=current_frame,
                              ax=ax,
                              legend=None,
                              ci=95)
        if df_baseline_rf is not None:
            lp.axes.axhline(df_baseline_rf[measure].mean(),
                            c="g",
                            ls="--",
                            label="rf-baseline-mean")
        if df_baseline_lr is not None:
            lp.axes.axhline(df_baseline_lr[measure].mean(),
                            c="m",
                            ls="--",
                            label="lr-baseline-mean")
        if measure not in ["rmse", "mse", "mae"]:
            if df_baseline_label_ranking is not None:
                lp.axes.axhline(df_baseline_label_ranking[measure].mean(),
                                c="brown",
                                ls="--",
                                label="label-ranking-baseline-mean")

        ax.set_title(scenario_name)
        ax.set_ylabel(name_map[measure])
        ax.set_xlabel("$\\lambda$")
        # ax.set_aspect(5.5, adjustable="box")
        # ax.legend()
#         plt.savefig(figures_path + scenario_name + "-" + params_string.replace(".","_") + "-" + measure + "-lineplot-mi.pdf")
    fig.set_size_inches(10.5, 3.0)
    # plt.subplots_adjust(right=0.85)
    fig.tight_layout()
    if measure in ["rmse", "mse", "mae"]:
        labels = [
            "PL-NN unweighted", "PL-NN weighted", "Random Forest",
            "Linear Regression"
        ]
    else:
        labels = [
            "PL-NN unweighted", "PL-NN weighted", "Random Forest",
            "Linear Regression", "Label Ranking"
        ]

    legend = fig.legend(list(axes),
                        labels=labels,
                        loc="lower center",
                        ncol=len(labels),
                        bbox_to_anchor=(0.5, -0.02))

    plt.show()
    
    # plt.savefig(fname=figures_path + "-".join(scenarios) + "-" +
    #             params_string.replace(".", "_") + "-" + measure + ".pdf",
    #             bbox_extra_artists=(legend, ),
    #             bbox_inches="tight")

    # os.system("pdfcrop " + figures_path + "-".join(scenarios) + "-" +
    #           params_string.replace(".", "_") + "-" + measure + ".pdf " +
    #           figures_path + "-".join(scenarios) + "-" +
    #           params_string.replace(".", "_") + "-" + measure + ".pdf")
    # plt.show()
