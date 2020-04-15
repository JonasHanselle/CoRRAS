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

figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/plots/hinge/"
scenario_names = ["MIP-2016", "SAT11-INDU", "CSP-2010"]

# lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambda_values = [0.5]
epsilon_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
max_pairs_per_instance = 5
maxiter = 1000
seeds = [15]
use_quadratic_transform_values = [False, True]
use_max_inverse_transform_values = ["None"]
scale_target_to_unit_interval_values = [True]
skip_censored_values = [False]
regulerization_params_values = [0.001]
use_weighted_samples_values = [False]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
seed = 15

params = [
    scenario_names, use_max_inverse_transform_values,
    scale_target_to_unit_interval_values, skip_censored_values,
    regulerization_params_values, use_weighted_samples_values, lambda_values
]

name_map = {
    "ndcg": "NDCG",
    "tau_corr": "Kendall $\\tau_b$",
    "tau_p": "Kendall $\\tau_b$ p-value",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "par10": "PAR10",
    "abs_distance_to_vbs": "MP",
    "success_rate": "SR"
}

param_product = list(product(*params))

# measures = ["tau_corr", "ndcg", "mae", "mse", "rmse"]
measures = ["par10", "abs_distance_to_vbs", "success_rate"]
measures = [
    "par10", "abs_distance_to_vbs", "success_rate", "success_rate", "tau_corr",
    "ndcg", "mae", "mse", "rmse"
]

for measure in measures:
    plt.clf()
    fig, axes = plt.subplots(1, 3)
    for index, (scenario_name, use_max_inverse_transform,
                scale_target_to_unit_interval, skip_censored,
                regulerization_param,
                use_weighted_samples, lambda_value) in enumerate(param_product):

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
            str(use_max_inverse_transform),
            str(scale_target_to_unit_interval)
        ])

        # continue
        try:
            # df_corras = pd.read_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")
            corras = pd.read_csv(evaluations_path + "corras-hinge-linear-" +
                                 scenario_name + "-new-weights-eps.csv")

            corras["lambda"] = 1.0 - corras["lambda"]

            # print(corras.head())
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")
            continue
        current_frame = corras.loc[
            (corras["seed"] == seed) &
            (corras["scale_to_unit_interval"] == scale_target_to_unit_interval)
            & (corras["max_inverse_transform"] == use_max_inverse_transform)
            & (corras["use_weighted_samples"] == use_weighted_samples)
            & (corras["regularization_param"] == regulerization_param)
            & (corras["lambda"] == lambda_value)]

        if measure == "success_rate":
            val_rf = df_baseline_rf["run_status"].value_counts(
                normalize=True)["ok"]
            val_lr = df_baseline_lr["run_status"].value_counts(
                normalize=True)["ok"]
            val_label_ranking = df_baseline_label_ranking["run_status"].value_counts(
                normalize=True)["ok"]
            epsilons = list(current_frame["epsilon"].unique())
            results = []
            for epsilon in epsilons:
                for quadratic_transform in [True, False]:
                    epsilon_frame = current_frame.loc[
                        (corras["epsilon"] == epsilon) &
                        (corras["quadratic_transform"] == quadratic_transform)]
                    try:
                        # print(lambd_frame["run_status"].value_counts(
                        #     normalize=False))
                        results.append([
                            epsilon, quadratic_transform,
                            epsilon_frame["run_status"].value_counts(
                                normalize=True)["ok"]
                        ])
                    except:
                        results.append([epsilon, quadratic_transform, 0.0])

            results_frame = pd.DataFrame(
                data=results,
                columns=["epsilon", "quadratic_transform", "success_rate"])

            # print(results_frame)
            lp = sns.lineplot(x="epsilon",
                              y=measure,
                              marker="o",
                              markersize=8,
                              hue="quadratic_transform",
                              data=results_frame,
                              ax=ax,
                              legend=None,
                              ci=None)
            lp.axes.axhline(val_rf, c="g", ls="--", label="rf-baseline-mean")
            lp.axes.axhline(val_lr, c="m", ls="--", label="lr-baseline-mean")
            if measure not in ["rmse", "mse", "mae"]:
                lp.axes.axhline(val_label_ranking, c="brown", ls="--", label="label-ranking-baseline-mean")
            ax.set_title(scenario_name)
            ax.set_ylabel(name_map[measure])
            ax.set_xlabel("$\\epsilon$")
            # fig.set_size_inches(10.5, 3.0)
            # fig.tight_layout()
            # labels = ["PL-GLM", "PL-QM", "Random Forest", "Linear Regression"]
            # legend = fig.legend(list(axes), labels=labels, loc="lower center", ncol=len(
            #     labels), bbox_to_anchor=(0.5, -0.02))
            # plt.savefig(fname=figures_path + "-".join(scenario_names) + "-" + params_string.replace(
            #     ".", "_") + "-" + measure + ".pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
            continue

        current_frame["rmse"] = current_frame["mse"].pow(1. / 2)
        print(current_frame.head())
        print(current_frame.columns)
        # print(current_frame["use_weighted_samples"].value_counts())
        print(current_frame["lambda"].value_counts())
        df_baseline_rf["rmse"] = df_baseline_rf["mse"].pow(1. / 2)
        df_baseline_lr["rmse"] = df_baseline_lr["mse"].pow(1. / 2)
        df_baseline_label_ranking["rmse"] = df_baseline_label_ranking["mse"].pow(1. / 2)

        if measure in ["mae", "mse", "rmse"]:
            ax.set_yscale("log")
            current_frame = current_frame.loc[(current_frame["lambda"] <=
                                               0.99)]
    #     print(current_frame[:])
    #     print(current_frame.iloc[:10,8:12].to_latex(na_rep="-", index=False, bold_rows=True, float_format="%.2f", formatters={"tau_corr" : max_formatter}, escape=False))
    #     for measure in current_frame.columns[8:]:
    #         plt.clf()
    # bp = sns.boxplot(x="lambda", y=measure, hue="epsilon", data=df_corras)
    # bp = sns.boxplot(x="lambda", y=measure, data=df_corras)
    # if df_baseline is not None:
    #     bp.axes.axhline(df_baseline[measure].mean(), c="g", ls="--", label="rf-baseline-mean")
    # plt.title(scenario_name)
    # plt.legend()
    # plt.savefig(figures_path+scenario_name+"-" + measure +"-boxplot.pdf")
    # print("length of current frame", len(current_frame))
    # print("columns", current_frame.columns[8:])
    # plt.clf()
    # bp = sns.lineplot(x="lambda", y=measure, hue="epsilon", data=df_corras, palette=sns.color_palette("Set1", len(pd.unique(df_corras["epsilon"]))))
    # g = sns.FacetGrid(df_corras, col="max_inverse_transform")
    # g.map(sns.lineplot, "lambda", measure)
        if measure in ["par10", "abs_distance_to_vbs"]:
            lp = sns.lineplot(x="epsilon",
                              y=measure,
                              marker="o",
                              markersize=8,
                              hue="quadratic_transform",
                              data=current_frame,
                              ax=ax,
                              legend=None,
                              ci=None)
        else:
            lp = sns.lineplot(x="epsilon",
                              y=measure,
                              marker="o",
                              markersize=8,
                              hue="quadratic_transform",
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
        ax.set_xlabel("$\\epsilon$")
        # ax.set_aspect(5.5, adjustable="box")
        # ax.legend()
#         plt.savefig(figures_path + scenario_name + "-" + params_string.replace(".","_") + "-" + measure + "-lineplot-mi.pdf")
    fig.set_size_inches(10.5, 3.0)
    # plt.subplots_adjust(right=0.85)
    fig.tight_layout()
    if measure in ["rmse", "mse", "mae"]:
        labels = ["Hinge-LM", "Hinge-QM", "Random Forest", "Linear Regression"]
    else:
        labels = ["Hinge-LM", "Hinge-QM", "Random Forest", "Linear Regression", "Label Ranking"]
    
    legend = fig.legend(list(axes),
                        labels=labels,
                        loc="lower center",
                        ncol=len(labels),
                        bbox_to_anchor=(0.5, -0.02))
    plt.savefig(fname=figures_path + "-".join(scenario_names) + "-" +
                params_string.replace(".", "_") + "-" + measure + "-eps.pdf",
                bbox_extra_artists=(legend, ),
                bbox_inches="tight")
    os.system("pdfcrop " + figures_path + "-".join(scenario_names) + "-" +
              params_string.replace(".", "_") + "-" + measure + "-eps.pdf " +
              figures_path + "-".join(scenario_names) + "-" +
              params_string.replace(".", "_") + "-" + measure + "-eps.pdf")
    # plt.show()
