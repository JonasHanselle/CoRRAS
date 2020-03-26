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
scenario_names = ["SAT11-HAND", "MIP-2016", "CSP-2010"]

lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lambda_values = [0.5]
epsilon_values = [1.0]
max_pairs_per_instance = 5
maxiter = 1000
seeds = [15]
use_quadratic_transform_values = [False, True]
use_max_inverse_transform_values = ["None"]
scale_target_to_unit_interval_values = [True]
skip_censored_values = [False]
regulerization_params_values = [0.1]
use_weighted_samples_values = [False]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
seed = 15

params = [
    scenario_names, use_max_inverse_transform_values,
    scale_target_to_unit_interval_values, skip_censored_values,
    regulerization_params_values, use_weighted_samples_values
]

name_map = {
    "ndcg": "NDCG",
    "tau_corr": "Kendall $\\tau_b$",
    "tau_p": "Kendall $\\tau_b$ p-value",
    "mae": "MAE",
    "mse": "MSE",
    "par10": "PAR 10",
    "absolute_distance_to_vbs": "mp",
    "success_rate": "sr"
}

param_product = list(product(*params))

measures = ["tau_corr", "ndcg", "mae", "mse"]

for measure in measures:
    plt.clf()
    fig, axes = plt.subplots(1, 3)
    for index, (scenario_name, use_max_inverse_transform,
                scale_target_to_unit_interval, skip_censored,
                regulerization_param,
                use_weighted_samples) in enumerate(param_product):

        ax = axes[index]
        df_baseline_lr = None
        df_baseline_rf = None
        try:
            df_baseline_lr = pd.read_csv(
                evaluations_path + "baseline-evaluation-linear-regression" +
                scenario_name + ".csv")
            df_baseline_rf = pd.read_csv(evaluations_path +
                                         "baseline-evaluation-random_forest" +
                                         scenario_name + ".csv")
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
                                 scenario_name + "-new.csv")
            # print(corras.head())
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")
            continue
        # print(corras.columns)
        current_frame = corras.loc[
            (corras["seed"] == seed)
            & (corras["max_inverse_transform"] == use_max_inverse_transform) &
            (corras["scale_to_unit_interval"] == scale_target_to_unit_interval)
            & (corras["skip_censored"] == skip_censored) &
            (corras["regularization_param"] == regulerization_param) &
            (corras["use_weighted_samples"] == use_weighted_samples)]

        # print("current Frame", current_frame.head())
        # if len(current_frame) != len(df_baseline_lr):
        # print("lengths not equal", len(current_frame) , len(df_baseline_lr))

        for lambda_value in lambda_values:
            for use_quadratic_transform in use_quadratic_transform_values:
                sub_frame = current_frame.loc[
                    (current_frame["quadratic_transform"] ==
                     use_quadratic_transform)
                    & (current_frame["lambda"] == lambda_value)]
                print(scenario_name, use_max_inverse_transform, scale_target_to_unit_interval,
                      skip_censored, regulerization_param,
                      use_weighted_samples, use_quadratic_transform,
                      lambda_value, len(sub_frame), len(current_frame),
                      len(df_baseline_lr), len(df_baseline_rf))
        # print("sub frame")
        # print(sub_frame.head())

        if measure in ["mae", "mse"]:
            current_frame = current_frame.loc[(current_frame["lambda"] <=
                                               0.99)]
        lp = sns.lineplot(x="lambda",
                          y=measure,
                          marker="o",
                          markersize=8,
                          hue="quadratic_transform",
                          data=current_frame,
                          ax=ax,
                          legend=None)
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
        ax.set_title(scenario_name)
        ax.set_ylabel(name_map[measure])
        ax.set_xlabel("$\\lambda$")
        # ax.set_aspect(5.5, adjustable="box")
        # ax.legend()
#         plt.savefig(figures_path + scenario_name + "-" + params_string.replace(".","_") + "-" + measure + "-lineplot-mi.pdf")
    fig.set_size_inches(10.5, 3.0)
    # plt.subplots_adjust(right=0.85)
    fig.tight_layout()
    labels = ["PL-LM", "PL-QM", "Random Forest", "Linear Regression"]
    legend = fig.legend(list(axes),
                        labels=labels,
                        loc="lower center",
                        ncol=len(labels),
                        bbox_to_anchor=(0.5, -0.02))
    plt.savefig(fname=figures_path + "-".join(scenario_names) + "-" +
                params_string.replace(".", "_") + "-" + measure + ".pdf",
                bbox_extra_artists=(legend, ),
                bbox_inches="tight")
    # plt.show()
