
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

figures_path = "../Masters_Thesis/Thesis/latex-thesis-template/gfx/plots/hinge/"
scenario_names = ["CSP-Minizinc-Time-2016", "MIP-2016", "CSP-2010"]
use_quadratic_transform_values = [True, False]
use_max_inverse_transform_values = ["max_cutoff"]
# scale_target_to_unit_interval_values = [True, False]
scale_target_to_unit_interval_values = [True]
seed = 15

params = [scenario_names, use_max_inverse_transform_values,
          scale_target_to_unit_interval_values]

name_map = {"ndcg": "NDCG",
            "tau_corr": "Kendall $\\tau_b$",
            "tau_p": "Kendall $\\tau_b$ p-value",
            "mae": "MAE",
            "mse": "MSE",
            "par10": "PAR 10",
            "absolute_distance_to_vbs": "mp",
            "success_rate": "sr"}

param_product = list(product(*params))

measures = ["tau_corr", "ndcg", "mae", "mse"]


for measure in measures:
    plt.clf()
    fig, axes = plt.subplots(1, 3)
    for index, (scenario_name, use_max_inverse_transform, scale_target_to_unit_interval) in enumerate(param_product):
        ax = axes[index]
        df_baseline_lr = None
        df_baseline_rf = None
        try:
            df_baseline_lr = pd.read_csv(
                evaluations_path + "baseline-evaluation-linear-regression" + scenario_name + ".csv")
            df_baseline_rf = pd.read_csv(
                evaluations_path + "baseline-evaluation-random_forest" + scenario_name + ".csv")
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")

        params_string = "-".join([scenario_name, str(use_max_inverse_transform),
                                  str(scale_target_to_unit_interval)])

        # continue
        try:
            # df_corras = pd.read_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")
            corras = pd.read_csv(
                evaluations_path + "corras-pl-log-linear-" + scenario_name + ".csv")
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")
            continue
        current_frame = corras.loc[(corras["seed"] == seed) & (corras["scale_to_unit_interval"]
                                                               == scale_target_to_unit_interval) & (corras["max_inverse_transform"] == "max_cutoff")]
        if measure in ["mae","mse"]:
            current_frame = current_frame.loc[(current_frame["lambda"] <= 0.99) ]

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
        lp = sns.lineplot(x="lambda", y=measure, marker="o", markersize=8,
                          hue="quadratic_transform", data=current_frame, ax=ax, legend=None)
        if df_baseline_rf is not None:
            lp.axes.axhline(df_baseline_rf[measure].mean(
            ), c="g", ls="--", label="rf-baseline-mean")
        if df_baseline_lr is not None:
            lp.axes.axhline(df_baseline_lr[measure].mean(
            ), c="m", ls="--", label="lr-baseline-mean")
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
    legend = fig.legend(list(axes), labels=labels, loc="lower center", ncol=len(
        labels), bbox_to_anchor=(0.5, -0.02))
    plt.savefig(fname=figures_path + "-".join(scenario_names) + "-" + params_string.replace(
        ".", "_") + "-" + measure + ".pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
    # plt.show()
