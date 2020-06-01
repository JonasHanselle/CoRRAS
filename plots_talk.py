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
    # "CSP-Minizinc-Time-2016",
    # "MAXSAT-WPMS-2016",
    # "MAXSAT-PMS-2016",
    # "QBF-2016"
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

# measures = ["tau_corr", "ndcg", "mae", "mse", "rmse"]
measures = ["par10", "abs_distance_to_vbs", "success_rate"]
measures = [
    "par10", "tau_corr", "mae", "mse"
]

measures = [
    "tau_corr", "par10"
]


# measures = ["par10", "mae", "tau_corr"]

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

        # continue
        try:
            # df_corras = pd.read_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")
            corras_linhinge = pd.read_csv(evaluations_path +
                                            "ki2020-linhinge-" +
                                            scenario_name + ".csv")
            corras_plnn = pd.read_csv(evaluations_path + "ki2020-plnet-" +
                                        scenario_name + ".csv")
            corras_linpl = pd.read_csv(evaluations_path +
                                        "ki2020_linpl-" +
                                        scenario_name + ".csv")
            corras_hinge_nn = corras = pd.read_csv(evaluations_path +
                                                    "ki2020-nnh-" +
                                                    scenario_name + ".csv")
            print(f"lengths {len(corras_linhinge)} {len(corras_hinge_nn)} {len(corras_plnn)} {len(corras_linpl)}")

            # for lambda_val in lambda_values:
            #     current_frame = corras_linhinge[corras_linhinge["lambda"] == lambda_val]
            #     print("linhinge", lambda_val, current_frame.count())
            #     current_frame = corras_plnn[corras_plnn["lambda"] == lambda_val]
            #     print("plnn", lambda_val, current_frame.count())
            #     current_frame = corras_hinge_nn[corras_hinge_nn["lambda"] == lambda_val]
            #     print("hnn", lambda_val, current_frame.count())
            #     current_frame = corras_linpl[corras_linpl["lambda"] == lambda_val]
            #     print("linpl", lambda_val, current_frame.count())
            # continue
        # continue
            corras_linhinge = corras_linhinge.groupby(["lambda", "seed", "quadratic_transform"]).agg("mean").reset_index()

            corras_plnn = corras_plnn.groupby(["lambda", "seed"]).agg("mean").reset_index()
            corras_hinge_nn = corras_hinge_nn.groupby(["lambda", "seed"]).agg("mean").reset_index()

            corras_linpl = corras_linpl.groupby(["lambda", "seed", "quadratic_transform"]).agg("mean").reset_index()

            # corras_linhinge = corras_linhinge.loc[
            #     corras_linhinge["use_weighted_samples"] == False]
            # corras_hinge_nn = corras_hinge_nn.loc[
            #     corras_hinge_nn["use_weighted_samples"] == False]
            # corras_plnn = corras_plnn.loc[corras_plnn["use_weighted_samples"]
            #                               == False]
            # corras_linpl = corras_linpl.loc[
            #     corras_linpl["use_weighted_samples"] == False]


            corras_linhinge["Approach"] = "Hinge-LM"
            print(corras_linhinge.head())
            corras_linpl["Approach"] = "PL-LM"
            corras_plnn["Approach"] = "PL-NN"
            corras_hinge_nn["Approach"] = "Hinge-NN"

            # print(corras_linhinge.loc[corras_linhinge["quadratic_transform"] ==
            #                     True]["Approach"])

            print(corras_linhinge.columns)
            corras_linhinge.loc[corras_linhinge["quadratic_transform"] == True,
                                "Approach"] = "Hinge-QM"
            corras_linpl.loc[corras_linpl["quadratic_transform"] == True,
                                "Approach"] = "PL-QM"
            print(corras_linpl.loc[corras_linpl["Approach"] == "PL-QM"])

            # df_all = corras_linhinge[[
            #     "approach", "lambda", "mae", "par10", "tau_corr",
            #     "problem_instance"
            # ]]
            # df_all = df_all.concat(corras_plnn[[
            #     "approach", "lambda", "mae", "par10", "tau_corr",
            #     "problem_instance"
            # ]])
            # df_all = df_all.concat(corras_linpl[[
            #     "approach", "lambda", "mae", "par10", "tau_corr",
            #     "problem_instance"
            # ]])
            # df_all = df_all.concat(corras_hinge_nn[[
            #     "approach", "lambda", "mae", "par10", "tau_corr",
            #     "problem_instance"
            # ]])
            # print(df_all)

            frames = [
                corras_hinge_nn[[
                    "Approach", "lambda", "mae", "par10", "tau_corr",
                    
                ]], corras_linpl[[
                    "Approach", "lambda", "mae", "par10", "tau_corr",
                    
                ]], corras_plnn[[
                    "Approach", "lambda", "mae", "par10", "tau_corr",
                    
                ]], corras_linhinge[[
                    "Approach", "lambda", "mae", "par10", "tau_corr",
                    
                ]]
            ]

            df_all = pd.concat(frames)

            # print(corras_linhinge)
            # print(corras_linpl)

            # print(len(corras_linhinge), len(corras_hinge_nn), len(corras_plnn),
            #       len(corras_linpl), len(df_baseline_lr))

            # print(corras.head())
        except Exception as ex:
            print("Scenario " + scenario_name +
                " not found in corras evaluation dataaaaaaaa!" + str(ex))
            continue
        # current_frame = corras.loc[
        #     (corras["seed"] == seed) &
        #     (corras["scale_to_unit_interval"] == scale_target_to_unit_interval)
        #     & (corras["max_inverse_transform"] == use_max_inverse_transform)
        #     & (corras["use_weighted_samples"] == use_weighted_samples)
        #     & (corras["regularization_param"] == regulerization_param)]
        # labels = list(df_all.Approach.unique())
        # labels = sorted(labels)
        labels = ["PL-LM","PL-QM","PL-NN","Hinge-LM","Hinge-QM","Hinge-NN"]
        if measure in ["mae", "mse", "rmse"]:
            ax.set_yscale("log")
            df_all = df_all.loc[(df_all["lambda"] <= 0.99)]
            # corras_linhinge = corras_linhinge.loc[(corras_linhinge["lambda"] <=
            #                                    0.99)]
            # corras_plnn = corras_plnn.loc[(corras_plnn["lambda"] <=
            #                                    0.99)]
            # corras_hinge_nn = corras_hinge_nn.loc[(corras_hinge_nn["lambda"] <=
            #                                    0.99)]
            # corras_linpl = corras_linpl.loc[(corras_linpl["lambda"] <=
            #                                    0.99)]
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
            # if measure in ["par10", "abs_distance_to_vbs"]:
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
        # if df_baseline_rf is not None:
        #     lp.axes.axhline(df_baseline_rf[measure].mean(),
        #                     c="g",
        #                     ls="--",
        #                     label="Random Forest")
        # ax.axhline(df_baseline_rf[measure].mean(),
        #                 c="g",
        #                 ls="--",
        #                 label="Random Forest")
        # if df_baseline_lr is not None:
        #     lp.axes.axhline(df_baseline_lr[measure].mean(),
        #                     c="m",
        #                     ls="--",
        #                     label="lr-baseline-mean")
        # if measure not in ["rmse", "mse", "mae"]:
        #     if df_baseline_label_ranking is not None:
        #         lp.axes.axhline(df_baseline_label_ranking[measure].mean(),
        #                         c="brown",
        #                         ls="--",
        #                         label="Label Ranking")

        ax.set_title(scenario_name)
        ax.set_ylabel(name_map[measure])
        ax.set_xlabel("$\\lambda$")
        # ax.set_aspect(5.5, adjustable="box")
        # ax.legend()
#         plt.savefig(figures_path + scenario_name + "-" + params_string.replace(".","_") + "-" + measure + "-lineplot-mi.pdf")
    fig.set_size_inches(10.5, 5.5)
    # if measure in ["rmse", "mse", "mae"]:
    plt.subplots_adjust(bottom=0.14, wspace=0.35, hspace=0.42)
    # labels = list(df_all.Approach.unique())
    # if measure in ["rmse", "mse", "mae"]:
    #     labels += ["Random Forest Regression"]
    # else:
    #     labels += ["Random Forest Regression", "Label Ranking"]

    legend = fig.legend(list(axes), labels=labels, loc="lower center", ncol=6)
    # plt.tight_layout(pad=7)

    # plt.show()

    plt.savefig(
        fname=figures_path + "-".join(scenario_names) + "-" + measure + ".pdf",
        # bbox_extra_artists=(legend, ),
        bbox_inches="tight")
    os.system("pdfcrop " + figures_path + "-".join(scenario_names) + "-" +
              measure + ".pdf " + figures_path + "-".join(scenario_names) + "-" + measure + ".pdf")
    # plt.show()
