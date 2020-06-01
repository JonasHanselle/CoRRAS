import sys
import os

import math
import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from itertools import product

# measures
from scipy.stats import kendalltau, describe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Corras.Evaluation.evaluation import ndcg_at_k, compute_relevance_scores_unit_interval

# plotting
import Orange
import matplotlib.pyplot as plt
import seaborn as sns

scenario_path = "./aslib_data-aslib-v4.0/"
evaluations_path = "./evaluations/"
# evaluations_path_nnh = "./evaluation-results/evaluations-nnh-config/"

figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/plots/talk/cd/"

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

use_quadratic_transform_values = [True, False]
use_max_inverse_transform_values = ["max_cutoff"]
# scale_target_to_unit_interval_values = [True, False]
scale_target_to_unit_interval_values = [True, False]
seed = 15

params = [
    scenario_names, use_quadratic_transform_values,
    use_max_inverse_transform_values, scale_target_to_unit_interval_values
]

param_product = list(product(*params))


def create_latex_max(df: pd.DataFrame,
                     decimal_format="{:10.3f}",
                     skip_max=2,
                     caption="\\TODO{caption}"):
    result = "\\begin{table}[H] \n"
    result += "\\centering \n"
    result += "\\begin{adjustbox}{max width=\\textwidth} \n"
    result += "\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "} \n"
    result += "\\toprule \n"
    result += " & ".join(df.columns) + " \\\\ \n"
    result += "\\midrule \n"
    for index, row in df.iterrows():
        result += row[0] + " & " + " & ".join([
            "\\textbf{" + decimal_format.format(x) +
            "}" if x == np.nanmax(row[skip_max:].to_numpy().astype("float64"))
            else decimal_format.format(x) for x in row[1:]
        ]) + " \\\\ \n"
    result += "\\toprule \n"
    result += "\\end{tabular} \n"
    result += "\\end{adjustbox} \n"
    result += f"\\caption{{{caption}}} \n"
    result += "\\label{tab:table_label} \n"
    result += "\\end{table} \n"

    result = result.replace("nan", "-")
    print(result)


def create_latex_min(df: pd.DataFrame,
                     decimal_format="{:10.3f}",
                     skip_min=2,
                     caption="\\TODO{caption}"):
    result = "\\begin{table}[H] \n"
    result += "\\centering \n"
    result += "\\begin{adjustbox}{max width=\\textwidth} \n"
    result += "\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "} \n"
    result += "\\toprule \n"
    result += " & ".join(df.columns) + " \\\\ \n"
    result += "\\midrule \n"
    for index, row in df.iterrows():
        result += row[0] + " & " + " & ".join([
            "\\textbf{" + decimal_format.format(x) +
            "}" if x == np.nanmin(row[skip_min:].to_numpy().astype("float64"))
            else decimal_format.format(x) for x in row[1:]
        ]) + " \\\\ \n"
    result += "\\toprule \n"
    result += "\\end{tabular} \n"
    result += "\\end{adjustbox} \n"
    result += f"\\caption{{{caption}}} \n"
    result += "\\label{tab:table_label} \n"
    result += "\\end{table} \n"

    result = result.replace("nan", "-")
    print(result)


def max_formatter(x):
    #     if x is None:
    #         return "blubb"
    if float(x) >= 0.8:
        return "$\\boldsymbol{" + str(x) + "}$"
    else:
        return str(x)


comparison_data_par10 = []
comparison_data_succ = []
comparison_data_rmses = []
comparison_data_ndcgs = []
comparison_data_taus = []

weighted_vs_unweighted_par10 = []
weighted_vs_unweighted_succ = []

for scenario_name in scenario_names:

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)

    # compute vbs values
    vbs_par10 = []
    vbs_succ = 0
    for index, row in scenario.performance_data.iterrows():
        vbs_par10.append(row.min())
    for index, row in scenario.runstatus_data.iterrows():
        if "ok" in row.to_numpy():
            vbs_succ += 1

    val_vbs_par10 = np.mean(vbs_par10)
    val_vbs_succ = vbs_succ / len(scenario.performance_data)

    # print("vbs par10", val_vbs_par10)
    # print("vbs succ", val_vbs_succ)

    # baselines
    df_baseline_rf = None
    df_baseline_lr = None
    df_baseline_label_ranking = None
    df_baseline_sbs = None
    df_baseline_sf = None

    # hinge neural network
    df_corras_nnh = None
    df_corras_nnh_weighted = None
    df_corras_nnh_unweighted = None

    # linear and quadratic hinge
    df_corras_hinge = None
    df_corras_hinge_quadratic_unweighted = None
    df_corras_hinge_quadratic_weighted = None
    df_corras_hinge_linear_unweighted = None
    df_corras_hinge_linear_weighted = None

    # linear and quadratic pl
    df_corras_pl = None
    df_corras_pl_quadratic_unweighted = None
    df_corras_pl_quadratic_weighted = None
    df_corras_pl_linear_unweighted = None
    df_corras_pl_linear_weighted = None

    # pl neural network
    df_corras_plnet = None
    df_corras_plnet_weighted = None
    df_corras_plnet_unweighted = None

    try:
        df_baseline_sbs = pd.read_csv(evaluations_path + "sbs-" +
                                      scenario_name + ".csv")
    except Exception as ex:
        print(ex)

    try:
        df_baseline_lr = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-linear_regression" +
                                     scenario_name + ".csv")
    except Exception as ex:
        print(ex)
    try:
        df_baseline_sf = pd.read_csv(
            evaluations_path + "baseline-evaluation-survival-forest-fixed-" +
            scenario_name + ".csv")
    except Exception as ex:
        print(ex)

    try:
        df_baseline_rf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-random_forest" +
                                     scenario_name + ".csv")
    except Exception as ex:
        print(ex)

    try:
        df_baseline_label_ranking = pd.read_csv(evaluations_path +
                                                "baseline-label-ranking-" +
                                                scenario_name + ".csv")
    except Exception as ex:
        print(ex)
    try:
        df_corras_plnet = pd.read_csv(evaluations_path + "ki2020-plnet-" +
                                      scenario_name + ".csv")
        df_corras_plnet_weighted = df_corras_plnet.loc[
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["use_weighted_samples"] == True)]

        df_corras_plnet_unweighted = df_corras_plnet.loc[
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["use_weighted_samples"] == False)]
    except Exception as ex:
        print(ex)

        # print("plnet head", df_corras_plnet.head())

        # load hinge neural network data
    try:
        df_corras_nnh = pd.read_csv(evaluations_path + "ki2020-nnh-" +
                                    scenario_name + ".csv")
        # print("plnet head", df_corras_plnet.head())

        df_corras_nnh_weighted = df_corras_plnet.loc[
            (df_corras_nnh["activation_function"] == "sigmoid") &
            (df_corras_nnh["layer_sizes"] == "[32]") &
            (df_corras_nnh["epsilon"] == 1.0) &
            (df_corras_nnh["use_weighted_samples"] == True)]

        df_corras_nnh_unweighted = df_corras_nnh.loc[
            (df_corras_nnh["activation_function"] == "sigmoid") &
            (df_corras_nnh["layer_sizes"] == "[32]") &
            (df_corras_nnh["epsilon"] == 1.0) &
            (df_corras_nnh["use_weighted_samples"] == False)]
    except Exception as ex:
        print(ex)

    try:
        df_corras_pl = pd.read_csv(evaluations_path + "ki2020_linpl-" +
                                   scenario_name + ".csv")

        df_corras_pl_linear_weighted = df_corras_pl.loc[
            (df_corras_pl["quadratic_transform"] == False) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["use_weighted_samples"] == True)]

        df_corras_pl_linear_unweighted = df_corras_pl.loc[
            (df_corras_pl["quadratic_transform"] == False) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["use_weighted_samples"] == False)]

        df_corras_pl_quadratic_weighted = df_corras_pl.loc[
            (df_corras_pl["quadratic_transform"] == True) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["use_weighted_samples"] == True)]

        df_corras_pl_quadratic_unweighted = df_corras_pl.loc[
            (df_corras_pl["quadratic_transform"] == True) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["use_weighted_samples"] == False)]
    except Exception as ex:
        print(ex)

    try:
        df_corras_hinge = pd.read_csv(evaluations_path +
                                      "ki2020-linhinge-" + scenario_name +
                                      ".csv")


        df_corras_hinge_linear_weighted = df_corras_hinge.loc[
            (df_corras_hinge["quadratic_transform"] == False) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == True)]

        df_corras_hinge_linear_unweighted = df_corras_hinge.loc[
            (df_corras_hinge["quadratic_transform"] == False) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == False)]

        df_corras_hinge_quadratic_weighted = df_corras_hinge.loc[
            (df_corras_hinge["quadratic_transform"] == True) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == True)]

        df_corras_hinge_quadratic_unweighted = df_corras_hinge.loc[
            (df_corras_hinge["quadratic_transform"] == True) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == False)]
    except Exception as ex:
        print(ex)

    # correct hinge lambda values

    approaches_dfs = [
        # df_baseline_sbs,
        # df_baseline_rf,
        # df_baseline_lr,
        # df_baseline_label_ranking,
        # df_baseline_sf,
        df_corras_pl_linear_unweighted,
        # df_corras_pl_linear_weighted,
        df_corras_pl_quadratic_unweighted,
        # df_corras_pl_quadratic_weighted,
        df_corras_plnet_unweighted,
        # df_corras_plnet_weighted,
        df_corras_hinge_linear_unweighted,
        # df_corras_hinge_linear_weighted,
        df_corras_hinge_quadratic_unweighted,
        # df_corras_hinge_quadratic_weighted,
        df_corras_nnh_unweighted,
        # df_corras_nnh_weighted
    ]

    approaches_names = [
        # "VBS",
        # "SBS",
        # "RF",
        # "LR",
        # "LR",
        # "RSF",
        "PL-LM",
        # "W PL-GLM",
        "PL-QM",
        # "W PL-QM",
        "PL-NN",
        # "W PL-NN",
        "Hinge-LM",
        # "W Hinge-LM",
        "Hinge-QM",
        # "W Hinge-QM",
        "Hinge-NN",
        # "W Hinge-NN"
    ]
    # print(scenario.scenario, len(scenario.performance_data))
    # print(len(df_corras_hinge))
    # for i, x in enumerate(approaches_dfs):
    #     if x is None:
    #         print(approaches_names[i], 0)
    #     else:
    #         print(approaches_names[i], len(x))

    # if df_corras_plnet_unweighted is not None:
    #     print("CORRAS PL NET", scenario_name,
    #           df_corras_plnet_unweighted["tau_corr"].mean(),
    #           len(scenario.performance_data), len(df_corras_plnet_unweighted))

    for lambda_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        par10_scores = [scenario_name]
        succ_rates = [scenario_name]
        for approach_name, approach_df in zip(approaches_names,
                                              approaches_dfs):
            # print(approach_df.columns)
            # print(approach_df.head())
            current_df = approach_df[approach_df["lambda"] == lambda_val]
            try:
                if len(current_df) == 6*len(scenario.performance_data):
                    par10_scores.append(current_df["par10"].mean())
                    succ_rates.append(current_df["run_status"].value_counts(
                        normalize=True)["ok"])
                else:

                    print(
                        f"Approach {approach_name} has {len(current_df)} entries but {scenario_name} has {len(scenario.performance_data)}!"
                    )
                    par10_scores.append(float("nan"))
                    succ_rates.append(float("nan"))
            except Exception as ex:
                print("exception for ", approach_name, ex)
                print(
                    f"File for {approach_name} for scenario {scenario_name} not found!"
                )
                par10_scores.append(float("nan"))
                succ_rates.append(float("nan"))

        comparison_data_par10.append(par10_scores + [lambda_val])
        comparison_data_succ.append(succ_rates + [lambda_val])

    for approach_df in approaches_dfs:
        if approach_df is not None:
            approach_df.loc[:, "rmse"] = approach_df["mse"].pow(1. / 2)

    for lambda_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        taus = [scenario_name]
        rmses = [scenario_name]
        ndcgs = [scenario_name]
        lambda_vals = [scenario_name]
        for approach_name, approach_df in zip(approaches_names,
                                              approaches_dfs):
            current_df = approach_df[approach_df["lambda"] == lambda_val]
            try:
                if len(current_df) == 6 * len(scenario.performance_data):
                    taus.append(current_df["tau_corr"].mean())
                    rmses.append(math.sqrt(current_df["mse"].mean()))
                    ndcgs.append(current_df["ndcg"].mean())
                    lambda_vals.append(lambda_val)
                else:
                    print(len(current_df), len(scenario.performance_data))
                    taus.append(float("nan"))
                    rmses.append(float("nan"))
                    ndcgs.append(float("nan"))
            except Exception as ex:
                print(ex)
                taus.append(float("nan"))
                rmses.append(float("nan"))
                ndcgs.append(float("nan"))
        # print("taus", taus)

        comparison_data_taus.append(taus + [lambda_val])
        comparison_data_rmses.append(rmses + [lambda_val])
        comparison_data_ndcgs.append(ndcgs + [lambda_val])

comparison_frame_taus = pd.DataFrame(data=comparison_data_taus,
                                     columns=["Scenario"] + approaches_names +
                                     ["lambda"])
comparison_frame_ndcgs = pd.DataFrame(data=comparison_data_ndcgs,
                                      columns=["Scenario"] + approaches_names +
                                      ["lambda"])
comparison_frame_rmses = pd.DataFrame(data=comparison_data_rmses,
                                      columns=["Scenario"] + approaches_names +
                                      ["lambda"])

comparison_frame_par10 = pd.DataFrame(data=comparison_data_par10,
                                      columns=["Scenario"] + approaches_names +
                                      ["lambda"])
comparison_frame_succ = pd.DataFrame(data=comparison_data_succ,
                                     columns=["Scenario"] + approaches_names +
                                     ["lambda"])

print(comparison_frame_taus)


counters_data = []

for approach_name in approaches_names:
    counters = [0,0,0]
    for scenario in scenario_names:
        cur_frame = comparison_frame_taus[comparison_frame_taus["Scenario"] ==
                                        scenario]
        
        best_lambda = cur_frame["lambda"][cur_frame[approach_name].idxmax()]
        if best_lambda == 0.0:
            counters[0] += 1
        elif best_lambda == 1.0:
            counters[2] += 1
        else:
            counters[1] += 1
    counters_data.append(counters)

results_frame_taus = pd.DataFrame(data=counters_data,index=approaches_names,columns=["lambda=0", "lambda in (0,1)", "lambda=1"])

results_frame_taus.to_csv("ki2020_taus.csv")


counters_data = []

for approach_name in approaches_names:
    counters = [0,0,0]
    for scenario in scenario_names:
        cur_frame = comparison_frame_par10[comparison_frame_taus["Scenario"] ==
                                        scenario]
        
        best_lambda = cur_frame["lambda"][cur_frame[approach_name].idxmin()]
        if best_lambda == 0.0:
            counters[0] += 1
        elif best_lambda == 1.0:
            counters[2] += 1
        else:
            counters[1] += 1
    counters_data.append(counters)

results_frame_taus = pd.DataFrame(data=counters_data,index=approaches_names,columns=["lambda=0", "lambda in (0,1)", "lambda=1"])

results_frame_taus.to_csv("ki2020_par10.csv")

# ranks = comparison_frame_par10.iloc[:, 3:].rank(axis=1, method="average", ascending=True)
# print(ranks.head())

# names = comparison_frame_par10.columns[3:].to_list()
# print(names)
# avranks = []
# for name in names:
#     avranks.append(ranks[name].mean())
# # avranks =  [1.9, 3.2, 2.8, 3.3 ]
# print(names)
# print(avranks)
# cd = Orange.evaluation.compute_CD(
#     avranks, len(comparison_frame_par10))  #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# plt.savefig(fname=figures_path + "par10.pdf", bbox_inches="tight")

# ranks = comparison_frame_succ.iloc[:, 3:].rank(axis=1, method="average", ascending=False)
# print(ranks.head())

# names = comparison_frame_succ.columns[3:].to_list()
# print(names)
# avranks = []
# for name in names:
#     avranks.append(ranks[name].mean())
# # avranks =  [1.9, 3.2, 2.8, 3.3 ]
# print(names)
# print(avranks)
# cd = Orange.evaluation.compute_CD(
#     avranks, len(comparison_frame_succ))  #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# plt.savefig(fname=figures_path + "succ.pdf", bbox_inches="tight")

# ranks = comparison_frame_taus.iloc[:, 1:].rank(axis=1, method="average", ascending=False)
# print(ranks.head())

# names = comparison_frame_taus.columns[1:].to_list()
# print(names)
# avranks = []
# for name in names:
#     avranks.append(ranks[name].mean())
# # avranks =  [1.9, 3.2, 2.8, 3.3 ]
# print(names)
# print(avranks)
# cd = Orange.evaluation.compute_CD(
#     avranks, len(comparison_frame_taus))  #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# plt.savefig(fname=figures_path + "taus.pdf", bbox_inches="tight")

# # ranks = comparison_frame_rmses.iloc[:, 1:].rank(axis=1, method="average", ascending=True)
# # print(ranks.head())

# # names = comparison_frame_rmses.columns[1:].to_list()
# # print(names)
# # avranks = []
# # for name in names:
# #     avranks.append(ranks[name].mean())
# # # avranks =  [1.9, 3.2, 2.8, 3.3 ]
# # print(names)
# # print(avranks)
# # cd = Orange.evaluation.compute_CD(
# #     avranks, len(comparison_frame_rmses))  #tested on 30 datasets
# # Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# # plt.savefig(fname=figures_path + "rmses.pdf", bbox_inches="tight")

# ranks = comparison_frame_ndcgs.iloc[:, 1:].rank(axis=1, method="average", ascending=False)
# print(ranks.head())

# names = comparison_frame_ndcgs.columns[1:].to_list()
# print(names)
# avranks = []
# for name in names:
#     avranks.append(ranks[name].mean())
# # avranks =  [1.9, 3.2, 2.8, 3.3 ]
# print(names)
# print(avranks)
# cd = Orange.evaluation.compute_CD(
#     avranks, len(comparison_frame_ndcgs))  #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# plt.savefig(fname=figures_path + "ndcgs.pdf", bbox_inches="tight")

# os.system("pdfcrop " + figures_path + "ndcgs.pdf " +
#               figures_path + "ndcgs.pdf")
# os.system("pdfcrop " + figures_path + "taus.pdf " +
#               figures_path + "taus.pdf")
# # os.system("pdfcrop " + figures_path + "rmses.pdf " +
# #               figures_path + "rmses.pdf")
# os.system("pdfcrop " + figures_path + "succ.pdf " +
#               figures_path + "succ.pdf")
# os.system("pdfcrop " + figures_path + "par10.pdf " +
#               figures_path + "par10.pdf")