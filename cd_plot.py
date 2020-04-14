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

figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/plots/cd/"

scenario_names = [
    "MIP-2016", "CSP-2010", "CPMP-2015", "SAT11-INDU", "SAT11-HAND",
    "SAT11-RAND", "QBF-2016", "MAXSAT-WPMS-2016", "MAXSAT-PMS-2016",
    "CSP-Minizinc-Time-2016"
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

lambda_value_pl = 0.5
lambda_value_hinge = 0.5
epsilon_value_hinge = 1.0


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
        df_corras_plnet = pd.read_csv(evaluations_path + "corras-pl-nn-" +
                                      scenario_name + "-scen.csv")
        df_corras_plnet_weighted = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed)
            & (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 0.5) &
            (df_corras_plnet["use_weighted_samples"] == True)]

        df_corras_plnet_unweighted = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed)
            & (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 0.5) &
            (df_corras_plnet["use_weighted_samples"] == False)]
    except Exception as ex:
        print(ex)

        # print("plnet head", df_corras_plnet.head())

        # load hinge neural network data
    try:
        df_corras_nnh = pd.read_csv(evaluations_path + "corras-hinge-nn-" +
                                    scenario_name + "-scen.csv")
        # print("plnet head", df_corras_plnet.head())

        df_corras_nnh_weighted = df_corras_plnet.loc[
            (df_corras_nnh["seed"] == seed)
            & (df_corras_nnh["activation_function"] == "sigmoid") &
            (df_corras_nnh["layer_sizes"] == "[32]") &
            (df_corras_nnh["lambda"] == 0.5) &
            (df_corras_nnh["epsilon"] == 1.0) &
            (df_corras_nnh["use_weighted_samples"] == True)]

        df_corras_nnh_unweighted = df_corras_nnh.loc[
            (df_corras_nnh["seed"] == seed)
            & (df_corras_nnh["activation_function"] == "sigmoid") &
            (df_corras_nnh["layer_sizes"] == "[32]") &
            (df_corras_nnh["lambda"] == 0.5) &
            (df_corras_nnh["epsilon"] == 1.0) &
            (df_corras_nnh["use_weighted_samples"] == False)]
    except Exception as ex:
        print(ex)

    try:
        df_corras_pl = pd.read_csv(evaluations_path + "corras-pl-log-linear-" +
                                   scenario_name + "-new-scen.csv")

        df_corras_pl_linear_weighted = df_corras_pl.loc[
            (df_corras_pl["seed"] == seed)
            & (df_corras_pl["quadratic_transform"] == False) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["lambda"] == 0.5) &
            (df_corras_pl["use_weighted_samples"] == True) &
            (df_corras_pl["lambda"] == lambda_value_pl)]

        df_corras_pl_linear_unweighted = df_corras_pl.loc[
            (df_corras_pl["seed"] == seed)
            & (df_corras_pl["quadratic_transform"] == False) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["lambda"] == 0.5) &
            (df_corras_pl["use_weighted_samples"] == False) &
            (df_corras_pl["lambda"] == lambda_value_pl)]

        df_corras_pl_quadratic_weighted = df_corras_pl.loc[
            (df_corras_pl["seed"] == seed)
            & (df_corras_pl["quadratic_transform"] == True) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["lambda"] == 0.5) &
            (df_corras_pl["use_weighted_samples"] == True) &
            (df_corras_pl["lambda"] == lambda_value_pl)]

        df_corras_pl_quadratic_unweighted = df_corras_pl.loc[
            (df_corras_pl["seed"] == seed)
            & (df_corras_pl["quadratic_transform"] == True) &
            (df_corras_pl["scale_to_unit_interval"] == True) &
            (df_corras_pl["max_inverse_transform"] == "max_cutoff") &
            (df_corras_pl["lambda"] == 0.5) &
            (df_corras_pl["use_weighted_samples"] == False) &
            (df_corras_pl["lambda"] == lambda_value_pl)]
    except Exception as ex:
        print(ex)

    try:
        df_corras_hinge = pd.read_csv(evaluations_path +
                                      "corras-hinge-linear-" + scenario_name +
                                      "-new-weights-scen.csv")

        df_corras_hinge_linear_weighted = df_corras_hinge.loc[
            (df_corras_hinge["seed"] == seed)
            & (df_corras_hinge["quadratic_transform"] == False) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "None") &
            (df_corras_hinge["lambda"] == 0.5) &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == True) &
            (df_corras_hinge["lambda"] == lambda_value_pl)]

        df_corras_hinge_linear_unweighted = df_corras_hinge.loc[
            (df_corras_hinge["seed"] == seed)
            & (df_corras_hinge["quadratic_transform"] == False) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "None") &
            (df_corras_hinge["lambda"] == 0.5) &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == False) &
            (df_corras_hinge["lambda"] == lambda_value_pl)]

        df_corras_hinge_quadratic_weighted = df_corras_hinge.loc[
            (df_corras_hinge["seed"] == seed)
            & (df_corras_hinge["quadratic_transform"] == True) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "None") &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == True) &
            (df_corras_hinge["lambda"] == lambda_value_pl)]

        df_corras_hinge_quadratic_unweighted = df_corras_hinge.loc[
            (df_corras_hinge["seed"] == seed)
            & (df_corras_hinge["quadratic_transform"] == True) &
            (df_corras_hinge["scale_to_unit_interval"] == True) &
            (df_corras_hinge["max_inverse_transform"] == "None") &
            (df_corras_hinge["epsilon"] == 1.0) &
            (df_corras_hinge["use_weighted_samples"] == False) &
            (df_corras_hinge["lambda"] == lambda_value_pl)]
    except Exception as ex:
        print(ex)

    approaches_dfs = [
        # df_baseline_sbs,
        df_baseline_rf,
        df_baseline_lr,
        df_baseline_label_ranking,
        df_baseline_sf,
        df_corras_nnh_unweighted,
        df_corras_nnh_weighted,
        df_corras_hinge_linear_unweighted,
        df_corras_hinge_linear_weighted,
        df_corras_hinge_quadratic_unweighted,
        df_corras_hinge_quadratic_weighted,
        df_corras_pl_linear_unweighted,
        df_corras_pl_linear_weighted,
        df_corras_pl_quadratic_unweighted,
        df_corras_pl_quadratic_weighted,
        df_corras_plnet_unweighted,
        df_corras_plnet_weighted
    ]

    approaches_names = [
        "VBS",
        "SBS",
        "RF",
        "LR",
        "Label Ranking",
        "RSF",
        "Hinge-NN",
        "W Hinge-NN",
        "Hinge-LM",
        "W Hinge-LM",
        "Hinge-QM",
        "W Hinge-QM",
        "PL-GLM",
        "W PL-GLM",
        "PL-QM",
        "W PL-QM",
        "PL-NN",
        "W PL-NN"
    ]
    # print(scenario.scenario, len(scenario.performance_data))
    # print(len(df_corras_hinge))
    # for i, x in enumerate(approaches_dfs):
    #     if x is None:
    #         print(approaches_names[i], 0)
    #     else:
    #         print(approaches_names[i], len(x))
    par10_scores = [
        scenario_name, val_vbs_par10,
        df_baseline_sbs["par10_sbs_par10"].mean()
    ]
    succ_rates = [
        scenario_name, val_vbs_succ,
        df_baseline_sbs["success_rate_sbs_succ"].iloc[0]
    ]

    if df_corras_plnet_unweighted is not None:
        print("CORRAS PL NET", scenario_name,
              df_corras_plnet_unweighted["tau_corr"].mean(),
              len(scenario.performance_data), len(df_corras_plnet_unweighted))

    for approach_name, approach_df in zip(approaches_names, approaches_dfs):
        try:
            if len(approach_df) == len(scenario.performance_data):
                par10_scores.append(approach_df["par10"].mean())
                succ_rates.append(approach_df["run_status"].value_counts(
                    normalize=True)["ok"])
            else:

                print(
                    f"Approach {approach_name} has {len(approach_df)} entries but {scenario_name} has {len(scenario.performance_data)}!"
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

    comparison_data_par10.append(par10_scores)
    comparison_data_succ.append(succ_rates)

    for approach_df in approaches_dfs:
        if approach_df is not None:
            approach_df["rmse"] = approach_df["mse"].pow(1. / 2)

    taus = [scenario_name]
    rmses = [scenario_name]
    ndcgs = [scenario_name]

    for approach_name, approach_df in zip(approaches_names, approaches_dfs):
        try:
            if len(approach_df) == len(scenario.performance_data):
                taus.append(approach_df["tau_corr"].mean())
                rmses.append(approach_df["rmse"].mean())
                ndcgs.append(approach_df["ndcg"].mean())
            else:
                print(len(approach_df), len(scenario.performance_data))
                taus.append(float("nan"))
                rmses.append(float("nan"))
                ndcgs.append(float("nan"))
        except Exception as ex:
            print(ex)
            taus.append(float("nan"))
            rmses.append(float("nan"))
            ndcgs.append(float("nan"))
    print("taus", taus)

    comparison_data_taus.append(taus)
    comparison_data_rmses.append(rmses)
    comparison_data_ndcgs.append(ndcgs)

comparison_frame_taus = pd.DataFrame(data=comparison_data_taus,
                                     columns=["Scenario"] +
                                     approaches_names[2:])
comparison_frame_ndcgs = pd.DataFrame(data=comparison_data_ndcgs,
                                      columns=["Scenario"] +
                                      approaches_names[2:])
comparison_frame_rmses = pd.DataFrame(data=comparison_data_rmses,
                                      columns=["Scenario"] +
                                      approaches_names[2:])

comparison_frame_par10 = pd.DataFrame(data=comparison_data_par10,
                                      columns=["Scenario"] + approaches_names)
comparison_frame_succ = pd.DataFrame(data=comparison_data_succ,
                                     columns=["Scenario"] + approaches_names)

ranks = comparison_frame_par10.iloc[:, 3:].rank(axis=1, method="average", ascending=True)
print(ranks.head())

names = comparison_frame_par10.columns[3:].to_list()
print(names)
avranks = []
for name in names:
    avranks.append(ranks[name].mean())
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
print(names)
print(avranks)
cd = Orange.evaluation.compute_CD(
    avranks, len(comparison_frame_par10))  #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(fname=figures_path + "par10.pdf", bbox_inches="tight")

ranks = comparison_frame_succ.iloc[:, 3:].rank(axis=1, method="average", ascending=False)
print(ranks.head())

names = comparison_frame_succ.columns[3:].to_list()
print(names)
avranks = []
for name in names:
    avranks.append(ranks[name].mean())
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
print(names)
print(avranks)
cd = Orange.evaluation.compute_CD(
    avranks, len(comparison_frame_succ))  #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(fname=figures_path + "succ.pdf", bbox_inches="tight")

ranks = comparison_frame_taus.iloc[:, 1:].rank(axis=1, method="average", ascending=False)
print(ranks.head())

names = comparison_frame_taus.columns[1:].to_list()
print(names)
avranks = []
for name in names:
    avranks.append(ranks[name].mean())
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
print(names)
print(avranks)
cd = Orange.evaluation.compute_CD(
    avranks, len(comparison_frame_taus))  #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(fname=figures_path + "taus.pdf", bbox_inches="tight")


ranks = comparison_frame_rmses.iloc[:, 1:].rank(axis=1, method="average", ascending=True)
print(ranks.head())

names = comparison_frame_rmses.columns[1:].to_list()
print(names)
avranks = []
for name in names:
    avranks.append(ranks[name].mean())
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
print(names)
print(avranks)
cd = Orange.evaluation.compute_CD(
    avranks, len(comparison_frame_rmses))  #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(fname=figures_path + "rmses.pdf", bbox_inches="tight")


ranks = comparison_frame_ndcgs.iloc[:, 1:].rank(axis=1, method="average", ascending=False)
print(ranks.head())

names = comparison_frame_ndcgs.columns[1:].to_list()
print(names)
avranks = []
for name in names:
    avranks.append(ranks[name].mean())
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
print(names)
print(avranks)
cd = Orange.evaluation.compute_CD(
    avranks, len(comparison_frame_ndcgs))  #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(fname=figures_path + "ndcgs.pdf", bbox_inches="tight")

os.system("pdfcrop " + figures_path + "ndcgs.pdf " +
              figures_path + "ndcgs.pdf")
os.system("pdfcrop " + figures_path + "taus.pdf " +
              figures_path + "taus.pdf")
os.system("pdfcrop " + figures_path + "rmses.pdf " +
              figures_path + "rmses.pdf")
os.system("pdfcrop " + figures_path + "succ.pdf " +
              figures_path + "succ.pdf")
os.system("pdfcrop " + figures_path + "par10.pdf " +
              figures_path + "par10.pdf")