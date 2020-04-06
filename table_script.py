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
import matplotlib.pyplot as plt
import seaborn as sns

scenario_path = "./aslib_data-aslib-v4.0/"
evaluations_path = "./evaluation-results/evaluations/"
evaluations_path_nnh = "./evaluation-results/evaluations-nnh-config/"

figures_path = "./figures/progression-plots/"

scenario_names = [
    "SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND",
    "CPMP-2015", "QBF-2016", "SAT12-ALL", "MAXSAT-WPMS-2016",
    "MAXSAT-PMS-2016", "CSP-Minizinc-Time-2016"
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


def create_latex_max(df: pd.DataFrame, decimal_format="{:10.3f}", skip_max = 2):
    result = "\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "} \n"
    result += "\\toprule \n"
    result += " & ".join(df.columns) + " \\\\ \n"
    result += "\\midrule \n"
    for index, row in df.iterrows():
        result += row[0] + " & " + " & ".join([
            "\\textbf{" + decimal_format.format(x) +
            "}" if x == np.nanmax(row[skip_max:].to_numpy().astype(
                "float64")) else decimal_format.format(x)
            for x in row[1:]
        ]) + " \\\\ \n"
    result += "\\toprule \n"
    result += "\\end{tabular} \n"

    result = result.replace("nan", "-")
    print(result)


def create_latex_min(df: pd.DataFrame, decimal_format="{:10.3f}"):
    result = "\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "} \n"
    result += "\\toprule \n"
    result += " & ".join(df.columns) + " \\\\ \n"
    result += "\\midrule \n"
    for index, row in df.iterrows():
        result += row[0] + " & " + " & ".join([
            "\\textbf{" + decimal_format.format(x) +
            "}" if x == np.nanmin(row[2:]) else decimal_format.format(x)
            for x in row[1:]
        ]) + " \\\\ \n"
    result += "\\toprule \n"
    result += "\\end{tabular} \n"
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

    print("vbs par10", val_vbs_par10)
    print("vbs succ", val_vbs_succ)

    # baselines
    df_baseline_rf = None
    df_baseline_lr = None
    df_baseline_sbs = None
    df_baseline_sf = None
    df_corras_nnh = None
    df_corras_nnh_weighted = None
    df_corras_nnh_unweighted = None
    df_corras_hinge = None
    df_corras_hinge_quadratic_unweighted = None
    df_corras_hinge_quadratic_weighted = None
    df_corras_hinge_linear_unweighted = None
    df_corras_hinge_linear_weighted = None
    df_corras_pl = None
    df_corras_pl_quadratic_unweighted = None
    df_corras_pl_quadratic_weighted = None
    df_corras_pl_linear_unweighted = None
    df_corras_pl_linear_weighted = None
    df_corras_plnet = None
    df_corras_plnet_weighted = None
    df_corras_plnet_unweighted = None
    try:
        df_baseline_sbs = pd.read_csv(evaluations_path + "sbs-" +
                                      scenario_name + ".csv")
        df_baseline_lr = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-linear_regression" +
                                     scenario_name + ".csv")
        df_baseline_sf = pd.read_csv(
            evaluations_path + "baseline-evaluation-survival-forest-fixed-" +
            scenario_name + ".csv")
        df_baseline_rf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-random_forest" +
                                     scenario_name + ".csv")
        df_corras_plnet = pd.read_csv(
            evaluations_path + "corras-pl-nn-" + scenario_name + ".csv")
        print("plnet head", df_corras_plnet.head())
        df_corras_plnet_1616 = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 1.0)]
        print("plnet 1616 head", df_corras_plnet_1616.head())
        df_corras_nnh_all = pd.read_csv(evaluations_path_nnh +
                                        "corras-hinge-nn-" + scenario_name +
                                        "-config-test.csv")
        df_corras_nnh = df_corras_nnh_all.loc[
            (df_corras_nnh_all["seed"] == seed)
            & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_nnh_all["learning_rate"] == 0.001) &
            (df_corras_nnh_all["lambda"] == lambda_value_hinge)]
        df_corras_all = pd.read_csv(evaluations_path +
                                    "corras-pl-log-linear-" + scenario_name +
                                    "-new.csv")
        df_corras_linear = df_corras_all.loc[
            (df_corras_all["seed"] == seed)
            & (df_corras_all["quadratic_transform"] == False) &
            (df_corras_all["scale_to_unit_interval"] == True) &
            (df_corras_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_all["lambda"] == lambda_value_pl)]
        df_corras_quadratic = df_corras_all.loc[
            (df_corras_all["seed"] == seed)
            & (df_corras_all["quadratic_transform"] == True) &
            (df_corras_all["scale_to_unit_interval"] == True) &
            (df_corras_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_all["lambda"] == lambda_value_pl)]
        df_corras_hinge_linear_all = pd.read_csv(evaluations_path +
                                                 "corras-hinge-linear-" +
                                                 scenario_name + "-new-unweighted" +
                                                 ".csv")
        df_corras_hinge_linear = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_hinge_linear_all["quadratic_transform"] == False) &
            (df_corras_hinge_linear_all["scale_to_unit_interval"] == True) &
            (df_corras_hinge_linear_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge_linear_all["use_weighted_samples"] == False)]
        df_corras_hinge_linear_weighted = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_hinge_linear_all["quadratic_transform"] == False) &
            (df_corras_hinge_linear_all["scale_to_unit_interval"] == True) &
            (df_corras_hinge_linear_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge_linear_all["use_weighted_samples"] == True)]
        print("not weighted", len(df_corras_hinge_linear))
        print("weighted", len(df_corras_hinge_linear_weighted))
        print("plnet", len(df_corras_plnet_1616))
        df_corras_hinge_quadratic = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_hinge_linear_all["quadratic_transform"] == True) &
            (df_corras_hinge_linear_all["scale_to_unit_interval"] == True) &
            (df_corras_hinge_linear_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge_linear_all["use_weighted_samples"] == False)]

        # dfs = [df_baseline_lr, df_baseline_rf,
        #        df_corras_linear, df_corras_quadratic]

    except Exception as ex:
        print(ex)

    val_rf = float("nan")
    val_lr = float("nan")
    val_sf = float("nan")
    val_pl_linear = float("nan")
    val_pl_quad = float("nan")
    val_nnh = float("nan")
    val_hinge_linear = float("nan")
    val_hinge_quad = float("nan")
    val_sbs_par10 = float("nan")
    val_sbs_succ = float("nan")
    val_hinge_weighted = float("nan")
    val_pl_net = float("nan")
    print(scenario_name)
    if df_baseline_sbs is not None:
        val_sbs_par10 = df_baseline_sbs["success_rate_sbs_par10"].iloc[0]
        val_sbs_succ = df_baseline_sbs["success_rate_sbs_succ"].iloc[0]
    if df_baseline_rf is not None:
        val_rf = df_baseline_rf["run_status"].value_counts(
            normalize=True)["ok"]
    if df_baseline_lr is not None:
        val_lr = df_baseline_lr["run_status"].value_counts(
            normalize=True)["ok"]
    if df_baseline_sf is not None:
        val_sf = df_baseline_sf["run_status"].value_counts(
            normalize=True)["ok"]
    if df_corras_plnet_1616 is not None:
        val_pl_net = float("nan")
        try:
            val_pl_net = df_corras_plnet_1616["run_status"].value_counts(
                normalize=True)["ok"]
        except:
            val_pl_net = float("nan")
    if df_baseline_sf is not None:
        val_baseline_sf = df_baseline_sf["run_status"].value_counts(
            normalize=True)["ok"]
    if df_corras_linear is not None:
        try:
            val_pl_linear = df_corras_linear["run_status"].value_counts(
                normalize=True)["ok"]
        except:
            print("corras linear run status", df_corras_linear["run_status"].value_counts(
                normalize=True))
            pass
    if df_corras_quadratic is not None:
        try:
            val_pl_quad = df_corras_quadratic["run_status"].value_counts(
                normalize=True)["ok"]
        except:
            pass
    if df_corras_hinge_linear_weighted is not None:
        try:
            val_hinge_weighted = df_corras_hinge_linear_weighted["run_status"].value_counts(
                normalize=True)["ok"]
        except:
            pass
    if df_corras_nnh is not None:
        val_nnh = df_corras_nnh["run_status"].value_counts(
            normalize=True)["ok"]
    if df_corras_hinge_linear is not None:
        val_hinge_linear = df_corras_hinge_linear["run_status"].value_counts(
            normalize=True)["ok"]
    if df_corras_hinge_quadratic is not None:
        try:
            val_hinge_quad = df_corras_hinge_quadratic["run_status"].value_counts(
                normalize=True)["ok"]
        except:
            pass
    comparison_data_succ.append([
        scenario_name, val_vbs_succ, val_sbs_par10, val_sbs_succ, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad, val_pl_net,
        val_hinge_linear, val_hinge_weighted, val_hinge_quad, val_nnh
    ])

    val_rf = float("nan")
    val_lr = float("nan")
    val_sf = float("nan")
    val_pl_linear = float("nan")
    val_pl_quad = float("nan")
    val_nnh = float("nan")
    val_pl_net = float("nan")
    val_hinge_linear = float("nan")
    val_hinge_weighted = float("nan")
    val_hinge_quad = float("nan")
    val_sbs_par10 = float("nan")
    val_sbs_succ = float("nan")
    print(scenario_name)
    if df_baseline_sbs is not None:
        val_sbs_par10 = df_baseline_sbs["par10_sbs_par10"].mean()
        val_sbs_succ = df_baseline_sbs["par10_sbs_succ"].mean()
    if df_baseline_rf is not None:
        val_rf = df_baseline_rf["par10"].mean()
    if df_baseline_lr is not None:
        val_lr = df_baseline_lr["par10"].mean()
    print(scenario_name, "hlw pre", val_hinge_weighted)
    if df_corras_hinge_linear_weighted is not None:
        val_hinge_weighted = df_corras_hinge_linear_weighted["par10"].mean()
    print(scenario_name, "hlw", val_hinge_weighted)
    if df_baseline_sf is not None:
        val_sf = df_baseline_sf["par10"].mean()
    if df_corras_plnet_1616 is not None:
        val_pl_net = df_corras_plnet_1616["par10"].mean()
    if df_corras_linear is not None:
        val_pl_linear = df_corras_linear["par10"].mean()
    if df_corras_quadratic is not None:
        val_pl_quad = df_corras_quadratic["par10"].mean()
    if df_corras_nnh is not None:
        val_nnh = df_corras_nnh["par10"].mean()
    if df_corras_hinge_linear is not None:
        val_hinge_linear = df_corras_hinge_linear["par10"].mean()
    if df_corras_hinge_quadratic is not None:
        val_hinge_quad = df_corras_hinge_quadratic["par10"].mean()
    comparison_data_par10.append([
        scenario_name, val_vbs_par10, val_sbs_par10, val_sbs_succ, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad, val_pl_net,
        val_hinge_linear, val_hinge_weighted, val_hinge_quad, val_nnh
    ])
comparison_frame_succ = pd.DataFrame(data=comparison_data_succ,
                                     columns=[
                                         "Scenario", "VBS", "SBS by PAR10", "SBS by Succ", "RF", "LR", "RSF", "PL-Lin",
                                         "PL-Quad", "PL-Net", "Hinge-Lin", "Hinge-Lin weighted", "Hinge-Quad",
                                         "Hinge-NN"
                                     ])
comparison_frame_par10 = pd.DataFrame(data=comparison_data_par10,
                                      columns=[
                                          "Scenario", "VBS", "SBS by PAR10", "SBS by Succ", "RF", "LR", "RSF", "PL-Lin",
                                          "PL-Quad", "PL-Net", "Hinge-Lin", "Hinge-Lin weighted", "Hinge-Quad",
                                          "Hinge-NN"
                                      ])

print("success rate")
create_latex_max(comparison_frame_succ)
# print(
#     comparison_frame_succ.to_latex(na_rep="-",
#                                    index=False,
#                                    bold_rows=True,
#                                    float_format="%.3f",
#                                    formatters={"tau_corr": max_formatter},
#                                    escape=False))

print("par10")
create_latex_min(comparison_frame_par10)
# print(
#     comparison_frame_par10.to_latex(na_rep="-",
#                                     index=False,
#                                     bold_rows=True,
#                                     float_format="%.3f",
#                                     formatters={"tau_corr": max_formatter},
#                                     escape=False))

comparison_frame_succ_over_sbs = comparison_frame_succ.copy()
comparison_frame_succ_over_sbs.iloc[:,
                                    1:] = comparison_frame_succ_over_sbs.iloc[:, 1:].div(
                                        comparison_frame_succ.iloc[:, 2],
                                        axis=0)
comparison_frame_par10_over_sbs = comparison_frame_par10.copy()
comparison_frame_par10_over_sbs.iloc[:,
                                     1:] = comparison_frame_par10_over_sbs.iloc[:, 1:].div(
                                         comparison_frame_par10.iloc[:, 2],
                                         axis=0)

# comparison_frame_succ_over_sbs.iloc[:, 1:] = comparison_frame_succ_over_sbs.iloc[:, 1:].div(
#     comparison_frame_succ.iloc[:, 2], axis=0)
# comparison_frame_par10_over_sbs = comparison_frame_par10.copy()
# comparison_frame_par10_over_sbs.iloc[:, 1:] = comparison_frame_par10_over_sbs.iloc[:, 1:].div(
#     comparison_frame_par10.iloc[:, 2], axis=0)

print("success rate over sbs")
create_latex_max(comparison_frame_succ_over_sbs)
# print(
#     comparison_frame_succ_over_sbs.to_latex(na_rep="-",
#                                             index=False,
#                                             bold_rows=True,
#                                             float_format="%.3f",
#                                             formatters={
#                                                 "tau_corr": max_formatter},
#                                             escape=False))

print("par10 over sbs")
create_latex_min(comparison_frame_par10_over_sbs)
# print(
#     comparison_frame_par10_over_sbs.to_latex(na_rep="-",
#                                              index=False,
#                                              bold_rows=True,
#                                              float_format="%.3f",
#                                              formatters={
#                                                  "tau_corr": max_formatter},
#                                              escape=False))

comparison_frame_sbs_over_succ = comparison_frame_succ_over_sbs.copy()
comparison_frame_sbs_over_par10 = comparison_frame_par10_over_sbs.copy()

comparison_frame_sbs_over_succ.iloc[:, 1:] = 1 / \
    comparison_frame_succ_over_sbs.iloc[:, 1:]
comparison_frame_sbs_over_par10.iloc[:, 1:] = 1 / \
    comparison_frame_par10_over_sbs.iloc[:, 1:]

print("sbs over success rate")
create_latex_min(comparison_frame_sbs_over_succ)
# print(
#     comparison_frame_sbs_over_succ.to_latex(na_rep="-",
#                                             index=False,
#                                             bold_rows=True,
#                                             float_format="%.3f",
#                                             formatters={
#                                                 "tau_corr": max_formatter},
#                                             escape=False))

print("sbs over par10")
create_latex_max(comparison_frame_sbs_over_par10)
# print(
#     comparison_frame_sbs_over_par10.to_latex(na_rep="-",
#                                              index=False,
#                                              bold_rows=True,
#                                              float_format="%.3f",
#                                              formatters={
#                                                  "tau_corr": max_formatter},
#                                              escape=False))

gap_numerator_succ = comparison_frame_succ.iloc[:, 1:].subtract(
    comparison_frame_succ.iloc[:, 1], axis=0)
gap_denominator_succ = comparison_frame_succ.iloc[:,
                                                  2] - comparison_frame_succ.iloc[:,
                                                                                  1]

gap_numerator_par10 = comparison_frame_par10.iloc[:, 1:].subtract(
    comparison_frame_par10.iloc[:, 1], axis=0)
gap_denominator_par10 = comparison_frame_par10.iloc[:,
                                                    2] - comparison_frame_par10.iloc[:, 1]

print(gap_numerator_succ)
print(gap_denominator_succ)

sbs_vbs_gap_succ = comparison_frame_succ.copy()

sbs_vbs_gap_par10 = comparison_frame_par10.copy()

sbs_vbs_gap_succ.iloc[:, 1:] = gap_numerator_succ.div(
    gap_denominator_succ, axis=0)
sbs_vbs_gap_par10.iloc[:, 1:] = gap_numerator_par10.div(
    gap_denominator_par10, axis=0)

print("gap succ")
create_latex_min(sbs_vbs_gap_succ)
print("gap par10")
create_latex_min(sbs_vbs_gap_par10)

# gap_numerator_par10 = comparison_frame_par10.iloc[:,1:].subtract(comparison_frame_par10.iloc[:,1])
# gap_denominator_par10 = comparison_frame_par10.iloc[:,2] -  comparison_frame_par10.iloc[:,1]

# comparison_frame_succ_gap = comparison_frame_succ.copy()
# comparison_frame_succ_gap.iloc[:,1:] = gap_numerator_succ[:,:]

# comparison_frame_par10_gap = comparison_frame_par10.copy()
# # comparison_frame_par10_gap.iloc[:,1:] = gap_numerator_par10

# print("success rate gap")
# # create_latex_max(comparison_frame)
# print(
#     comparison_frame_succ_gap.to_latex(na_rep="-",
#                                             index=False,
#                                             bold_rows=True,
#                                             float_format="%.3f",
#                                             formatters={
#                                                 "tau_corr": max_formatter},
#                                             escape=False))

# print("par10 gap")
# # create_latex_max(comparison_frame)
# print(
#     comparison_frame_par10_gap.to_latex(na_rep="-",
#                                              index=False,
#                                              bold_rows=True,
#                                              float_format="%.3f",
#                                              formatters={
#                                                  "tau_corr": max_formatter},
#                                              escape=False))

comparison_data_tau = []
for scenario_name in scenario_names:

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)

    df_baseline_rf = None
    df_baseline_lr = None
    df_baseline_sbs = None
    df_baseline_sf = None
    df_corras_nnh = None
    df_corras_hinge_linear = None
    df_corras_hinge_quadratic = None
    df_corras_linear = None
    df_corras_quadratic = None
    df_corras_hinge_linear_weighted = None
    df_corras_plnet = None
    df_corras_plnet_1616 = None
    df_corras_nnh_all = None
    try:
        df_baseline_sbs = pd.read_csv(evaluations_path + "sbs-" +
                                      scenario_name + ".csv")
        df_baseline_sf = pd.read_csv(
            evaluations_path + "baseline-evaluation-survival-forest-fixed-" +
            scenario_name + ".csv")
        df_baseline_lr = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-linear_regression" +
                                     scenario_name + ".csv")
        df_baseline_rf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-random_forest" +
                                     scenario_name + ".csv")
        df_corras_nnh_all = pd.read_csv(evaluations_path_nnh +
                                "corras-hinge-nn-" + scenario_name +
                                "-config-test.csv")
        df_corras_nnh = df_corras_nnh_all.loc[
            (df_corras_nnh_all["seed"] == seed)
            & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_nnh_all["learning_rate"] == 0.001) &
            (df_corras_nnh_all["lambda"] == lambda_value_hinge)]
        df_corras_plnet = pd.read_csv(
            evaluations_path + "corras-pl-nn-" + scenario_name + ".csv")
        print("plnet head", df_corras_plnet.head())
        df_corras_plnet_1616 = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 1.0)]
        df_corras_all = pd.read_csv(evaluations_path +
                                    "corras-pl-log-linear-" + scenario_name +
                                    "-new.csv")
        print("plnet 1616 head", df_corras_plnet_1616.head())
        df_corras_linear = df_corras_all.loc[
            (df_corras_all["seed"] == seed)
            & (df_corras_all["quadratic_transform"] == False) &
            (df_corras_all["scale_to_unit_interval"] == True) &
            (df_corras_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_all["lambda"] == lambda_value_pl)]
        df_corras_quadratic = df_corras_all.loc[
            (df_corras_all["seed"] == seed)
            & (df_corras_all["quadratic_transform"] == True) &
            (df_corras_all["scale_to_unit_interval"] == True) &
            (df_corras_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_all["lambda"] == lambda_value_pl)]
        df_corras_hinge_linear_all = pd.read_csv(evaluations_path +
                                                 "corras-hinge-linear-" +
                                                 scenario_name + "-new-unweighted" +
                                                 ".csv")
        df_corras_hinge_linear = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_hinge_linear_all["quadratic_transform"] == False) &
            (df_corras_hinge_linear_all["scale_to_unit_interval"] == True) &
            (df_corras_hinge_linear_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge_linear_all["use_weighted_samples"] == False)]
        print("hinge all", df_corras_hinge_linear_all)
        print("hinge linear", df_corras_hinge_linear)
        df_corras_hinge_quadratic = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_hinge_linear_all["quadratic_transform"] == True) &
            (df_corras_hinge_linear_all["scale_to_unit_interval"] == True) &
            (df_corras_hinge_linear_all["max_inverse_transform"] == "max_cutoff") &
            (df_corras_hinge_linear_all["use_weighted_samples"] == False)]

    except Exception as ex:
        print(ex)

    val_rf = float("nan")
    val_lr = float("nan")
    val_sf = float("nan")
    val_pl_linear = float("nan")
    val_pl_quad = float("nan")
    val_nnh = float("nan")
    val_hinge_linear = float("nan")
    val_hinge_quad = float("nan")
    val_sbs_par10 = float("nan")
    val_sbs_succ = float("nan")
    val_pl_net = float("nan")
    if df_baseline_rf is not None:
        val_rf = df_baseline_rf["tau_corr"].mean()
    if df_baseline_lr is not None:
        val_lr = df_baseline_lr["tau_corr"].mean()
    if df_baseline_sf is not None:
        val_sf = df_baseline_sf["tau_corr"].mean()
    if df_corras_plnet_1616 is not None:
        val_pl_net = df_corras_plnet_1616["tau_corr"].mean()
    if df_corras_linear is not None:
        val_pl_linear = df_corras_linear["tau_corr"].mean()
    if df_corras_quadratic is not None:
        val_pl_quad = df_corras_quadratic["tau_corr"].mean()
    if df_corras_nnh is not None:
        val_nnh = df_corras_nnh["tau_corr"].mean()
    if df_corras_hinge_linear is not None:
        val_hinge_linear = df_corras_hinge_linear["tau_corr"].mean()
    if df_corras_hinge_quadratic is not None:
        val_hinge_quad = df_corras_hinge_quadratic["tau_corr"].mean()

    comparison_data_tau.append([
        scenario_name, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad, val_pl_net,
        val_hinge_linear, val_hinge_quad, val_nnh
    ])
comparison_frame_tau = pd.DataFrame(data=comparison_data_tau,
                                    columns=[
                                        "Scenario", "RF", "LR", "RSF",
                                        "PL-Lin", "PL-Quad", "PL-Net", "Hinge-Lin",
                                        "Hinge-Quad", "Hinge-NN"
                                    ])

comparison_frame_tau.iloc[:,
                          1:] = comparison_frame_tau.iloc[:,
                                                          1:].astype("float64")

print(comparison_frame_tau.head())

print("tau_corr")
create_latex_max(comparison_frame_tau, skip_max=1)
print(comparison_frame_tau.to_latex(na_rep="-",
                                    index=False,
                                    bold_rows=True,
                                    float_format="%.3f",
                                    formatters={"tau_corr": max_formatter},
                                    escape=False))


# comparison_data_tau = []
# comparison_data_par10 = []
# for scenario_name in scenario_names:

#     scenario = ASRankingScenario()
#     scenario.read_scenario(scenario_path + scenario_name)

#     df_baseline_rf = None
#     df_baseline_lr = None
#     df_baseline_sf = None
#     df_corras_nnh = None
#     df_corras_hinge_linear = None
#     df_corras_hinge_quadratic = None
#     df_corras_linear = None
#     df_corras_quadratic = None
#     try:
#         df_linhinge_all = pd.read_csv(evaluations_path +
#                                       "sbs-" +
#                                       scenario_name + ".csv")
#         df_baseline_lr = pd.read_csv(evaluations_path +
#                                      "baseline-evaluation-linear_regression" +
#                                      scenario_name + ".csv")
#         df_baseline_rf = pd.read_csv(evaluations_path +
#                                      "baseline-evaluation-random_forest" +
#                                      scenario_name + ".csv")
#         df_baseline_sf = pd.read_csv(evaluations_path +
#                                      "baseline-evaluation-survival-forest-fixed-" +
#                                      scenario_name + ".csv")
#         df_corras_nnh = df_corras_nnh_all.loc[
#             (df_corras_nnh_all["seed"] == seed)
#             & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
#             (df_corras_nnh_all["learning_rate"] == 0.001) &
#             (df_corras_nnh_all["lambda"] == lambda_value_hinge)]
#         df_corras_all = pd.read_csv(evaluations_path +
#                                     "corras-pl-log-linear-" + scenario_name +
#                                     "-new.csv")
#         df_corras_linear = df_corras_all.loc[
#             (df_corras_all["seed"] == seed)
#             & (df_corras_all["quadratic_transform"] == False) &
#             (df_corras_all["scale_to_unit_interval"] == True) &
#             (df_corras_all["max_inverse_transform"] == "max_cutoff") &
#             (df_corras_all["lambda"] == lambda_value_pl)]
#         df_corras_quadratic = df_corras_all.loc[
#             (df_corras_all["seed"] == seed)
#             & (df_corras_all["quadratic_transform"] == True) &
#             (df_corras_all["scale_to_unit_interval"] == True) &
#             (df_corras_all["max_inverse_transform"] == "max_cutoff") &
#             (df_corras_all["lambda"] == lambda_value_pl)]
#         df_corras_nnh_all = pd.read_csv(evaluations_path_nnh + "corras-hinge-nn-" +
#                                         scenario_name + "-config-test.csv")
#         df_corras_hinge_linear_all = pd.read_csv(evaluations_path +
#                                                  "corras-hinge-linear-" +
#                                                  scenario_name + "-new-weighted" +
#                                                  ".csv")
#         df_corras_hinge_linear = df_corras_hinge_linear_all.loc[
#             (df_corras_hinge_linear_all["seed"] == seed)
#             & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
#             (df_corras_all["quadratic_transform"] == False) &
#             (df_corras_all["scale_to_unit_interval"] == True) &
#             (df_corras_all["max_inverse_transform"] == "max_cutoff")]
#         print("hinge all", df_corras_hinge_linear_all)
#         print("hinge linear", df_corras_hinge_linear)
#         df_corras_hinge_quadratic = df_corras_hinge_linear_all.loc[
#             (df_corras_hinge_linear_all["seed"] == seed)
#             & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
#             (df_corras_all["quadratic_transform"] == True) &
#             (df_corras_all["scale_to_unit_interval"] == True) &
#             (df_corras_all["max_inverse_transform"] == "max_cutoff")]

#     except Exception as ex:
#         print(ex)

#     val_rf = float("nan")
#     val_lr = float("nan")
#     val_sf = float("nan")
#     val_pl_linear = float("nan")
#     val_pl_quad = float("nan")
#     val_nnh = float("nan")
#     val_hinge_linear = float("nan")
#     val_hinge_quad = float("nan")
#     val_sbs_par10 = float("nan")
#     val_sbs_succ = float("nan")
#     if df_baseline_linhinge is not None:
#         val_rf = df_baseline_rf["tau_corr"].mean()
#     if df_baseline_lr is not None:
#         val_lr = df_baseline_lr["tau_corr"].mean()


#     comparison_data_tau.append([
#         scenario_name, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad,
#         val_hinge_linear, val_hinge_quad, val_nnh
#     ])

# comparison_frame_tau = pd.DataFrame(data=comparison_data_tau,
#                                     columns=[
#                                         "Scenario", "RF", "LR", "RSF", "PL-Lin",
#                                         "PL-Quad", "Hinge-Lin", "Hinge-Quad",
#                                         "Hinge-NN"
#                                     ])

# print("tau_corr")
# create_latex_max(comparison_frame_tau)
# # print(comparison_frame_tau.to_latex(na_rep="-",
# #                                     index=False,
# #                                     bold_rows=True,
# #                                     float_format="%.3f",
# #                                     formatters={"tau_corr": max_formatter},
# #                                     escape=False))
