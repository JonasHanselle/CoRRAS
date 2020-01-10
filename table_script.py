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
evaluations_path = "./evaluations/"

figures_path = "./figures/progression-plots/"

scenario_names = ["SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND", "CPMP-2015",
    "QBF-2016", "SAT12-ALL", "SAT03-16_INDU", "TTP-2016", "MAXSAT-WPMS-2016", "MAXSAT-PMS-2016", "CSP-Minizinc-Time-2016"]

use_quadratic_transform_values = [True]
use_max_inverse_transform_values = ["max_cutoff", "max_par10"]
# scale_target_to_unit_interval_values = [True, False]
scale_target_to_unit_interval_values = [True]
seed = 15

params = [
    scenario_names, use_quadratic_transform_values,
    use_max_inverse_transform_values, scale_target_to_unit_interval_values
]

param_product = list(product(*params))

lambda_value_pl = 0.9
lambda_value_hinge = 0.5
epsilon_value_hinge = 1.0


def create_latex(df: pd.DataFrame, decimal_format="{:10.3f}"):
    result = "\\begin{tabular}{" + "l" + "r" * (len(df.columns)-1) + "} \n"
    result += "\\toprule \n"
    result += " & ".join(df.columns) + " \\\\ \n"
    result += "\\midrule \n"
    for index, row in df.iterrows():
        result += row[0] + " & " + " & ".join(["\\textbf{" + decimal_format.format(
            x) + "}" if x == row[1:].values.min() else decimal_format.format(x) for x in row[1:]]) + " \\\\ \n"
    result += "\\toprule \n"
    result += "\\end{tabular} \n"
    print(result)


def max_formatter(x):
    #     if x is None:
    #         return "blubb"
    if float(x) >= 0.8:
        return "$\\boldsymbol{" + str(x) + "}$"
    else:
        return str(x)


comparison_data = []
for scenario_name in scenario_names:
    df_baseline_rf = None
    df_baseline_lr = None
    df_baseline_sf = None
    df_corras_nnh = None
    df_corras_hinge_linear = None
    df_corras_hinge_quadratic = None
    df_corras_linear = None
    df_corras_quadratic = None
    try:
        df_baseline_lr = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-linear_regression" +
                                     scenario_name + ".csv")
        df_baseline_rf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-random_forest" +
                                     scenario_name + ".csv")
        df_baseline_sf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-survival-forest-" +
                                     scenario_name + ".csv")
        df_corras_all = pd.read_csv(evaluations_path +
                                    "corras-pl-log-linear-" + scenario_name +
                                    ".csv")
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
        df_corras_nnh_all = pd.read_csv(evaluations_path + "corras-hinge-nn-" +
                                        scenario_name + "-short.csv")
        df_corras_nnh = df_corras_nnh_all.loc[
            (df_corras_nnh_all["seed"] == seed)
            & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_nnh_all["learning_rate"] == 0.001) &
            (df_corras_nnh_all["lambda"] == lambda_value_hinge)]
        print(df_corras_nnh.columns)
        # dfs = [df_baseline_lr, df_baseline_rf,
        #        df_corras_linear, df_corras_quadratic]

    except:
        print("Scenario " + scenario_name +
              " not found in corras evaluation data!")
    val_rf = float("nan")
    val_lr = float("nan")
    val_sf = float("nan")
    val_pl_linear = float("nan")
    val_pl_quad = float("nan")
    val_nnh = float("nan")

    if df_baseline_rf is not None:
        val_rf = df_baseline_rf["run_status"].value_counts(normalize=True)["ok"]
    if df_baseline_lr is not None:
        val_lr = df_baseline_lr["run_status"].value_counts(normalize=True)["ok"]
    if df_baseline_sf is not None:
        val_sf = df_baseline_sf["run_status"].value_counts(normalize=True)["ok"]
    if df_baseline_sf is not None:
        val_baseline_sf = df_baseline_sf["run_status"].value_counts(normalize=True)["ok"]
    if df_corras_linear is not None:
        val_pl_linear = df_corras_linear["run_status"].value_counts(normalize=True)["ok"]
    if df_corras_quadratic is not None:
        val_pl_quad = df_corras_quadratic["run_status"].value_counts(normalize=True)["ok"]
    if df_corras_nnh is not None:
        val_nnh = df_corras_nnh["run_status"].value_counts(normalize=True)["ok"]
    comparison_data.append([
        scenario_name, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad, val_nnh
    ])
comparison_frame = pd.DataFrame(
    data=comparison_data,
    columns=["Scenario", "RF", "LR", "SF", "PL-Lin", "PL-Quad", "Hinge-NN"])

print(comparison_frame.to_latex())
# print(
#     comparison_frame.to_latex(na_rep="-",
#                               index=False,
#                               bold_rows=True,
#                               float_format="%.3f",
#                               formatters={"tau_corr": max_formatter},
#                               escape=False))

# comparison_data = []
# for scenario_name in scenario_names:
#     try:
#         df_baseline_lr = pd.read_csv(evaluations_path +
#                                      "baseline-evaluation-linear-regression" +
#                                      scenario_name + ".csv")
#         df_baseline_rf = pd.read_csv(evaluations_path +
#                                      "baseline-evaluation-random_forest" +
#                                      scenario_name + ".csv")
#         df_corras_all = pd.read_csv(evaluations_path +
#                                     "corras-pl-log-linear-" + scenario_name +
#                                     ".csv")
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
#         df_corras_hinge_linear_all = pd.read_csv(evaluations_path +
#                                                  "corras-hinge-linear-" +
#                                                  scenario_name + "-new" +
#                                                  ".csv")
#         df_corras_hinge_linear = df_corras_hinge_linear_all.loc[
#             (df_corras_hinge_linear_all["seed"] == seed)
#             & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
#             (df_corras_all["quadratic_transform"] == False) &
#             (df_corras_all["scale_to_unit_interval"] == True) &
#             (df_corras_all["max_inverse_transform"] == "max_cutoff")]
#         df_corras_hinge_quadratic = df_corras_hinge_linear_all.loc[
#             (df_corras_hinge_linear_all["seed"] == seed)
#             & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
#             (df_corras_all["quadratic_transform"] == True) &
#             (df_corras_all["scale_to_unit_interval"] == True) &
#             (df_corras_all["max_inverse_transform"] == "max_cutoff")]
#         df_corras_nnh_all = pd.read_csv(evaluations_path + "corras-hinge-nn-" +
#                                         scenario_name + "-short.csv")
#         df_corras_nnh = df_corras_nnh_all.loc[
#             (df_corras_nnh_all["seed"] == seed)
#             & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
#             (df_corras_nnh_all["learning_rate"] == 0.001) &
#             (df_corras_nnh_all["lambda"] == lambda_value_hinge) &
#             (df_corras_nnh_all["split"] == 5)]
#         print(len(df_corras_hinge_linear), len(df_corras_hinge_quadratic),
#               len(df_corras_linear), len(df_corras_quadratic),
#               len(df_corras_nnh), len(df_baseline_lr), len(df_baseline_rf))
#         print(df_corras_nnh)

#     except:
#         print("Scenario " + scenario_name +
#               " not found in corras evaluation data!")
#         continue
#     comparison_data.append([
#         scenario_name, df_baseline_rf["par10"].mean(),
#         df_baseline_lr["par10"].mean(), df_corras_linear["par10"].mean(),
#         df_corras_quadratic["par10"].mean(), df_corras_nnh["par10"].mean(),
#         df_corras_hinge_linear["par10"].mean(),
#         df_corras_hinge_quadratic["par10"].mean()
#     ])
# comparison_frame = pd.DataFrame(data=comparison_data,
#                                 columns=[
#                                     "Scenario", "Random Forest",
#                                     "Linear Regression", "pl linear",
#                                     "pl quadratic", "hinge nn", "hinge lin",
#                                     "hinge quad"
#                                 ])

comparison_data = []
for scenario_name in scenario_names:
    df_baseline_rf = None
    df_baseline_lr = None
    df_baseline_sf = None
    df_corras_nnh = None
    df_corras_hinge_linear = None
    df_corras_hinge_quadratic = None
    df_corras_linear = None
    df_corras_quadratic = None
    try:
        df_baseline_lr = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-linear_regression" +
                                     scenario_name + ".csv")
        df_baseline_rf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-random_forest" +
                                     scenario_name + ".csv")
        df_baseline_sf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-survival-forest-" +
                                     scenario_name + ".csv")
        df_corras_all = pd.read_csv(evaluations_path +
                                    "corras-pl-log-linear-" + scenario_name +
                                    ".csv")
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
        df_corras_nnh_all = pd.read_csv(evaluations_path + "corras-hinge-nn-" +
                                        scenario_name + "-short.csv")
        df_corras_hinge_linear = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_all["quadratic_transform"] == False) &
            (df_corras_all["scale_to_unit_interval"] == True) &
            (df_corras_all["max_inverse_transform"] == "max_cutoff")]
        df_corras_hinge_quadratic = df_corras_hinge_linear_all.loc[
            (df_corras_hinge_linear_all["seed"] == seed)
            & (df_corras_hinge_linear_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_all["quadratic_transform"] == True) &
            (df_corras_all["scale_to_unit_interval"] == True) &
            (df_corras_all["max_inverse_transform"] == "max_cutoff")]
        df_corras_nnh = df_corras_nnh_all.loc[
            (df_corras_nnh_all["seed"] == seed)
            & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_nnh_all["learning_rate"] == 0.001) &
            (df_corras_nnh_all["lambda"] == lambda_value_hinge)]
        print(df_corras_nnh.columns)
        # dfs = [df_baseline_lr, df_baseline_rf,
        #        df_corras_linear, df_corras_quadratic]

    except:
        print("Scenario " + scenario_name +
              " not found in corras evaluation data!")
    val_rf = float("nan")
    val_lr = float("nan")
    val_sf = float("nan")
    val_pl_linear = float("nan")
    val_pl_quad = float("nan")
    val_nnh = float("nan")
    val_hinge_linear = float("nan")
    val_hinge_quad = float("nan")
    if df_baseline_rf is not None:
        val_rf = df_baseline_rf["par10"].mean()
    if df_baseline_lr is not None:
        val_lr = df_baseline_lr["par10"].mean()
    if df_baseline_sf is not None:
        val_sf = df_baseline_sf["par10"].mean()
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
    comparison_data.append([
        scenario_name, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad, val_hinge_linear, val_hinge_quad, val_nnh
    ])
comparison_frame = pd.DataFrame(
    data=comparison_data,
    columns=["Scenario", "RF", "LR", "RSF", "PL-Lin", "PL-Quad", "Hinge-Lin", "Hinge-Quad", "Hinge-NN"])

create_latex(comparison_frame)


print(
    comparison_frame.to_latex(na_rep="-",
                              index=False,
                              bold_rows=True,
                              float_format="%.3f",
                              formatters={"tau_corr": max_formatter},
                              escape=False))

create_latex(comparison_frame)

comparison_data = []
for scenario_name in scenario_names:
    df_baseline_rf = float("nan")
    df_baseline_lr = float("nan")
    df_baseline_sf = float("nan")
    df_corras_nnh = float("nan")
    df_corras_hinge_linear = float("nan")
    df_corras_hinge_quadratic = float("nan")
    df_corras_linear = float("nan")
    df_corras_quadratic = float("nan")
    try:
        df_baseline_lr = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-linear_regression" +
                                     scenario_name + ".csv")
        df_baseline_rf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-random_forest" +
                                     scenario_name + ".csv")
        df_baseline_sf = pd.read_csv(evaluations_path +
                                     "baseline-evaluation-survival-forest-" +
                                     scenario_name + ".csv")
        df_corras_all = pd.read_csv(evaluations_path +
                                    "corras-pl-log-linear-" + scenario_name +
                                    ".csv")
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
        df_corras_nnh_all = pd.read_csv(evaluations_path + "corras-hinge-nn-" +
                                        scenario_name + "-short.csv")
        df_corras_nnh = df_corras_nnh_all.loc[
            (df_corras_nnh_all["seed"] == seed)
            & (df_corras_nnh_all["epsilon"] == epsilon_value_hinge) &
            (df_corras_nnh_all["learning_rate"] == 0.001) &
            (df_corras_nnh_all["lambda"] == lambda_value_hinge)]
        print(df_corras_nnh.columns)
        # dfs = [df_baseline_lr, df_baseline_rf,
        #        df_corras_linear, df_corras_quadratic]

    except:
        print("Scenario " + scenario_name +
              " not found in corras evaluation data!")
    val_rf = float("nan")
    val_lr = float("nan")
    val_sf = float("nan")
    val_pl_linear = float("nan")
    val_pl_quad = float("nan")
    val_nnh = float("nan")
    if df_baseline_rf is not None:
        val_rf = df_baseline_rf["tau_corr"].mean()
    if df_baseline_lr is not None:
        val_lr = df_baseline_lr["tau_corr"].mean()
    if df_baseline_sf is not None:
        val_sf = df_baseline_sf["tau_corr"].mean()
    if df_corras_linear is not None:
        val_pl_linear = df_corras_linear["tau_corr"].mean()
    if df_corras_quadratic is not None:
        val_pl_quad = df_corras_quadratic["tau_corr"].mean()
    if df_corras_nnh is not None:
        val_nnh = df_corras_nnh["tau_corr"].mean()
    comparison_data.append([
        scenario_name, val_rf, val_lr, val_sf, val_pl_linear, val_pl_quad, val_nnh
    ])
comparison_frame = pd.DataFrame(
    data=comparison_data,
    columns=["Scenario", "RF", "LR", "RSF", "PL-Lin", "PL-Quad", "Hinge-NN"])

create_latex(comparison_frame)
