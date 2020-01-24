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
evaluations_path = "./evaluations/"
evaluations_path_nnh = "./evaluations-nnh-config/"

figures_path = "./figures/progression-plots/"

seed = 15

scenario_names = ["MIP-2016", "SAT12-ALL", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND"]


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



comparison_data_tau = []
for scenario_name in scenario_names:

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)

    try:
        df_corras_plnet = pd.read_csv(
            evaluations_path + "corras-pl-nn-" + scenario_name + ".csv")
        df_corras_plnet_1616_1_sigmoid = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[16, 16]") &
            (df_corras_plnet["lambda"] == 1.0)]
        df_corras_plnet_32_1_sigmoid = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 1.0)]
        df_corras_plnet_1616_05_sigmoid = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[16, 16]") &
            (df_corras_plnet["lambda"] == 0.5)]
        df_corras_plnet_32_05_sigmoid = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "sigmoid") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 0.5)]
        df_corras_plnet_1616_1_relu = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "relu") &
            (df_corras_plnet["layer_sizes"] == "[16, 16]") &
            (df_corras_plnet["lambda"] == 1.0)]
        df_corras_plnet_32_1_relu = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "relu") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 1.0)]
        df_corras_plnet_1616_05_relu = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "relu") &
            (df_corras_plnet["layer_sizes"] == "[16, 16]") &
            (df_corras_plnet["lambda"] == 0.5)]
        df_corras_plnet_32_05_relu = df_corras_plnet.loc[
            (df_corras_plnet["seed"] == seed) &
            (df_corras_plnet["activation_function"] == "relu") &
            (df_corras_plnet["layer_sizes"] == "[32]") &
            (df_corras_plnet["lambda"] == 0.5)]
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

    comparison_data_tau.append([
        scenario_name, df_corras_plnet_1616_1_sigmoid["tau_corr"].mean(), df_corras_plnet_1616_05_sigmoid["tau_corr"].mean(), df_corras_plnet_32_1_sigmoid["tau_corr"].mean(), df_corras_plnet_32_05_sigmoid["tau_corr"].mean(), df_corras_plnet_1616_1_relu["tau_corr"].mean(), df_corras_plnet_1616_05_relu["tau_corr"].mean(), df_corras_plnet_32_1_relu["tau_corr"].mean(), df_corras_plnet_32_05_relu["tau_corr"].mean()])
comparison_frame_tau = pd.DataFrame(data=comparison_data_tau,
                                    columns=[
                                        "Scenario", "1616_1_sigmoid", "1616_05_sigmoid", "32_1_sigmoid", "32_05_sigmoid",  "1616_1_relu", "1616_05_relu", "32_1_relu", "32_05_relu"
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
