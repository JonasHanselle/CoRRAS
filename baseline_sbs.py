import sys

import autograd.numpy as np
import pandas as pd

# ASLib
import aslib_scenario.aslib_scenario as as_scenario

# Baseline
from sklearn.linear_model import LinearRegression

scenario_path = "./aslib_data-aslib-v4.0/"
evaluations_path = "./evaluations/"
result_path = "./results/"

num_splits = 10
result_data_corras = []
baselines = None

scenario_names = [
    "SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND",
    "CPMP-2015", "QBF-2016", "SAT12-ALL", "MAXSAT-WPMS-2016",
    "MAXSAT-PMS-2016", "CSP-Minizinc-Time-2016", "SAT12-HAND", "SAT12-INDU", "SAT12-RAND"
]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


for scenario_name in scenario_names:
    scenario = as_scenario.ASlibScenario()
    scenario.read_scenario(scenario_path + scenario_name)

    for index, row in scenario.runstatus_data.iterrows():
        if not np.any(row.to_numpy() == "ok"):
            scenario.performance_data.drop(index)
            scenario.performance_data.drop(index)

    for i_split in splits:

        test_scenario, train_scenario = scenario.get_split(i_split)

        sbs_par10 = train_scenario.performance_data.sum().idxmin()
        success_rates = {}
        for algorithm in scenario.algorithms:
            success_rate = train_scenario.runstatus_data[algorithm].value_counts(normalize=True)[
                "ok"]
            success_rates[algorithm] = success_rate
        sbs_succ = max(success_rates, key=success_rates.get)
        result_data_sbs = []
        for index, row in test_scenario.feature_data.iterrows():
            par10_sbs_par10 = test_scenario.performance_data.loc[index][sbs_par10]
            par10_sbs_succ = test_scenario.performance_data.loc[index][sbs_succ]
            success_rate_sbs_succ = test_scenario.runstatus_data[sbs_succ].value_counts(normalize=True)[
                "ok"]
            success_rate_sbs_par10 = test_scenario.runstatus_data[sbs_par10].value_counts(normalize=True)[
                "ok"]
            result_data_sbs.append(
                [i_split, index, sbs_par10, sbs_succ, par10_sbs_par10, par10_sbs_succ, success_rate_sbs_par10, success_rate_sbs_succ])

    result_columns_sbs = ["split", "problem_instance",
                          "sbs_par10", "sbs_succ", "par10_sbs_par10", "par10_sbs_succ", "success_rate_sbs_par10", "success_rate_sbs_succ"]
    results_sbs = pd.DataFrame(data=result_data_sbs,
                               columns=result_columns_sbs)
    results_sbs.to_csv(evaluations_path + "sbs-" + scenario.scenario + ".csv",
                       index_label="id")

    # print("sbs", scenario_name, scenario.performance_data.mean().idxmin())
    sbs_perf = []
    vbs_perf = []
    for index, row in scenario.performance_data.iterrows():
        vbs_perf.append(row.min())

    print("sbs", scenario_name, results_sbs["par10_sbs_par10"].mean())
    print("vbs", scenario_name, np.mean(vbs_perf))
