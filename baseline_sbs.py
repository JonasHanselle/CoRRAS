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

        sbs = train_scenario.performance_data.sum().idxmin()
        print(sbs)
        result_data_sbs = []
        for index, row in test_scenario.feature_data.iterrows():
            performance = test_scenario.performance_data.loc[index][sbs]
            result_data_sbs.append(
                [i_split, index, sbs, performance])

    result_columns_sbs = ["split", "problem_instance", "algorithm", "performance"]
    results_sbs = pd.DataFrame(data=result_data_sbs,
                                columns=result_columns_sbs)
    results_sbs.to_csv(result_path + "sbs-" + scenario.scenario + ".csv",
                        index_label="id")

    print(scenario_name, "performance", results_sbs[["performance"]].mean())

    # print("sbs", scenario_name, scenario.performance_data.mean().idxmin())
    sbs_perf = []
    vbs_perf = []
    for index, row in scenario.performance_data.iterrows():
        vbs_perf.append(row.min())
    

    print("sbs", scenario_name, np.mean(sbs_perf))
    print("vbs", scenario_name, np.mean(vbs_perf))