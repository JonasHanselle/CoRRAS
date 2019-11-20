import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.stats import kendalltau, describe

scenario_name = "MIP-2016"
scenario_path = "./aslib_data-aslib-v4.0/"
results_path = "./results/"

scenario = ASRankingScenario()
scenario.read_scenario(scenario_path + scenario_name)
scenario.compute_rankings(False)

baseline = pd.read_csv(results_path + "rf-" + scenario_name + ".csv")
corras = pd.read_csv(results_path + "corras-" + scenario_name + ".csv")
baseline.set_index("problem_instance", inplace=True)
corras.set_index("problem_instance", inplace=True)
performance_indices = [x for x in corras.columns if x.endswith("_performance")]

baseline_rankings = baseline[performance_indices].rank(axis=1, method="min").fillna(-1).astype("int16")
# corras_rankings = corras[performance_indices].rank(axis=1, method="min").fillna(-1).astype("int16")

taus_baseline = []
lambda_values = pd.unique(corras["lambda"])

print("lambdas", lambda_values)

for problem_instance, performances in scenario.performance_data.iterrows():
    true_ranking = scenario.performance_rankings.loc[problem_instance]
    baseline_ranking = baseline_rankings.loc[problem_instance]
    taus_baseline.append(kendalltau(true_ranking, baseline_ranking))
    for lambda_value in lambda_values:
        corras_ranking = corras.loc[]
        

print(describe(taus_baseline).mean)