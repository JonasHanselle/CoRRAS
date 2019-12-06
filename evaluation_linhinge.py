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

sns.set_style("darkgrid")

scenario_path = "./aslib_data-aslib-v4.0/"
results_path_corras = "./results/results-hinge/"
evaluations_path = "./evaluations/"
figures_path = "./figures/"
scenario_names = ["SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND"]

def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(true_performances)
    return result

for scenario_name in scenario_names:

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)
    scenario.compute_rankings(False)
    relevance_scores = compute_relevance_scores_unit_interval(scenario)

    try:
        corras = pd.read_csv(results_path_corras + "corras-linhinge-" + scenario_name + ".csv")
    except:
        print("Scenario " + scenario_name + " not found in corras result data!")
        continue
    corras.set_index("problem_instance", inplace=True)
    performance_indices = [x for x in corras.columns if x.endswith("_performance")]

    lambda_values = pd.unique(corras["lambda"])
    epsilon_values = pd.unique(corras["epsilon"])
    print(lambda_values)
    print(epsilon_values)

    corras_measures = []

    print(scenario.performance_data)
    print(relevance_scores)

    for problem_instance, performances in scenario.performance_data.iterrows():
        true_performances = scenario.performance_data.loc[problem_instance].astype("float64").to_numpy()
        true_ranking = scenario.performance_rankings.loc[problem_instance].astype("float64").to_numpy()
        # print(corras.loc[problem_instance])
        for lambda_value, epsilon_value in  product(lambda_values,epsilon_values):
            tau_corr = 0
            tau_p = 0
            ndcg = 0
            mse = 0
            mae = 0
            abs_vbs_distance = 0
            par10 = 0
            # print(corras)
            corras_performances = corras.loc[(corras["lambda"] == lambda_value) & (corras["epsilon"] == epsilon_value)].loc[problem_instance][performance_indices].astype("float64").to_numpy()
            # print(corras.loc[problem_instance])
            corras_ranking = corras.loc[(corras["lambda"] == lambda_value) & (corras["epsilon"] == epsilon_value)].loc[problem_instance][performance_indices].astype("float64").rank(method="min").fillna(-1).astype("int16").to_numpy()
            if np.isinf(corras_performances).any():
                print("Warning, NaN in performance prediction for " + problem_instance + "!")
                continue
            # print(true_ranking, corras_ranking)
            tau_corr, tau_p = kendalltau(true_ranking, corras_ranking)
            mse = mean_squared_error(true_performances, corras_performances)
            mae = mean_absolute_error(true_performances, corras_performances)
            abs_vbs_distance = compute_distance_to_vbs(corras_performances, true_performances)
            ndcg = ndcg_at_k(corras_ranking,relevance_scores.loc[problem_instance].to_numpy(), len(scenario.algorithms))
            par10 = true_performances[np.argmin(corras_performances)]
            corras_measures.append([problem_instance,lambda_value,epsilon_value,tau_corr,tau_p,ndcg,mse,mae,abs_vbs_distance,par10])

    df_corras = pd.DataFrame(data=corras_measures,columns=["problem_instance", "lambda", "epsilon", "tau_corr", "tau_p", "ndcg", "mse", "mae","abs_distance_to_vbs", "par10"])
    print(df_corras.head())
    df_corras.to_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")