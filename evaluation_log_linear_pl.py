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


def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(
        true_performances)
    return result


scenario_path = "./aslib_data-aslib-v4.0/"
results_path_corras = "./results-pl/"
evaluations_path = "./evaluations/"
figures_path = "./figures/"

scenarios = ["CPMP-2015", "MAXSAT-PMS-2016", "MAXSAT-WPMS-2016", "SAT11-INDU" "SAT12-ALL", "TTP-2016"]
# scenarios = ["MIP-2016"]
lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                 0.7, 0.8, 0.9, 0.99, 0.999, 1.0]
# lambda_values = [0.0, 0.2, 0.4, 0.6,
#                  0.8, 1.0]
# lambda_values = [0.5]
max_pairs_per_instance = 5
maxiter = 100
seeds = [15]
use_quadratic_transform_values = [True, False]
# use_quadratic_transform_values = [True]
use_max_inverse_transform_values = ["max_cutoff"]
# use_max_inverse_transform_values = ["max_cutoff"]
scale_target_to_unit_interval_values = [False]
# scale_target_to_unit_interval_values = [True]

splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

params = [
    lambda_values, splits, seeds, use_quadratic_transform_values,
    use_max_inverse_transform_values, scale_target_to_unit_interval_values
]

param_product = list(product(*params))

for scenario_name in scenarios:

    corras_measures = []

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)
    scenario.compute_rankings(False)
    relevance_scores = compute_relevance_scores_unit_interval(scenario)

    # params_string = "-".join([scenario_name,
    #     str(lambda_value), str(split), str(seed), str(use_quadratic_transform), str(use_max_inverse_transform), str(scale_target_to_unit_interval)])

    filename = "pl_log_linear" + "-" + scenario_name + ".csv"
    # loss_filename = "pl_log_linear" + "-" + params_string + "-losses.csv"
    filepath = results_path_corras + filename
    # print(filepath)
    # loss_filepath = results_path_corras + loss_filename
    corras = None
    try:
        corras = pd.read_csv(filepath)
    except Exception as exc:
        print("File for " + scenario_name +
              " not found in corras result data! Exception " + str(exc))
        continue
    # for lambda_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval in param_product:
    # print(corras.head())
    corras.set_index("problem_instance", inplace=True)
    performance_indices = [
        x for x in corras.columns if x.endswith("_performance")
    ]

    # lambda_values = pd.unique(corras["lambda"])
    # epsilon_values = pd.unique(corras["epsilon"])
    # print(lambda_values)
    # print(epsilon_values)

    # print(scenario.performance_data)
    # print(relevance_scores)

    for lambda_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval in param_product:
        current_frame = corras.loc[
            (corras["lambda"] == lambda_value) & (corras["split"] == split) &
            (corras["seed"] == seed) &
            (corras["use_quadratic_transform"] == use_quadratic_transform) &
            (corras["use_max_inverse_transform"] == use_max_inverse_transform)
            & (corras["scale_target_to_unit_interval"] ==
               scale_target_to_unit_interval)]
        # current_frame = corras.loc[(corras["lambda"] == lambda_value)]
        # print(current_frame)
        if current_frame.empty:
            continue
        for problem_instance, performances in scenario.performance_data.iterrows(
        ):
            if not problem_instance in current_frame.index:
                continue
            true_performances = scenario.performance_data.loc[
                problem_instance].astype("float64").to_numpy()
            true_ranking = scenario.performance_rankings.loc[
                problem_instance].astype("float64").to_numpy()
            run_stati = scenario.runstatus_data.loc[problem_instance]

            # print(current_frame.loc[problem_instance])
            feature_cost = 0
            # we use all features, so we sum up the individual costs
            if scenario.feature_cost_data is not None:
                feature_cost = scenario.feature_cost_data.loc[
                    problem_instance].sum()
            tau_corr = 0
            tau_p = 0
            ndcg = 0
            mse = 0
            mae = 0
            abs_vbs_distance = 0
            par10 = 0
            par10_with_feature_cost = 0
            # print(corras)
            corras_performances = current_frame.loc[problem_instance][
                performance_indices].astype("float64").to_numpy()
            corras_ranking = current_frame.loc[problem_instance][
                performance_indices].astype("float64").rank(
                    method="min").fillna(-1).astype("int16").to_numpy()
            if np.isinf(corras_performances).any():
                print("Warning, NaN in performance prediction for " +
                      problem_instance + "!")
                continue
            tau_corr, tau_p = kendalltau(true_ranking, corras_ranking)
            mse = mean_squared_error(true_performances, corras_performances)
            mae = mean_absolute_error(true_performances, corras_performances)
            abs_vbs_distance = compute_distance_to_vbs(corras_performances,
                                                       true_performances)
            ndcg = ndcg_at_k(corras_ranking,
                             relevance_scores.loc[problem_instance].to_numpy(),
                             len(scenario.algorithms))
            par10 = true_performances[np.argmin(corras_performances)]
            par10_with_feature_cost = par10 + feature_cost
            run_status = run_stati.iloc[np.argmin(corras_performances)]
            corras_measures.append([
                split, seed, problem_instance, lambda_value,
                use_quadratic_transform, use_max_inverse_transform,
                scale_target_to_unit_interval, tau_corr, tau_p, ndcg, mse, mae,
                abs_vbs_distance, par10, par10_with_feature_cost, run_status
            ])
            # print(corras_measures)
    df_corras = pd.DataFrame(
        data=corras_measures,
        columns=[
            "split", "seed", "problem_instance", "lambda",
            "quadratic_transform", "max_inverse_transform",
            "scale_to_unit_interval", "tau_corr", "tau_p", "ndcg", "mse",
            "mae", "abs_distance_to_vbs", "par10", "par10_with_feature_cost",
            "run_status"
        ])
    df_corras.to_csv(evaluations_path + "corras-pl-log-linear-" +
                     scenario_name + ".csv")
