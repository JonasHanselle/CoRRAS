import sys

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

# Database
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import exists, select, and_, or_
import urllib


def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(
        true_performances)
    return result


scenario_path = "./aslib_data-aslib-v4.0/"
results_path_corras = "./results-pl/"
evaluations_path = "./evaluations/"
figures_path = "./figures/"

# DB data
db_url = sys.argv[1]
db_user = sys.argv[2]
db_pw = urllib.parse.quote_plus(sys.argv[3])
db_db = sys.argv[4]

seeds = [15]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


scenarios = [
    "CPMP-2015",
    "MIP-2016",
    "CSP-2010",
    "SAT11-HAND",
    "SAT11-INDU",
    "SAT11-RAND",
    # "CSP-Minizinc-Time-2016",
    # "MAXSAT-WPMS-2016",
    # "MAXSAT-PMS-2016",
    # "QBF-2016"
]
lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# scenarios = ["CPMP-2015"]
max_pairs_per_instance = 5
maxiter = 100
seeds = [15]
use_quadratic_transform_values = [False, True]
# use_quadratic_transform_values = [True]
use_max_inverse_transform_values = ["max_cutoff"]
# use_max_inverse_transform_values = ["max_cutoff"]
scale_target_to_unit_interval_values = [True]
# scale_target_to_unit_interval_values = [True]
regularization_params_values = [0.001]
use_weighted_samples_values = [False]

params = [
    splits, lambda_values, seeds, use_quadratic_transform_values,
    use_max_inverse_transform_values, scale_target_to_unit_interval_values,
    use_weighted_samples_values, regularization_params_values
]

param_product = list(product(*params))

for scenario_name in scenarios:
    print(scenario_name)
    corras_measures = []

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)
    scenario.compute_rankings(False)
    relevance_scores = compute_relevance_scores_unit_interval(scenario)

    corras = None
    try:
        if scenario_name == "CSP-Minizinc-Time-2016":
            table_name = "linear-plackett-luce-new" + "CSP-MT-2016" + "-weighted-iter100-fix"
        else:
            table_name = "linear-plackett-luce-new" + scenario_name + "-weighted-iter100-fix"

        engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" +
                                   db_url + "/" + db_db,
                                   echo=False)
        connection = engine.connect()
        corras = pd.read_sql_table(table_name=table_name, con=connection)
        connection.close()
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

    for split, lambda_value, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval, use_weighted_samples, regularization_param in param_product:

        test_scenario, train_scenario = scenario.get_split(split)

        current_frame = corras.loc[
            (corras["lambda"] == lambda_value) & (corras["split"] == split) &
            (corras["seed"] == seed) &
            (corras["use_quadratic_transform"] == use_quadratic_transform) &
            (corras["use_max_inverse_transform"] == use_max_inverse_transform)
            & (corras["scale_target_to_unit_interval"] ==
               scale_target_to_unit_interval)
            & (corras["use_weighted_samples"] == use_weighted_samples) &
            (corras["regularization_param"] == regularization_param)]

        if len(current_frame) != len(test_scenario.performance_data):
            print(
                f"The frame contains {len(current_frame)} entries, but the {scenario_name} contains {len(test_scenario.performance_data)} entries! For lambda = {lambda_value} and quadratic transform = {use_quadratic_transform}"
            )
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
                scale_target_to_unit_interval, use_weighted_samples,
                regularization_param, tau_corr, tau_p, ndcg, mse, mae,
                abs_vbs_distance, par10, par10_with_feature_cost, run_status
            ])
            # print(corras_measures)
    df_corras = pd.DataFrame(
        data=corras_measures,
        columns=[
            "split", "seed", "problem_instance", "lambda",
            "quadratic_transform", "max_inverse_transform",
            "scale_to_unit_interval", "use_weighted_samples",
            "regularization_param", "tau_corr", "tau_p", "ndcg", "mse", "mae",
            "abs_distance_to_vbs", "par10", "par10_with_feature_cost",
            "run_status"
        ])
    df_corras.to_csv(evaluations_path + "corras-pl-log-linear-" +
                     scenario_name + "-new-short.csv")
