import sys

import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from itertools import product

# Database
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import exists, select, and_, or_
import urllib

# measures
from scipy.stats import kendalltau, describe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Corras.Evaluation.evaluation import ndcg_at_k, compute_relevance_scores_unit_interval

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# sksurv
# from sksurv.ensemble import RandomSurvivalForest

sns.set_style("darkgrid")


def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(
        true_performances)
    return result


scenario_path = "./aslib_data-aslib-v4.0/"
results_path_corras = "./results-nnh-new/"
evaluations_path = "./evaluations/"
figures_path = "./figures/"
# DB data
db_url = sys.argv[1]
db_user = sys.argv[2]
db_pw = urllib.parse.quote_plus(sys.argv[3])
db_db = sys.argv[4]

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

lambda_values = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
epsilon_values = [1.0]
max_pairs_per_instance = 5
maxiter = 1000
seeds = [1,2,3,4,5,15]

learning_rates = [0.001]
batch_sizes = [128]
es_patiences = [8]
es_intervals = [8]
es_val_ratios = [0.3]
layer_sizes_vals = ["[32]"]
activation_functions = ["sigmoid"]
use_max_inverse_transform_values = ["max_cutoff"]
scale_target_to_unit_interval_values = [True]
use_weighted_samples_values = [False]

splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

params = [
    lambda_values, epsilon_values, splits, seeds, learning_rates, es_intervals,
    es_patiences, es_val_ratios, batch_sizes, layer_sizes_vals,
    activation_functions, use_weighted_samples_values,
    scale_target_to_unit_interval_values, use_max_inverse_transform_values
]

param_product = list(product(*params))

for scenario_name in scenarios:

    corras_measures = []

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)
    scenario.compute_rankings(False)
    relevance_scores = compute_relevance_scores_unit_interval(scenario)

    # params_string = "-".join([scenario_name,
    # str(lambda_value), str(split), str(seed), str(use_quadratic_transform),
    # str(use_max_inverse_transform), str(scale_target_to_unit_interval)])

    filename = "ki2020_nnh" + "-" + scenario_name
    # loss_filename = "pl_log_linear" + "-" + params_string + "-losses.csv"
    filepath = results_path_corras + filename
    # print(filepath)
    # loss_filepath = results_path_corras + loss_filename
    corras = None
    try:
        table_name = "ki2020_nnh-" + scenario_name

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

    # print(corras.head())

    for     lambda_value, epsilon_value, split, seed, learning_rate, es_interval, es_patience, es_val_ratio, batch_size, layer_sizes, activation_function, use_weighted_samples, scale_target_to_unit_interval, use_max_inverse_transform in param_product:
        # print(
        #     f"lambda {lambda_value} epsilon {epsilon_value} split {split} seed {seed} learning_rate {learning_rate} batch_size {batch_size} es_patience {es_patience} es_interval {es_interval} es_val_ratio {es_val_ratio} layer_sizes {layer_sizes} activation_function {activation_function} use_weighted_samples {use_weighted_samples} scale_target_to_unit_interval {scale_target_to_unit_interval} use_max_inverse_transform {use_max_inverse_transform}"
        # )
        test_scenario, train_scenario = scenario.get_split(split)

        current_frame = corras.loc[
            (corras["lambda"] == lambda_value)
            & (corras["epsilon"] == epsilon_value) & (corras["split"] == split)
            & (corras["seed"] == seed) &
            (corras["learning_rate"] == learning_rate) &
            (corras["es_interval"] == es_interval) &
            (corras["es_patience"] == es_patience) &
            (corras["es_val_ratio"] == es_val_ratio) &
            (corras["batch_size"] == batch_size) &
            (corras["layer_sizes"] == layer_sizes) &
            (corras["use_weighted_samples"] == use_weighted_samples) &
            (corras["scale_target_to_unit_interval"] ==
             scale_target_to_unit_interval) &
            (corras["use_max_inverse_transform"] == use_max_inverse_transform)
            & (corras["activation_function"] == activation_function)]
        # current_frame = corras.loc[(corras["lambda"] == lambda_value)]
        # print(current_frame)
        if len(current_frame) != len(test_scenario.performance_data):
            print(f"The frame contains {len(current_frame)} entries, but the {scenario_name} contains {len(test_scenario.performance_data)} entries in split {split} with seed {seed}!")
            continue      
        # continue

        # print(len(current_frame), len(scenario.instances))
        for problem_instance, performances in scenario.performance_data.iterrows(
        ):
            if not problem_instance in current_frame.index:
                continue
            true_performances = scenario.performance_data.loc[
                problem_instance].astype("float64").to_numpy()
            true_ranking = scenario.performance_rankings.loc[
                problem_instance].astype("float64").to_numpy()

            feature_cost = 0
            # we use all features, so we sum up the individual costs
            if scenario.feature_cost_data is not None:
                feature_cost = scenario.feature_cost_data.loc[
                    problem_instance].sum()
            # print(current_frame.loc[problem_instance])
            tau_corr = 0
            tau_p = 0
            ndcg = 0
            mse = 0
            mae = 0
            abs_vbs_distance = 0
            par10 = 0
            par10_with_feature_cost = 0
            run_stati = scenario.runstatus_data.loc[problem_instance]
            # print(corras)
            corras_performances = current_frame.loc[problem_instance][
                performance_indices].astype("float64").to_numpy()
            # if (len(true_performances) != len(corras_performances)):
            #     corras_performances = corras_performances[0]
            #     continue
            corras_ranking = current_frame.loc[problem_instance][
                performance_indices].astype("float64").rank(
                    method="min").astype("int16").to_numpy()
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
                split, seed, problem_instance, lambda_value, epsilon_value,
                learning_rate, es_interval, es_patience, es_val_ratio,
                batch_size, layer_sizes, activation_function,
                use_weighted_samples,
                scale_target_to_unit_interval,
                use_max_inverse_transform, tau_corr, tau_p, ndcg, mse,
                mae, abs_vbs_distance, par10, par10_with_feature_cost,
                run_status
            ])
            # print(corras_measures)
    df_corras = pd.DataFrame(
        data=corras_measures,
        columns=[
            "split", "seed", "problem_instance", "lambda", "epsilon",
            "learning_rate", "es_interval", "es_patience", "es_val_ratio",
            "batch_size", "layer_sizes", "activation_function",
            "use_weighted_samples",
            "scale_target_to_unit_interval",
            "use_max_inverse_transform", "tau_corr", "tau_p", "ndcg",
            "mse", "mae", "abs_distance_to_vbs", "par10",
            "par10_with_feature_cost", "run_status"
        ])
    df_corras.to_csv(evaluations_path + "ki2020-nnh-" + scenario_name +
                     ".csv")
