import sys
import os.path

import autograd.numpy as np
import pandas as pd

from itertools import product

# evaluation stuff
from sklearn.model_selection import KFold
from scipy.stats import kendalltau

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Corras
from Corras.Model import log_linear
from Corras.Scenario import aslib_ranking_scenario
from Corras.Util import ranking_util as util

result_path = "./results-pl/"
loss_path = "./losses-pl/"

total_shards = int(sys.argv[1])
shard_number = int(sys.argv[2])

scenarios = ["MIP-2016", "CSP-2010", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND"]
# scenarios = ["MIP-2016"]
lambda_values = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                 0.7, 0.8, 0.9, 0.99, 0.999, 1.0]
epsilon_values = [0, 0.0001, 0.001, 0.01, 0.1,
                  0.2, 0.3]

# lambda_values = [0.0, 0.2, 0.4, 0.6,
#                  0.8, 1.0]
# lambda_values = [0.5]
max_pairs_per_instance = 5
maxiter = 100
seeds = [15]
use_quadratic_transform_values = [True, False]
# use_quadratic_transform_values = [True]
use_max_inverse_transform_values = ["none", "max_cutoff", "max_par10"]
# use_max_inverse_transform_values = ["max_cutoff"]
scale_target_to_unit_interval_values = [True, False]
# scale_target_to_unit_interval_values = [True]


splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# splits = [1]

params = [scenarios, lambda_values, splits, seeds, use_quadratic_transform_values,
          use_max_inverse_transform_values, scale_target_to_unit_interval_values]

param_product = list(product(*params))

shard_size = len(param_product) // total_shards

lower_bound = shard_number * shard_size
upper_bound = lower_bound + shard_size

shard = []
if shard_number == total_shards:
    shard = param_product[lower_bound:upper_bound]
else:
    shard = param_product[lower_bound:upper_bound]

for scenario_name, lambda_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval in shard:
    params_string = "-".join([scenario_name,
                              str(lambda_value), str(split), str(seed), str(use_quadratic_transform), str(use_max_inverse_transform), str(scale_target_to_unit_interval)])

    # filename = "pl_log_linear" + "-" + params_string + ".csv"
    filename = "pl_log_linear-" + scenario_name + ".csv"
    loss_filename = "pl_log_linear" + "-" + params_string + "-losses.csv"
    filepath = result_path + filename
    loss_filepath = loss_path + loss_filename
    exists = os.path.exists(filepath)
    result_data_corras = []
    try:
        scenario_path = "./aslib_data-aslib-v4.0/"+scenario_name
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario(scenario_path)
        if not exists:
            performance_cols_corras = [
                x + "_performance" for x in scenario.performance_data.columns]

            result_columns_corras = [
                "split", "problem_instance", "lambda", "seed", "use_quadratic_transform", "use_max_inverse_transform", "scale_target_to_unit_interval", "predicted_ranking"]
            result_columns_corras += performance_cols_corras

            results_corras = pd.DataFrame(
                data=[], columns=result_columns_corras)
            results_corras.to_csv(filepath, index_label="id")

        test_scenario, train_scenario = scenario.get_split(split)

        train_performances = train_scenario.performance_data
        train_features = train_scenario.feature_data
        # preprocessing
        imputer = SimpleImputer()
        polytransform = PolynomialFeatures(2)
        scaler = StandardScaler()

        # Impute
        train_features[train_features.columns] = imputer.fit_transform(
            train_features[train_features.columns])

        # Create polynomial features
        if use_quadratic_transform:
            quad_data = polytransform.fit_transform(train_features.to_numpy())
            new_cols = polytransform.get_feature_names(train_features.columns)
            train_features = pd.DataFrame(
                data=quad_data, index=train_features.index, columns=new_cols)

        # Standardize
        train_features[train_features.columns] = scaler.fit_transform(
            train_features[train_features.columns])

        # inst, perf, rank = util.construct_numpy_representation_with_pairs_of_rankings(
        #     train_features, train_performances, max_pairs_per_instance=max_pairs_per_instance, seed=seed)

        cutoff = scenario.algorithm_cutoff_time
        par10 = cutoff*10

        perf = train_performances.to_numpy()

        order = "asc"

        if use_max_inverse_transform == "max_cutoff":
            perf = perf.clip(0, cutoff)
            perf = cutoff - perf
            order = "desc"
        elif use_max_inverse_transform == "max_par10":
            perf = par10 - perf
            order = "desc"

        perf_max = 1

        if scale_target_to_unit_interval:
            perf_max = np.max(perf)
            perf = perf/perf_max

        train_performances = pd.DataFrame(
            data=perf, index=train_performances.index, columns=train_performances.columns)
        print(order)
        inst, perf, rank = util.construct_numpy_representation_with_pairs_of_rankings(
            train_features, train_performances, max_pairs_per_instance=max_pairs_per_instance, seed=seed, order=order)

        model = log_linear.LogLinearModel(
            use_exp_for_regression=False, use_reciprocal_for_regression=False)

        model.fit_np(len(scenario.algorithms), rank, inst,
                     perf, lambda_value=lambda_value, regression_loss="Squared", maxiter=maxiter, print_output=False, log_losses=True)

        for index, row in test_scenario.feature_data.iterrows():
            row_values = row.to_numpy().reshape(1, -1)

            # Impute
            imputed_row = imputer.transform(row_values)

            # Quadratic features
            if use_quadratic_transform:
                imputed_row = polytransform.transform(imputed_row)

            # Standardize
            scaled_row = scaler.transform(imputed_row).flatten()
            predicted_ranking = model.predict_ranking(scaled_row)
            predicted_performances = model.predict_performances(scaled_row)

            if scale_target_to_unit_interval:
                predicted_performances = perf_max * predicted_performances
            if use_max_inverse_transform == "max_cutoff":
                predicted_performances = cutoff - predicted_performances
            elif use_max_inverse_transform == "max_par10":
                predicted_performances = par10 - predicted_performances

            result_data_corras.append(
                [split, index, lambda_value, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval, predicted_ranking, *predicted_performances])
            # scenario_name, lambda_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval

        performance_cols_corras = [
            x + "_performance" for x in scenario.performance_data.columns]

        result_columns_corras = [
            "split", "problem_instance", "lambda", "seed", "use_quadratic_transform", "use_max_inverse_transform", "scale_target_to_unit_interval", "predicted_ranking"]
        result_columns_corras += performance_cols_corras
        results_corras = pd.DataFrame(
            data=result_data_corras, columns=result_columns_corras)
        results_corras.to_csv(filepath, index_label="id",
                              mode="a", header=False)
        model.save_loss_history(loss_filepath)

    except Exception as exc:
        print("Something went wrong during computation with parameters " +
              params_string + " message: " + str(exc))
