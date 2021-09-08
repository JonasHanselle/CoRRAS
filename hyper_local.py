import sys
import os.path

import autograd.numpy as np
import pandas as pd

from itertools import product

# model selection and evaluation stuff
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import kendalltau

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Corras
import Corras.Model.neural_net_hinge as nn_hinge
import Corras.Model.neural_net as nn_pl
from Corras.Scenario import aslib_ranking_scenario
from Corras.Util import ranking_util as util

from statistics import mean

# hyperparameter optimization
from skopt import forest_minimize, gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_evaluations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Database
import urllib

result_path = "./results-nnh-new/"
loss_path = "./losses-nnh-new/"

# print(sys.argv)

total_shards = int(sys.argv[1])
shard_number = int(sys.argv[2])

# DB data
# db_url = sys.argv[3]
# db_user = sys.argv[4]
# db_pw = urllib.parse.quote_plus(sys.argv[5])
# db_db = sys.argv[6]

scenarios = [
    "CPMP-2015", "MIP-2016", "CSP-2010", "SAT11-HAND", "SAT11-INDU",
    "SAT11-RAND"
]

scenarios = ["MIP-2016"]

epsilon_value = 1.0
max_pairs_per_instance = 5
maxiter = 200

learning_rate = 0.001
batch_size = 128
es_patience = 8
es_interval = 1
es_val_ratio = 0.3
layer_sizes_val = [32]
activation_function = "sigmoid"
use_max_inverse_transform_value = "max_cutoff"
scale_target_to_unit_interval_value = True
use_weighted_samples_value = False

lambda_value = 0.5

splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
splits = [2]

seeds = [1, 2, 3, 4, 5]
seeds = [2]

params = [scenarios, splits, seeds]

param_product = list(product(*params))

shard_size = len(param_product) // total_shards

lower_bound = shard_number * shard_size
upper_bound = lower_bound + shard_size

shard = []
if shard_number == total_shards:
    shard = param_product[lower_bound:upper_bound]
else:
    shard = param_product[lower_bound:upper_bound]

for scenario_name, split, seed in shard:

    # table_name = "neural-net-squared-hinge-" + scenario_name + "-seeded"
    table_name = "hyper-test-" + scenario_name

    params_string = "-".join([
        scenario_name,
        str(lambda_value),
        str(epsilon_value),
        str(split),
        str(seed),
        str(learning_rate),
        str(es_interval),
        str(es_patience),
        str(es_val_ratio),
        str(batch_size)
    ])

    # filename = "pl_log_linear" + "-" + params_string + ".csv"
    filename = "nn_hinge-" + scenario_name + ".csv"
    loss_filename = "nn_hinge" + "-" + params_string + "-losses.csv"
    es_val_filename = "nn_hinge" + "-" + params_string + "-es-val.csv"
    filepath = result_path + filename
    loss_filepath = loss_path + loss_filename
    es_val_filepath = loss_path + es_val_filename
    exists = os.path.exists(filepath)
    result_data_corras = []
    try:
        scenario_path = "./aslib_data-aslib-v4.0/" + scenario_name
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario(scenario_path)
        if not exists:
            performance_cols_corras = [
                x + "_performance" for x in scenario.performance_data.columns
            ]

            result_columns_corras = [
                "split", "problem_instance", "lambda", "epsilon", "seed",
                "learning_rate", "es_interval", "es_patience", "es_val_ratio",
                "batch_size", "layer_sizes", "activation_function"
            ]
            result_columns_corras += performance_cols_corras

            results_corras = pd.DataFrame(data=[],
                                          columns=result_columns_corras)
            results_corras.to_csv(filepath, index_label="id")

        test_scenario, train_scenario = scenario.get_split(split)

        train_performances = train_scenario.performance_data
        train_features = train_scenario.feature_data

        # hyperparameter optimization 

        def hyopt():
            print("HYPERPAREMETER OPTIMIZATION")
            train_features_hyopt = train_scenario.feature_data
            train_target_hyopt = train_scenario.performance_data

            X_train, X_val, y_train, y_val = train_test_split(
                train_features_hyopt, train_target_hyopt, test_size=0.33, random_state=seed)
            
            print("X_train", X_train)
            print("y_train", y_train)
            print("X_val", X_val)
            print("y_val", y_val)

            # preprocessing
            imputer = SimpleImputer()
            scaler = StandardScaler()

            # Impute
            X_train[X_train.columns] = imputer.fit_transform(
                X_train[X_train.columns])

            # Standardize
            X_train[X_train.columns] = scaler.fit_transform(
                X_train[X_train.columns])

            cutoff = scenario.algorithm_cutoff_time
            par10 = cutoff * 10

            perf = y_train.to_numpy()

            order = "asc"

            if use_max_inverse_transform_value == "max_cutoff":
                perf = perf.clip(0, cutoff)
                perf = cutoff - perf
                order = "desc"
            elif use_max_inverse_transform_value == "max_par10":
                perf = par10 - perf
                order = "desc"

            perf_max = 1

            if scale_target_to_unit_interval_value:
                perf_max = np.max(perf)
                perf = perf / perf_max

            # print("perf", perf)

            train_target_hyopt = pd.DataFrame(data=perf,
                                            index=y_train.index,
                                            columns=y_train.columns)
            # print(order)
            inst, perf, rank, sample_weights = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(
                X_train,
                train_target_hyopt,
                max_pairs_per_instance=max_pairs_per_instance,
                seed=seed,
                order=order,
                skip_value=None)

            sample_weights = sample_weights / sample_weights.max()
            if not use_weighted_samples_value:
                sample_weights = np.ones(len(sample_weights))
            # print("sample weights", sample_weights)

            rank = rank.astype("int32")
            search_space = [(Real(0.0, 1.0, "uniform", name='lambda_value'))]
            @use_named_args(search_space)
            def fit_and_predict(**params):
                lambda_value = params["lambda_value"]
                print("current lambda: ", lambda_value)
                model = nn_hinge.NeuralNetworkSquaredHinge()
                model.fit(len(scenario.algorithms),
                    rank,
                    inst,
                    perf,
                    lambda_value=lambda_value,
                    regression_loss="Squared",
                    num_epochs=maxiter,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    seed=seed,
                    patience=es_patience,
                    es_val_ratio=es_val_ratio,
                    reshuffle_buffer_size=1000,
                    early_stop_interval=es_interval,
                    log_losses=False,
                    activation_function=activation_function,
                    hidden_layer_sizes=layer_sizes_val,
                    sample_weights=sample_weights)
                par10s = []
                tau_corrs = []
                for index, row in X_train.iterrows():
                    row_values = row.to_numpy().reshape(1, -1)

                    # Impute
                    imputed_row = imputer.transform(row_values)

                    # Standardize
                    scaled_row = scaler.transform(imputed_row).flatten()
                    # predicted_ranking = model.predict_ranking(scaled_row)
                    predicted_performances = model.predict_performances(scaled_row)
                    # scenario.performance_data.loc[
                    #     index].astype("float64").to_numpy()

                    if scale_target_to_unit_interval_value:
                        predicted_performances = perf_max * predicted_performances
                    if use_max_inverse_transform_value == "max_cutoff":
                        predicted_performances = cutoff - predicted_performances
                    elif use_max_inverse_transform_value == "max_par10":
                        predicted_performances = - predicted_performances
                        # predicted_performances = par10 - predicted_performances
                    true_performances = y_train.loc[
                        index].astype("float64").to_numpy()
                    print("true performances", true_performances)
                    true_ranking = y_train.loc[
                        index].astype("float64").to_numpy()

                    # calculate rank correlation
                    corras_ranking = np.argsort(np.argsort(predicted_performances))
                    tau_corr, tau_p = kendalltau(true_ranking, corras_ranking)
                    tau_corrs.append(tau_corr)
                    par10s.append(true_performances[np.argmin(predicted_performances)])
                par10 = mean(par10s)
                tau = mean(tau_corrs)
                print("par10", par10)
                print("tau", tau)

                return par10
                # return -tau

            # fit_and_predict(lambda_value=0.5)
            # Minimize using SMBO with random forests
            # minimizer = forest_minimize(fit_and_predict,search_space,n_calls=25, acq_func="LCB", x0=[[0.1],[0.3],[0.5],[0.7],[0.9]])
            minimizer = forest_minimize(fit_and_predict,search_space,n_calls=50, acq_func="LCB", n_jobs=6, n_initial_points=10, initial_point_generator="sobol")
            # plot convergence of objective
            plot_convergence(minimizer)
            plt.savefig("convergence.pdf")
            plot_evaluations(minimizer)
            plt.savefig("evaluations.pdf")
            print("x iters: ", minimizer.x_iters)
        hyopt()

        train_performances = train_scenario.performance_data
        train_features = train_scenario.feature_data

        # preprocessing
        imputer = SimpleImputer()
        scaler = StandardScaler()

        # Impute
        train_features[train_features.columns] = imputer.fit_transform(
            train_features[train_features.columns])

        # Standardize
        train_features[train_features.columns] = scaler.fit_transform(
            train_features[train_features.columns])

        cutoff = scenario.algorithm_cutoff_time
        par10 = cutoff * 10

        perf = train_performances.to_numpy()

        order = "desc"

        if use_max_inverse_transform_value == "max_cutoff":
            perf = perf.clip(0, cutoff)
            perf = cutoff - perf
            order = "desc"
        elif use_max_inverse_transform_value == "max_par10":
            perf = par10 - perf
            order = "desc"

        perf_max = 1

        if scale_target_to_unit_interval_value:
            perf_max = np.max(perf)
            perf = perf / perf_max

        # print("perf", perf)

        train_performances = pd.DataFrame(data=perf,
                                          index=train_performances.index,
                                          columns=train_performances.columns)
        # print(order)
        inst, perf, rank, sample_weights = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(
            train_features,
            train_performances,
            max_pairs_per_instance=max_pairs_per_instance,
            seed=seed,
            order=order,
            skip_value=None)

        sample_weights = sample_weights / sample_weights.max()
        if not use_weighted_samples_value:
            sample_weights = np.ones(len(sample_weights))
        # print("sample weights", sample_weights)

        rank = rank.astype("int32")

        # model = nn_hinge.NeuralNetworkSquaredHinge()
        # model.fit(len(scenario.algorithms),
        #           rank,
        #           inst,
        #           perf,
        #           lambda_value=lambda_value,
        #           epsilon_value=epsilon_value,
        #           regression_loss="Squared",
        #           num_epochs=maxiter,
        #           learning_rate=learning_rate,
        #           batch_size=batch_size,
        #           seed=seed,
        #           patience=es_patience,
        #           es_val_ratio=es_val_ratio,
        #           reshuffle_buffer_size=1000,
        #           early_stop_interval=es_interval,
        #           log_losses=False,
        #           activation_function=activation_function,
        #           hidden_layer_sizes=layer_sizes_val,
        #           sample_weights=sample_weights)
        model = nn_hinge.NeuralNetworkSquaredHinge()
        model.fit(len(scenario.algorithms),
                  rank,
                  inst,
                  perf,
                  lambda_value=lambda_value,
                  regression_loss="Squared",
                  num_epochs=maxiter,
                  learning_rate=learning_rate,
                  batch_size=batch_size,
                  seed=seed,
                  patience=es_patience,
                  es_val_ratio=es_val_ratio,
                  reshuffle_buffer_size=1000,
                  early_stop_interval=es_interval,
                  log_losses=False,
                  activation_function=activation_function,
                  hidden_layer_sizes=layer_sizes_val,
                  sample_weights=sample_weights)

        for index, row in test_scenario.feature_data.iterrows():
            row_values = row.to_numpy().reshape(1, -1)

            # Impute
            imputed_row = imputer.transform(row_values)

            # Standardize
            scaled_row = scaler.transform(imputed_row).flatten()
            # predicted_ranking = model.predict_ranking(scaled_row)
            predicted_performances = model.predict_performances(scaled_row)

            if scale_target_to_unit_interval_value:
                predicted_performances = perf_max * predicted_performances
            if use_max_inverse_transform_value == "max_cutoff":
                predicted_performances = cutoff - predicted_performances
            elif use_max_inverse_transform_value == "max_par10":
                predicted_performances = par10 - predicted_performances

            # selected algorithm 

            result_data_corras.append([
                split, index, lambda_value, epsilon_value, seed, learning_rate,
                es_interval, es_patience, es_val_ratio, batch_size,
                str(layer_sizes_val), activation_function, use_weighted_samples_value,
                scale_target_to_unit_interval_value, use_max_inverse_transform_value,
                *predicted_performances
            ])
            # scenario_name, lambda_value, split, seed, use_quadratic_transform, use_max_inverse_transform, scale_target_to_unit_interval

        performance_cols_corras = [
            x + "_performance" for x in scenario.performance_data.columns
        ]
        # print("perf corr", len(performance_cols_corras))
        result_columns_corras = [
            "split", "problem_instance", "lambda", "epsilon", "seed",
            "learning_rate", "es_interval", "es_patience", "es_val_ratio",
            "batch_size", "layer_sizes", "activation_function",
            "use_weighted_samples", "scale_target_to_unit_interval",
            "use_max_inverse_transform"
        ]
        # print("result len", len(result_columns_corras))
        result_columns_corras += performance_cols_corras
        results_corras = pd.DataFrame(data=result_data_corras,
                                      columns=result_columns_corras)
        # results_corras.to_csv(filepath, index_label="id",
        #                         mode="a", header=False)
        # connection = engine.connect()
        # results_corras.to_sql(name=table_name,
        #                       con=connection,
        #                       if_exists="append")
        # connection.close()
        # model.save_loss_history(loss_filepath)
        # model.save_es_val_history(es_val_filepath)

    except Exception as exc:
        print("Something went wrong during computation with parameters " +
              params_string + " message: " + str(exc))