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
import sqlalchemy

# Corras
import Corras.Model.neural_net_hinge as nn_hinge
from Corras.Scenario import aslib_ranking_scenario
from Corras.Util import ranking_util as util

# Database
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import exists, select, and_, or_
import urllib

result_path = "./results-nnh-new/"
loss_path = "./losses-nnh-new/"

print(sys.argv)

total_shards = int(sys.argv[1])
shard_number = int(sys.argv[2])

# DB data
db_url = sys.argv[3]
db_user = sys.argv[4]
db_pw = urllib.parse.quote_plus(sys.argv[5])
db_db = sys.argv[6]

scenarios = [
    "CPMP-2015", "MIP-2016", "CSP-2010", "SAT11-HAND", "SAT11-INDU",
    "SAT11-RAND"
]

scenarios = ["MIP-2016"]

epsilon_value = 1.0
max_pairs_per_instance = 5
maxiter = 5

learning_rate = 0.001
batch_size = 128
es_patience = 8
es_interval = 8
es_val_ratio = 0.3
layer_sizes_val = [32]
activation_function = "sigmoid"
use_max_inverse_transform_value = "max_cutoff"
scale_target_to_unit_interval_value = True
use_weighted_samples_value = False

lambda_value = 0.5

splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
splits = [1]

seeds = [1, 2, 3, 4, 5]
seeds = [1]

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

engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" + db_url +
                           "/" + db_db,
                           echo=False,
                           pool_recycle=300,
                           pool_size=1)

for scenario_name, split, seed in shard:

    # table_name = "neural-net-squared-hinge-" + scenario_name + "-seeded"
    table_name = "hyper-test-" + scenario_name

    connection = engine.connect()
    insp = sqlalchemy.inspect(engine)
    if not insp.has_table(table_name):
        pass
    else:
        meta = MetaData(engine)
        experiments = Table(table_name,
                            meta,
                            autoload=True,
                            autoload_with=engine)

        slct = experiments.select(
            and_(
                experiments.columns["split"] == split,
                experiments.columns["lambda"] == lambda_value,
                experiments.columns["epsilon"] == epsilon_value,
                experiments.columns["seed"] == seed,
                experiments.columns["learning_rate"] == learning_rate,
                experiments.columns["es_interval"] == es_interval,
                experiments.columns["es_patience"] == es_patience,
                experiments.columns["es_val_ratio"] == es_val_ratio,
                experiments.columns["batch_size"] == batch_size,
                experiments.columns["layer_sizes"] == str(layer_sizes_val),
                experiments.columns["activation_function"] ==
                activation_function, experiments.
                columns["use_weighted_samples"] == use_weighted_samples_value,
                experiments.columns["scale_target_to_unit_interval"] ==
                scale_target_to_unit_interval_value,
                experiments.columns["use_max_inverse_transform"] ==
                use_max_inverse_transform_value)).limit(1)
        rs = connection.execute(slct)
        result = rs.first()
        if result == None:
            print("not in db")
            pass
        else:
            print("already in db")
            rs.close()
            connection.close()
            continue
        rs.close()
        connection.close()

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

        print("perf", perf)

        train_performances = pd.DataFrame(data=perf,
                                          index=train_performances.index,
                                          columns=train_performances.columns)
        print(order)
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
        print("sample weights", sample_weights)

        rank = rank.astype("int32")

        model = nn_hinge.NeuralNetworkSquaredHinge()
        model.fit(len(scenario.algorithms),
                  rank,
                  inst,
                  perf,
                  lambda_value=lambda_value,
                  epsilon_value=epsilon_value,
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
        print("perf corr", len(performance_cols_corras))
        result_columns_corras = [
            "split", "problem_instance", "lambda", "epsilon", "seed",
            "learning_rate", "es_interval", "es_patience", "es_val_ratio",
            "batch_size", "layer_sizes", "activation_function",
            "use_weighted_samples", "scale_target_to_unit_interval",
            "use_max_inverse_transform"
        ]
        print("result len", len(result_columns_corras))
        result_columns_corras += performance_cols_corras
        results_corras = pd.DataFrame(data=result_data_corras,
                                      columns=result_columns_corras)
        # results_corras.to_csv(filepath, index_label="id",
        #                         mode="a", header=False)
        connection = engine.connect()
        results_corras.to_sql(name=table_name,
                              con=connection,
                              if_exists="append")
        connection.close()
        # model.save_loss_history(loss_filepath)
        # model.save_es_val_history(es_val_filepath)

    except Exception as exc:
        print("Something went wrong during computation with parameters " +
              params_string + " message: " + str(exc))
