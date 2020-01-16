import sys

import numpy as np
import pandas as pd

from itertools import product

# evaluation stuff
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import kendalltau

# Corras
from Corras.Model import log_linear
from Corras.Scenario import aslib_ranking_scenario
from Corras.Util import ranking_util as util

# Baseline
from sklearn.linear_model import LinearRegression

# Database
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import exists, select, and_, or_
import urllib

# sksurv
from sksurv.ensemble import RandomSurvivalForest

result_path = "./results/"

# DB data
# db_url = sys.argv[1]
# db_user = sys.argv[2]
# db_pw = urllib.parse.quote_plus(sys.argv[3])
# db_db = sys.argv[4]

# max_rankings_per_instance = 5
seed = 1
num_splits = 10
result_data_corras = []
baselines = None

scenarios = ["CPMP-2015"]

for scenario_name in scenarios:
    try:
        result_data_rf = []
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/" + scenario_name)

        # scenario.create_cv_splits(n_folds=num_splits)

        table_name = "baseline_linear_regression-" + scenario.scenario

        for i_split in range(1, num_splits + 1):

            test_scenario, train_scenario = scenario.get_split(i_split)
            train_features_np, train_performances_np = util.construct_numpy_representation_only_performances(
                train_scenario.feature_data, train_scenario.performance_data)
            print(train_scenario.performance_data)
            print(train_performances_np)
            print(train_scenario.feature_data)
            print(train_features_np)

            print(train_scenario)
            # preprocessing
            imputer = SimpleImputer()
            scaler = StandardScaler()
            train_features_np = imputer.fit_transform(train_features_np)
            train_features_np = scaler.fit_transform(train_features_np)

            # Create one linear regressor per label
            baselines = []
            for label in range(0,
                               len(train_scenario.performance_data.columns)):
                baselines.append(RandomSurvivalForest(n_jobs=-1, random_state=1))
            for label in range(0,
                               len(train_scenario.performance_data.columns)):
                y_train = train_performances_np[:, label]
                mask = y_train < scenario.algorithm_cutoff_time

                structured_y_train = np.rec.fromarrays(
                    [mask, y_train], names="terminated,runtime")
                print(structured_y_train)
                baselines[label].fit(train_features_np,structured_y_train)

            for index, row in test_scenario.feature_data.iterrows():
                imputed_row = imputer.transform([row.values])
                scaled_row = scaler.transform(imputed_row)
                predicted_performances = [-1] * len(
                    train_scenario.performance_data.columns)
                for label in range(
                        0, len(train_scenario.performance_data.columns)):
                    predicted_performances[label] = baselines[label].predict(
                        scaled_row)[0]
                print("predicted performances", predicted_performances)
                result_data_rf.append(
                    [i_split, index, *predicted_performances])

        performance_cols = [
            x + "_performance" for x in scenario.performance_data.columns
        ]

        result_columns_rf = ["split", "problem_instance"]
        result_columns_rf += performance_cols
        results_rf = pd.DataFrame(data=result_data_rf,
                                  columns=result_columns_rf)

        results_rf.to_csv(result_path + "rsf-multi-" + scenario.scenario + ".csv",
                          index_label="id")
        # engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" +
        #                            db_url + "/" + db_db,
        #                            echo=False)
        # connection = engine.connect()
        # results_rf.to_sql(name=table_name, con=connection)
        # connection.close()
    except Exception as exc:
        print("Something went wrong during computation. Message: " + str(exc))
