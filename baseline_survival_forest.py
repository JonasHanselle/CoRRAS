import sys

import numpy as np
import pandas as pd

from itertools import product

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder

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

result_path = "./results/"

# DB data
# db_url = sys.argv[1]
# db_user = sys.argv[2]
# db_pw = urllib.parse.quote_plus(sys.argv[3])
# db_db = sys.argv[4]

# max_rankings_per_instance = 5
seed = 15
num_splits = 10
result_data_corras = []
baselines = None

scenarios = ["MIP-2016"]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for scenario_name in scenarios:
    try:
        result_data_rsf = []
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/" + scenario_name)

        # scenario.create_cv_splits(n_folds=num_splits)

        table_name = "baseline_random_survival_forest-temp-" + scenario.scenario

        for i_split in splits:

            test_scenario, train_scenario = scenario.get_split(i_split)

            # Prepare data for survival analysis
            train_performances = train_scenario.performance_data
            train_features = train_scenario.feature_data

            test_performances = test_scenario.performance_data
            test_features = test_scenario.feature_data

            print(train_performances)
            melted_train_performances = pd.melt(train_performances.reset_index(
            ), id_vars="instance_id", value_name="performance")
            joined_train_data = train_features.join(
                melted_train_performances.set_index("instance_id"))
            joined_train_data["algorithm"] = joined_train_data["algorithm"].astype(
                "category")
            encoder = OneHotEncoder()
            train_data = encoder.fit_transform(joined_train_data)

            melted_test_performances = pd.melt(test_performances.reset_index(
            ), id_vars="instance_id", value_name="performance")
            joined_test_data = test_features.join(
                melted_test_performances.set_index("instance_id"))
            joined_test_data["algorithm"] = joined_test_data["algorithm"].astype(
                "category")
            test_data = encoder.transform(joined_test_data)

            # preprocessing
            imputer = SimpleImputer()
            scaler = StandardScaler()

            scalable_columns = [col for (i, col) in enumerate(
                train_data.columns) if "algorithm=" not in col and "performance" not in col]

            train_data[scalable_columns] = imputer.fit_transform(
                train_data[scalable_columns])
            train_data[scalable_columns] = scaler.fit_transform(
                train_data[scalable_columns])
            print("train data", train_data)
            X_train = train_data.iloc[:, :-1]
            print("X_train", X_train)
            y_train = train_data.iloc[:, -1]
            print("y_train", y_train)
            # X_test = test_data.iloc[:, :-1]
            # y_test = test_data.iloc[:, -1]

            model = RandomSurvivalForest(n_estimators=1000,
                                        max_depth=5,
                                         min_samples_split=10,
                                         min_samples_leaf=15,
                                         max_features="sqrt",
                                         n_jobs=1,
                                         random_state=seed)

            mask = y_train != scenario.algorithm_cutoff_time * 10

            timeouted_runs = ~mask

            # the time at which the observation ends is actually the cutoff, not the par10
            y_train[timeouted_runs] = scenario.algorithm_cutoff_time * 10

            structured_y_train = np.rec.fromarrays([mask, y_train],
                                                   names="terminated,runtime")

            print(structured_y_train)

            print("Starting to fit model")
            model.fit(X_train, structured_y_train)

            # X_test = imputer.transform(X_test)
            # X_test = scaler.transform(X_test)

            # predictions = model.predict(X_test)
            # print("predictions", predictions)

            for index, row in test_scenario.feature_data.iterrows():
                predicted_performances = []
                # predicted_performances = [-1] * len(
                #     test_scenario.performance_data.columns)
                for algorithm in scenario.algorithms:
                    print(algorithm)
                    temp_features = row.append(
                        pd.Series(data=[algorithm], index=["algorithm"]))
                    features_df = pd.DataFrame([temp_features])
                    features_df["algorithm"] = features_df["algorithm"].astype(
                        "category")
                    encoded_features = encoder.transform(features_df)
                    encoded_features[scalable_columns] = imputer.transform(
                        encoded_features[scalable_columns])
                    encoded_features[scalable_columns] = scaler.transform(
                        encoded_features[scalable_columns])
                    encoded_features = encoded_features.iloc[:,:-1]
                    print("encoded features", encoded_features)
                    encoded_features_np = encoded_features.to_numpy()
                    print("encoded features np", encoded_features_np)
                    predicted_performance = [0]
                    predicted_performance = model.predict(encoded_features_np)
                    predicted_performances.append(predicted_performance[0])
                result_data_rsf.append(
                    [i_split, index, *predicted_performances])

        performance_cols = [
            x + "_performance" for x in scenario.performance_data.columns
        ]
        result_columns_rsf = ["split", "problem_instance"]
        result_columns_rsf += performance_cols
        results_rsf = pd.DataFrame(data=result_data_rsf,
                                   columns=result_columns_rsf)
        results_rsf.to_csv(result_path + "rsf-" + scenario.scenario + ".csv",
                           index_label="id")
        engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" +
                                   db_url + "/" + db_db,
                                   echo=False)
        connection = engine.connect()
        results_rsf.to_sql(name=table_name, con=connection)
        connection.close()
    except Exception as exc:
        print("Something went wrong during computation. Message: " + str(exc))
