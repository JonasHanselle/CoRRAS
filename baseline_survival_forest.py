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

# scenarios = [
#     "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "MAXSAT-WPMS-2016",
#     "MAXSAT-PMS-2016", "MIP-2016", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND",
#     "SAT12-ALL", "TTP-2016"
# ]
scenarios = ["MIP-2016"]

for scenario_name in scenarios:
    try:
        result_data_rsf = []
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/" + scenario_name)

        scenario.create_cv_splits(n_folds=num_splits)

        table_name = "baseline_random_survival_forest-" + scenario.scenario

        for i_split in range(1, num_splits + 1):

            test_scenario, train_scenario = scenario.get_split(i_split)

            # Prepare data for survival analysis
            train_performances = train_scenario.performance_data
            train_features = train_scenario.feature_data

            test_performances = test_scenario.performance_data
            test_features = test_scenario.feature_data

            melted_train_performances = pd.melt(train_performances.reset_index(
            ), id_vars="instance_id", value_name="performance")
            joined_train_data = train_features.join(
                melted_train_performances.set_index("instance_id"))
            joined_train_data["algorithm"] = joined_train_data["algorithm"].astype(
                "category")
            encoder = OneHotEncoder()
            train_data = encoder.fit_transform(joined_train_data)
            X_train = train_data.to_numpy()[:,:-1]
            y_train = train_data.to_numpy()[:,-1]

            melted_test_performances = pd.melt(test_performances.reset_index(
            ), id_vars="instance_id", value_name="performance")
            joined_test_data = test_features.join(
                melted_test_performances.set_index("instance_id"))
            joined_test_data["algorithm"] = joined_test_data["algorithm"].astype(
                "category")
            test_data = encoder.transform(joined_test_data)
            X_test = test_data.to_numpy()[:,:-1]
            y_test = test_data.to_numpy()[:,-1]
            one_hot_columns = [i for (i,col) in enumerate(train_data.columns) if "algorithm=" in col]
            for i in one_hot_columns:
                print(train_data.columns[i])
            # print(X_train, y_train)
            # print(X_test, y_test)

            # preprocessing
            # imputer = SimpleImputer()
            # scaler = StandardScaler()
            # X_train = imputer.fit_transform(X_train)
            # X_train = scaler.fit_transform(X_train)

            # model = RandomSurvivalForest(n_estimators=1000,
            #                              min_samples_split=10,
            #                              min_samples_leaf=15,
            #                              max_features="sqrt",
            #                              n_jobs=1,
            #                              random_state=seed)

            # mask = y_train != scenario.algorithm_cutoff_time * 10

            # timeouted_runs = ~mask

            # # the time at which the observation ends is actually the cutoff, not the par10
            # y_train[timeouted_runs] = scenario.algorithm_cutoff_time

            # structured_y_train = np.rec.fromarrays([mask, y_train],
            #                                        names="terminated,runtime")

            # print(X_train, structured_y_train)
            # # model.fit(X_train, structured_y_train)

            # X_test = imputer.transform(X_test)
            # X_test = scaler.transform(X_test)

            # # predictions = model.predict(X_test)
            # # print("predictions", predictions)

            # for index, row in test_scenario.feature_data.iterrows():
            #     imputed_row = imputer.transform([row.values])
            #     scaled_row = scaler.transform(imputed_row)
            #     predicted_performances = [-1] * len(
            #         train_scenario.performance_data.columns)
            #     for algorithm in scenario.algorithms:
            #         new_row = row.append(a)
            #         predicted_performances[label] = predict(
            #             scaled_row)[0]
            #     result_data_rsf.append(
            #         [i_split, index, *predicted_performances])

        # performance_cols = [
        #     x + "_performance" for x in scenario.performance_data.columns
        # ]
        # result_columns_rsf = ["split", "problem_instance"]
        # result_columns_rsf += performance_cols
        # results_rsf = pd.DataFrame(data=result_data_rsf,
        #                            columns=result_columns_rsf)
        # results_rsf.to_csv(result_path + "rf-" + scenario.scenario + ".csv",
        #                    index_label="id")
        # engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" +
        #                            db_url + "/" + db_db,
        #                            echo=False)
        # connection = engine.connect()
        # results_rsf.to_sql(name=table_name, con=connection)
        # connection.close()
    except Exception as exc:
        print("Something went wrong during computation. Message: " + str(exc))
