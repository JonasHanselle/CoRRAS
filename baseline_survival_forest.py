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
db_url = sys.argv[1]
db_user = sys.argv[2]
db_pw = urllib.parse.quote_plus(sys.argv[3])
db_db = sys.argv[4]

# max_rankings_per_instance = 5
seed = 1
num_splits = 10
result_data_corras = []
baselines = None

scenarios = ["SAT11-HAND",
             "SAT11-INDU", "SAT11-RAND", "MIP-2016",
             "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "MAXSAT-WPMS-2016",
             "SAT12-ALL", "MAXSAT-PMS-2016",  "SAT12-ALL", "TTP-2016"]
splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for scenario_name in scenarios:
    # try:
    result_data_rsf = []
    scenario = aslib_ranking_scenario.ASRankingScenario()
    scenario.read_scenario("aslib_data-aslib-v4.0/" + scenario_name)

    # scenario.create_cv_splits(n_folds=num_splits)

    table_name = "baseline_random_survival_forest-attempt-" + scenario.scenario

    for i_split in splits:

        test_scenario, train_scenario = scenario.get_split(i_split)

        # Prepare data for survival analysis
        train_performances = train_scenario.performance_data
        train_features = train_scenario.feature_data

        test_performances = test_scenario.performance_data
        test_features = test_scenario.feature_data
        dataset = []
        indices = []
        performances = []
        for inst_index, row in train_performances.iterrows():
            for alg_index, algorithm in enumerate(scenario.performance_data.columns):
                cur_features = scenario.feature_data.loc[inst_index]
                alg_enc = len(scenario.performance_data.columns) * [0]
                alg_enc[alg_index] = 1
                alg_columns = ["algorithm=" +
                               alg for alg in scenario.performance_data.columns]
                alg_one_hot = pd.Series(alg_enc, index=alg_columns)
                cur_performance = row.loc[algorithm]
                new_row = cur_features.append(alg_one_hot).append(
                    pd.Series(data=[cur_performance],
                              index=["performance"]))
                dataset.append(new_row)
                indices.append(inst_index)
                performances.append(cur_performance)

        train_data = pd.DataFrame(dataset, index=indices)
        # print(scenario.performance_data.columns)
        # preprocessing
        imputer = SimpleImputer()
        scaler = StandardScaler()

        scalable_columns = [
            col for (i, col) in enumerate(train_data.columns)
            if "algorithm=" not in col and "performance" not in col
        ]

        train_data[scalable_columns] = imputer.fit_transform(
            train_data[scalable_columns])
        train_data[scalable_columns] = scaler.fit_transform(
            train_data[scalable_columns])
        X_train = train_data.to_numpy()[:, :-1]
        # print(scenario.performance_data)
        # y_train = train_data.to_numpy()[:, -1]
        y_train = np.asarray(performances)
        # print(y_train)
        # X_test = test_data.iloc[:, :-1]
        # y_test = test_data.iloc[:, -1]

        model = RandomSurvivalForest(n_estimators=1000,
                                     min_samples_split=10,
                                     min_samples_leaf=15,
                                     max_features="sqrt",
                                     n_jobs=-1,
                                     random_state=seed)

        mask = y_train != scenario.algorithm_cutoff_time * 10

        # timeouted_runs = ~mask

        # the time at which the observation ends is actually the cutoff, not the par10
        # y_train[timeouted_runs] = scenario.algorithm_cutoff_time * 10

        structured_y_train = np.rec.fromarrays([mask, y_train],
                                               names="terminated,runtime")

        # print(train_scenario.performance_data.head())
        # print(train_scenario.performance_data.tail())
        # print(X_train.to_numpy()[0])
        # print(X_train.to_numpy()[1])
        # print(X_train.to_numpy()[2])
        # print(X_train.to_numpy()[3])
        # print(X_train.to_numpy()[4])
        # print(X_train.to_numpy()[5])
        # print(y_train.head())
        # print(y_train.tail())
        # print(structured_y_train)
        model.fit(X_train, structured_y_train)

        for index, row in test_scenario.feature_data.iterrows():
            predicted_performances = []
            # predicted_performances = [-1] * len(
            #     test_scenario.performance_data.columns)
            for alg_index, algorithm in enumerate(scenario.performance_data.columns):
                cur_features = row
                alg_enc = len(scenario.performance_data.columns) * [0]
                alg_enc[alg_index] = 1
                alg_columns = ["algorithm=" +
                               alg for alg in scenario.performance_data.columns]
                alg_one_hot = pd.Series(alg_enc, index=alg_columns)
                new_row = cur_features.append(alg_one_hot)
                new_row = pd.DataFrame([new_row])
                new_row[scalable_columns] = imputer.transform(
                    new_row[scalable_columns])
                new_row[scalable_columns] = scaler.transform(
                    new_row[scalable_columns])
                new_row_np = new_row.to_numpy().astype(np.float64).reshape(
                    1, -1)
                # print(algorithm)
                # print(new_row_np)
                predicted_performance = model.predict(new_row_np)
                predicted_performances.append(predicted_performance[0])
            result_data_rsf.append([i_split, index, *predicted_performances])

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
    results_rsf.to_sql(name=table_name, con=connection, if_exists="append")
    connection.close()
    # except Exception as exc:
    #     print("Exception occured", str(exc))
