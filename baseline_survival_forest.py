import sys

import numpy as np
import pandas as pd

from itertools import product

# evaluation stuff
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import kendalltau

# sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import load_gbsg2

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

scenarios = [
    "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "MAXSAT-WPMS-2016",
    "MAXSAT-PMS-2016", "MIP-2016", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND",
    "SAT12-ALL", "TTP-2016"
]
# scenarios = ["MIP-2016"]

X, y = load_gbsg2()

for scenario_name in scenarios:
    try:
        result_data_rsf = []
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/" + scenario_name)

        scenario.create_cv_splits(n_folds=num_splits)

        table_name = "baseline_random_survival_forest-" + scenario.scenario

        for i_split in range(1, num_splits + 1):

            test_scenario, train_scenario = scenario.get_split(i_split)
            train_features_np, train_performances_np = util.construct_numpy_representation_only_performances(
                train_scenario.feature_data, train_scenario.performance_data)

            mask = train_performances_np != scenario.algorithm_cutoff_time * 10

            timeouted_runs = ~mask
            train_performances_np[
                timeouted_runs] = scenario.algorithm_cutoff_time
            structured_array = np.rec.fromarrays([mask, train_performances_np],
                                                 names="terminated,runtime")


            # preprocessing
            imputer = SimpleImputer()
            scaler = StandardScaler()
            train_features_np = imputer.fit_transform(train_features_np)
            train_features_np = scaler.fit_transform(train_features_np)

            # Create one linear regressor per label

            baselines = []
            for label in range(0,
                               len(train_scenario.performance_data.columns)):
                baselines.append(
                    RandomSurvivalForest(n_estimators=1000,
                                         min_samples_split=10,
                                         min_samples_leaf=15,
                                         max_features="sqrt",
                                         n_jobs=1,
                                         random_state=seed))

            for label in range(0,
                               len(train_scenario.performance_data.columns)):
                baselines[label].fit(train_features_np,
                                     structured_array[:, label])

            for index, row in test_scenario.feature_data.iterrows():
                imputed_row = imputer.transform([row.values])
                scaled_row = scaler.transform(imputed_row)
                predicted_performances = [-1] * len(
                    train_scenario.performance_data.columns)
                for label in range(
                        0, len(train_scenario.performance_data.columns)):
                    predicted_performances[label] = baselines[label].predict(
                        scaled_row)[0]
                result_data_rsf.append(
                    [i_split, index, *predicted_performances])

        performance_cols = [
            x + "_performance" for x in scenario.performance_data.columns
        ]
        result_columns_rsf = ["split", "problem_instance"]
        result_columns_rsf += performance_cols
        results_rsf = pd.DataFrame(data=result_data_rsf,
                                   columns=result_columns_rsf)
        results_rsf.to_csv(result_path + "rf-" + scenario.scenario + ".csv",
                           index_label="id")
        engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" +
                                   db_url + "/" + db_db,
                                   echo=False)
        connection = engine.connect()
        results_rsf.to_sql(name=table_name, con=connection)
        connection.close()
    except Exception as exc:
        print("Something went wrong during computation. Message: " + str(exc))

