import sys

import autograd.numpy as np
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
from sklearn.ensemble import RandomForestRegressor

result_path = "./results/"

scenario = aslib_ranking_scenario.ASRankingScenario()
scenario.read_scenario("aslib_data-aslib-v4.0/"+sys.argv[1])


# max_rankings_per_instance = 5
seed = 15
num_splits = 10
result_data_corras = []
result_data_rf = []
baselines = None

scenario.create_cv_splits(n_folds=num_splits)
for i_split in range(1, num_splits+1):

    test_scenario, train_scenario = scenario.get_split(i_split)
    train_features_np, train_performances_np = util.construct_numpy_representation_only_performances(
        train_scenario.feature_data, train_scenario.performance_data)
    print(train_scenario.performance_data)
    print(train_performances_np)
    print(train_scenario.feature_data)
    print(train_features_np)
    # train_features_np = scenario.feature_data.to_numpy
    # train_performances_np = scenario.perform

    print(train_scenario)
    # preprocessing
    imputer = SimpleImputer()
    scaler = StandardScaler()
    train_features_np = imputer.fit_transform(train_features_np)
    train_features_np = scaler.fit_transform(train_features_np)

    # Create one random forest regressor per label
    baselines = []
    for label in range(0,len(train_scenario.performance_data.columns)):
        baselines.append(RandomForestRegressor(random_state = seed, n_estimators=100))
    for label in range(0,len(train_scenario.performance_data.columns)):
        baselines[label].fit(train_features_np, train_performances_np[:,label])

    for index, row in test_scenario.feature_data.iterrows():
        imputed_row = imputer.transform([row.values])
        scaled_row = scaler.transform(imputed_row)
        predicted_performances = [-1] * len(train_scenario.performance_data.columns) 
        for label in range(0,len(train_scenario.performance_data.columns)):
            predicted_performances[label] = baselines[label].predict(scaled_row)[0]
        result_data_rf.append([i_split, index, *predicted_performances])

performance_cols = [x + "_performance" for x in scenario.performance_data.columns]

result_columns_rf = ["split", "problem_instance"]
result_columns_rf += performance_cols
results_rf = pd.DataFrame(data=result_data_rf, columns=result_columns_rf)
results_rf.to_csv(result_path+"rf-"+scenario.scenario+".csv", index_label="id")