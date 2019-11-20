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

scenario = aslib_ranking_scenario.ASRankingScenario()
scenario.read_scenario("aslib_data-aslib-v4.0/"+sys.argv[1])

maxiter = 100
seed = 15
lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_splits = 10
result_data_corras = []
result_data_rf = []
baselines = None

for i_split in range(1, num_splits+1):

    test_scenario, train_scenario = scenario.get_split(i_split)

    train_features_np, train_performances_np, train_rankings_np = util.construct_numpy_representation(
        train_scenario.feature_data, train_scenario.performance_data)
    
    # preprocessing
    imputer = SimpleImputer()
    scaler = StandardScaler()
    train_features_np = imputer.fit_transform(train_features_np)
    train_features_np = scaler.fit_transform(train_features_np)

    # Create one random forest regressor per label
    baselines = []
    for label in range(0,len(train_scenario.performance_data.columns)):
        baselines.append(RandomForestRegressor(random_state = seed))
    for label in range(0,len(train_scenario.performance_data.columns)):
        baselines[label].fit(train_features_np, train_performances_np[:,label])

    for index, row in test_scenario.feature_data.iterrows():
        imputed_row = imputer.transform([row.values])
        scaled_row = scaler.transform(imputed_row)
        predicted_performances = [-1] * len(train_scenario.performance_data.columns) 
        for label in range(0,len(train_scenario.performance_data.columns)):
            predicted_performances[label] = baselines[label].predict(scaled_row)[0]
        result_data_rf.append([i_split, index, *predicted_performances])

    for lambda_value in lambda_values:


        train_rankings_list = util.ordering_to_ranking_list(train_rankings_np)

        model = log_linear.LogLinearModel()

        model.fit_list(len(scenario.algorithms),train_rankings_list, train_features_np,
                     train_performances_np, lambda_value=lambda_value, regression_loss="Squared", maxiter=maxiter)


        for index, row in test_scenario.feature_data.iterrows():
            imputed_row = imputer.transform([row.values])
            scaled_row = scaler.transform(imputed_row).flatten()
            predicted_ranking = model.predict_ranking(scaled_row)
            predicted_performances = model.predict_performances(scaled_row)
            result_data_corras.append([i_split, index, lambda_value, predicted_ranking, *predicted_performances])

performance_cols_corras = [x + "_performance" for x in scenario.performance_data.columns]

result_columns_corras = ["split", "problem_instance", "lambda", "predicted_ranking"]
result_columns_corras += performance_cols_corras
results_corras = pd.DataFrame(data=result_data_corras, columns=result_columns_corras)
results_corras.to_csv("corras-"+scenario.scenario+".csv", index_label="id")
performance_cols = [x + "_performance" for x in scenario.performance_data.columns]

result_columns_rf = ["split", "problem_instance"]
result_columns_rf += performance_cols
results_rf = pd.DataFrame(data=result_data_rf, columns=result_columns_rf)
results_rf.to_csv("rf-"+scenario.scenario+".csv", index_label="id")