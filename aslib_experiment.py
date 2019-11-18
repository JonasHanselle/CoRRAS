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
scenario.read_scenario("aslib_data-aslib-v4.0/CSP-2010")
print(scenario.performance_data)

maxiter = 5
# lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambda_values = [0.0, 0.5, 1.0]
num_splits = 1
result_data = []
baselines = None

for i_split in range(1, num_splits+1):

    test_scenario, train_scenario = scenario.get_split(i_split)

    imputer = SimpleImputer()
    scaler = StandardScaler()
    train_features_np = imputer.fit_transform(train_features_np)
    train_features_np = scaler.fit_transform(train_features_np)
    # Create one random forest regressor per label
    baselines = [RandomForestRegressor()] * len(train_scenario.performance_data.columns)

    for lambda_value in lambda_values:

        train_features_np, train_performances_np, train_rankings_np = util.construct_numpy_representation(
            train_scenario.feature_data, train_scenario.performance_data)

        train_rankings_np = util.ordering_to_ranking_matrix(train_rankings_np)

        model = log_linear.LogLinearModel()

        model.fit_np(train_rankings_np, train_features_np,
                     train_performances_np, lambda_value=lambda_value, regression_loss="Squared", maxiter=maxiter)

        print("model weights", model.weights)

        for index, row in test_scenario.feature_data.iterrows():
            print("test features", row)
            imputed_row = imputer.transform([row.values])
            scaled_row = scaler.transform(imputed_row).flatten()
            predicted_ranking = model.predict_ranking(scaled_row)
            predicted_performances = model.predict_performances(scaled_row)
            result_data.append([i_split, index, lambda_value, predicted_ranking, *predicted_performances])

performance_cols = [x + "_performance" for x in scenario.performance_data.columns]

result_columns = ["split", "problem_instance", "lambda", "predicted_ranking"]
result_columns += performance_cols
print("results_cols", result_columns)
results = pd.DataFrame(data=result_data, columns=result_columns)
results.to_csv(""+scenario.scenario+".csv")
