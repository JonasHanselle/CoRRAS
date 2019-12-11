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
from Corras.Model import linear_hinge as lh
from Corras.Scenario import aslib_ranking_scenario
from Corras.Util import ranking_util as util

# Baseline
from sklearn.ensemble import RandomForestRegressor

result_path = "./results/"

scenario = aslib_ranking_scenario.ASRankingScenario()
scenario.read_scenario("aslib_data-aslib-v4.0/"+sys.argv[1])

lambda_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
epsilon_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
max_pairs_per_instance = 5
maxiter = 5
seed = 15
num_splits = 10
result_data_corras = []
result_data_rf = []
baselines = None

scenario.create_cv_splits(n_folds=num_splits)
for i_split in range(1, num_splits+1):

    test_scenario, train_scenario = scenario.get_split(i_split)
    
    train_performances = train_scenario.performance_data
    train_features = train_scenario.feature_data

    print(train_features)

    # preprocessing
    imputer = SimpleImputer()
    scaler = StandardScaler()
    train_features[train_features.columns] = imputer.fit_transform(train_features[train_features.columns])
    train_features[train_features.columns] = scaler.fit_transform(train_features[train_features.columns])

    print(train_features)

    inst,perf,rank = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features(train_features,train_performances,max_pairs_per_instance=max_pairs_per_instance,seed=15)
    
    # scale performances to unit interval
    perf = perf/np.max(perf) 
    # train models
    for lambda_value, epsilon_value in product(lambda_values,epsilon_values):
        model = model1 = lh.LinearHingeModel()

        model.fit_np(len(scenario.algorithms),rank, inst,
                     perf, lambda_value=lambda_value, epsilon_value=epsilon_value, regression_loss="Squared", maxiter=maxiter, print_output=False)

        loss_file = "./loss-hists/loss-hist-hinge-" + scenario.scenario + "-" + str(lambda_value) + "-" + str(epsilon_value) + "-" + str(i_split) + "-" + str(seed) + ".csv"
        model.save_loss_history(loss_file)

        for index, row in test_scenario.feature_data.iterrows():
            imputed_row = imputer.transform([row.values])
            scaled_row = scaler.transform(imputed_row).flatten()
            predicted_ranking = model.predict_ranking(scaled_row)
            predicted_performances = model.predict_performances(scaled_row)
            result_data_corras.append([i_split, index, lambda_value, epsilon_value, predicted_ranking, *predicted_performances])

performance_cols_corras = [x + "_performance" for x in scenario.performance_data.columns]

result_columns_corras = ["split", "problem_instance", "lambda", "epsilon", "predicted_ranking"]
result_columns_corras += performance_cols_corras
results_corras = pd.DataFrame(data=result_data_corras, columns=result_columns_corras)
filename = result_path+"corras-linhinge-"+scenario.scenario
loss_file = filename + "-losses.csv"
filename += ".csv"
results_corras.to_csv(filename, index_label="id")
model.save_loss_history(loss_file)
