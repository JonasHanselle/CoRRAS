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

# plotting
import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style("darkgrid")

scenario = aslib_ranking_scenario.ASRankingScenario()
scenario.read_scenario("aslib_data-aslib-v4.0/CSP-2010")
print(scenario.performance_data)


# training_portions = np.linspace(start=0, stop=1, num=4)
# lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambda_values = [0.6]
num_splits = 1
result_data = []


for i_split in range(1, num_splits+1):

    test_scenario, train_scenario = scenario.get_split(2)

    print("testfeatures", test_scenario.feature_data)

    for lambda_value in product(lambda_values):

        train_features_np, train_performances_np, train_rankings_np = util.construct_numpy_representation(
            train_scenario.feature_data, train_scenario.performance_data)

        # scale features
        imputer = SimpleImputer()
        scaler = StandardScaler()
        train_features_np = imputer.fit_transform(train_features_np)
        train_features_np = scaler.fit_transform(train_features_np)

        print("features", train_features_np)
        print("rankings", train_rankings_np)
        print("performances", train_performances_np)
        print("orderings", train_rankings_np)
        train_rankings_np = util.ordering_to_ranking_matrix(train_rankings_np)
        print("rankings", train_rankings_np)
        print("\n\n")

        model = log_linear.LogLinearModel()

        model.fit_np(train_rankings_np, train_features_np,
                     train_performances_np, lambda_value=lambda_value, regression_loss="Squared", maxiter=15)

        print("model weights", model.weights)

        for index, row in test_scenario.feature_data.iterrows():
            print("test features", row)
            imputed_row = imputer.transform([row.values])
            scaled_row = scaler.transform(imputed_row).flatten()
            predicted_ranking = model.predict_ranking(scaled_row)
            predicted_performances = model.predict_performances(scaled_row)
            result_data.append([i_split, index, lambda_value, predicted_ranking, predicted_performances])

results = pd.DataFrame(data=result_data, columns=[
                       "split", "problem_instance", "lambda", "predicted_ranking", "predicted_performances"])
results.to_csv(""+scenario.scenario+".csv")
