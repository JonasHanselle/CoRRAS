import numpy as np
import pandas as pd

from itertools import product

# evaluation stuff
from sklearn.model_selection import KFold
from scipy.stats import kendalltau

# Corras
from Corras.Model import log_linear
from Corras.Scenario import aslib_ranking_scenario

# plotting
import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style("darkgrid")

scenario = aslib_ranking_scenario.ASRankingScenario()
scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")

scenario.compute_rankings(break_up_ties=True)

# print("lens",len(scenario.performance_rankings), len(scenario.performance_data), len(scenario.feature_data), len(scenario.performance_data_all))

# scenario.remove_duplicates()

print(scenario.performance_data)


kf = KFold(n_splits=5, shuffle=True, random_state=10)
training_portions = np.linspace(start=0, stop=1, num=4)
lambda_values = [0.0, 0.5, 1.0]
taus = []
result_data = []

for split_num, split in enumerate(kf.split(scenario.performance_rankings)):

    for portion, lambda_value in product(training_portions, lambda_values):
        train_rankings = scenario.performance_rankings.iloc[split[0]].astype(
            'int32')
        train_features = scenario.feature_data.loc[train_rankings.index.drop_duplicates()].astype(
            'float64')
        train_performances = scenario.performance_data.loc[train_rankings.index.drop_duplicates()].astype(
            'float64')

        # print("lengths", len(train_features), len(train_performances))
        # print("features", train_features.loc[train_features.duplicated()])
        # print("performances", train_performances.loc[train_performances.duplicated()])
        test_rankings = scenario.performance_rankings.iloc[split[1]].astype(
            'int32')
        test_features = scenario.feature_data.loc[test_rankings.index.drop_duplicates()].astype(
            'float64')
        test_performances = scenario.performance_data.loc[test_rankings.index.drop_duplicates()].astype(
            'float64')


        train_features = train_features[:int(portion*len(train_features))]
        train_rankings = train_rankings[:int(portion*len(train_rankings))]
        train_performances = train_performances[:int(
            portion*len(train_performances))]

        if(len(train_rankings) == 0):
            continue

        model = log_linear.LogLinearModel()
        model.fit(train_rankings,None,train_features,train_performances,lambda_value=0,regression_loss="Squared")

        current_taus = []

        for index, row in test_features.iterrows():
            predicted_ranking = model.predict_ranking(row.values)
            true_ranking = test_rankings.loc[index].values
            print(true_ranking)
            tau = kendalltau(predicted_ranking, true_ranking).correlation
            print(tau)
            result_data.append([split_num, portion, tau, lambda_value])

        print("current avg", np.average(current_taus))

results = pd.DataFrame(data=result_data, columns=[
                       "split", "train_portion", "tau", "lambda"])
sb.lineplot(x="train_portion", y="tau", hue="lambda", data=results)
plt.show()
