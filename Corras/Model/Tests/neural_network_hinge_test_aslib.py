import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.neural_net_hinge as nnh
import Corras.Model.neural_net as nn
import Corras.Model.linear_hinge as lh
import Corras.Util.ranking_util as util
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralNetworkHingeTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(NeuralNetworkHingeTest, self).__init__(*args, **kwargs)
        self.scenario = scen.ASRankingScenario()
        self.scenario.read_scenario("aslib_data-aslib-v4.0/MIP-2016")
        self.scenario.compute_rankings()
        # preprocessing of data
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer()
        self.test_scen, self.train_scen = self.scenario.get_split(indx=5)
        self.test_scen.remove_duplicates()
        self.train_scen.remove_duplicates()

        self.train_inst = self.train_scen.feature_data
        self.test_inst = self.test_scen.feature_data
        self.train_performances = self.train_scen.performance_data
        self.test_performances = self.test_scen.performance_data
        self.train_ranking = self.train_scen.performance_rankings
        self.test_ranking = self.test_scen.performance_rankings

    def test_fit(self):
        model1 = nnh.NeuralNetwork()
        features, performances, rankings, sample_weights = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(
            self.train_inst, self.train_performances, order="asc", skip_value=None)
        rankings = rankings.astype("int32")
        performances = performances / np.max(performances)
        print("raw features", features)
        features = self.imputer.fit_transform(features)
        features = self.scaler.fit_transform(features)
        print("scaled features", features)
        print(performances)
        print(rankings)
        maximum = np.max(performances)
        performances_max_inv = maximum - performances
        max_inv = np.max(performances_max_inv)
        performances_max_inv = performances_max_inv / max_inv
        max_entry = performances.max()
        scaled_perf = performances / performances.max()
        lambda_value = 0.5
        model1.fit(5,
                   rankings,
                   features,
                   performances,
                   sample_weights=None,
                   lambda_value=0.5,
                #    epsilon_value=1.0,
                   regression_loss="Squared",
                   num_epochs=150,
                   learning_rate=0.001,
                   batch_size=32,
                   early_stop_interval=5,
                   patience=16,
                   hidden_layer_sizes=[32],
                   activation_function="sigmoid")
        for i, (index, row) in enumerate(self.train_performances.iterrows()):
            instance_values = self.train_inst.loc[index].values
            imputed_row = self.imputer.transform([instance_values])
            scaled_row = self.scaler.transform(imputed_row).flatten()
            print("predicted", model1.predict_performances(scaled_row))
            print("truth performance", row.values)
            print("\n")
            print("Predicted Ranking", model1.predict_ranking(scaled_row))
            print("True Ranking", np.argsort(np.argsort(row.values))+1)
            print("\n")
        # sns.set_style("darkgrid")
        # df = model1.get_loss_history_frame()
        # print(df)
        # df = df.rename(columns={"NLL":"PL-NLL"})
        # df["$\lambda$ PL-NLL"] = lambda_value * df["PL-NLL"]
        # df["$(1 - \lambda)$ MSE"] = (1 - lambda_value) * df["MSE"]
        # df["TOTAL_LOSS"] = df["$\lambda$ PL-NLL"] + df["$(1 - \lambda)$ MSE"]
        # df = df.melt(id_vars=["iteration"])
        # plt.clf()
        # # plt.tight_layout()
        # # plt.annotate(text,(0,0), (0,-40), xycoords="axes fraction", textcoords="offset points", va="top")
        # print(df.head())
        # lp = sns.lineplot(x="iteration", y="value", hue="variable", data=df)
        # plt.title(self.scenario.scenario)
        # plt.show()


if __name__ == "__main__":
    unittest.main()
