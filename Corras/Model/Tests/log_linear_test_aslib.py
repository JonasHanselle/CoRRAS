import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
import Corras.Util.ranking_util as util
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau


class TestLogLinearModelASLib(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLogLinearModelASLib, self).__init__(*args, **kwargs)
        self.scenario = scen.ASRankingScenario()
        self.scenario.read_scenario("aslib_data-aslib-v4.0/CSP-2010")
        self.scenario.compute_rankings()
        # preprocessing of data
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer()
        self.test_scen, self.train_scen = self.scenario.get_split(indx=6)
        self.test_scen.remove_duplicates()
        self.train_scen.remove_duplicates()

        self.train_inst = self.train_scen.feature_data
        self.test_inst = self.test_scen.feature_data
        self.train_performances = self.train_scen.performance_data
        self.test_performances = self.test_scen.performance_data
        self.train_ranking = self.train_scen.performance_rankings
        self.test_ranking = self.test_scen.performance_rankings

    # def test_fit(self):
    #     model = ll.LogLinearModel()
    #     for index, row in self.test_ranking.iterrows():
    #         instance_values = self.test_inst.loc[index].values
    #         print("True Performances", self.test_performances.loc[index].values)
    #         print("Predicted Performances", model.predict_performances(instance_values))
    #         print("\n")
    #         print("True Ranking", self.test_ranking.loc[index].values)
    #         print("Predicted Ranking", model.predict_ranking(instance_values))
    #         print("\n")

    def test_regression(self):
        model1 = ll.LogLinearModel(use_exp_for_regression=False)
        perf_bar = self.train_performances.to_numpy()
        inst_bar = self.train_inst.to_numpy()
        features, performances, rankings, sample_weights = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(
            self.train_inst, self.train_performances, order="asc")
        features = features[:1001]
        performances = performances[:1001]
        rankings = rankings[:1001]
        # quad_feature_transform = PolynomialFeatures(2)
        # print("features", features.shape)
        # quad_features = quad_feature_transform.fit_transform(features[:999])
        # print("quad features", quad_features.shape)
        # feature_vec = features[1000]
        # print("feature vec", feature_vec.shape)
        # quad_feature_vec = quad_feature_transform.transform(
        #     feature_vec.reshape(1, -1))
        # print("quad feature vec", quad_feature_vec.shape)
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
        lambda_value = 1.0
        model1.fit_np(2,
                      rankings,
                      features,
                      performances_max_inv,
                      lambda_value=lambda_value,
                      regression_loss="Squared",
                      maxiter=100,
                      log_losses=True, sample_weights=None)
        # baselines = [
        #     LinearRegression(),
        #     LinearRegression(),
        #     LinearRegression(),
        #     LinearRegression(),
        #     LinearRegression()
        # ]
        # for label in range(0, 5):
        #     print(perf_bar[:, label].shape)
        #     print(inst_bar.shape)
            # baselines[label].fit(inst_bar, perf_bar[:, label])
        taus = []
        for i, (index, row) in enumerate(self.train_performances.iterrows()):
            instance_values = self.train_inst.loc[index].values
            imputed_row = self.imputer.transform([instance_values])
            scaled_row = self.scaler.transform(imputed_row).flatten()
            print("predicted", maximum - model1.predict_performances(scaled_row) * max_inv)
            # print(
            #     "Predicted Performances Model LR", ",".join([
            #         str(baseline.predict(scaled_row.reshape(1, -1)))
            #         for baseline in baselines
            #     ]))
            print("truth performance", row.values)
            print("\n")
            print("Predicted Ranking", model1.predict_ranking(scaled_row))
            print("True Ranking", np.argsort(np.argsort(row.values)) + 1)
            print("\n")
            taus.append(kendalltau(np.argsort(np.argsort(row.values)) + 1, model1.predict_ranking(scaled_row))[0])
        print("Average tau", np.mean(taus))

#         sns.set_style("darkgrid")
#         df = model1.get_loss_history_frame()
#         print(df)
#         df = df.rename(columns={"NLL":"PL-NLL"})
#         df["$\lambda$ PL-NLL"] = lambda_value * df["PL-NLL"]
#         df["$(1 - \lambda)$ MSE"] = (1 - lambda_value) * df["MSE"]
#         df["TOTAL_LOSS"] = df["$\lambda$ PL-NLL"] + df["$(1 - \lambda)$ MSE"]
#         df = df.melt(id_vars=["iteration"])
#         plt.clf()
#         # plt.tight_layout()
#         # plt.annotate(text,(0,0), (0,-40), xycoords="axes fraction", textcoords="offset points", va="top")
#         print(df.head())
#         lp = sns.lineplot(x="iteration", y="value", hue="variable", data=df)
#         plt.title(self.scenario.scenario)
#         plt.show()

# if __name__ == "__main__":
#     unittest.main()