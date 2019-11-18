import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
import Corras.Util.ranking_util as util
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class TestLogLinearModelASLib(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLogLinearModelASLib, self).__init__(*args, **kwargs)
        self.scenario = scen.ASRankingScenario()
        self.scenario.read_scenario("aslib_data-aslib-v4.0/CSP-2010")
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
        model = ll.LogLinearModel()
        features, performances, rankings = util.construct_numpy_representation(self.train_inst, self.train_performances)
        print("raw features", features)
        features = self.imputer.fit_transform(features)
        features = self.scaler.fit_transform(features)
        print("scaled features", features)
        print(performances)
        print(rankings)
        rankings = util.ordering_to_ranking_matrix(rankings)
        model.fit_np(rankings,features,performances,lambda_value=1,regression_loss="Squared",maxiter=1000)
        for index, row in self.train_performances.iterrows():
            instance_values = self.train_inst.loc[index].values
            imputed_row = self.imputer.transform([instance_values])
            scaled_row = self.scaler.transform(imputed_row).flatten()
            print("predicted", model.predict_performances(instance_values))
            print("truth performance", row.values)
            print("predicted ranking", model.predict_ranking(instance_values))
            print("\n")
            print("\n")
        print("features", features)
        print(model.weights)

if __name__ == "__main__":
    unittest.main()