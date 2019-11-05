import unittest
import jax.numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
import Corras.Util.ranking_util as util
from sklearn.preprocessing import StandardScaler

class TestLogLinearModelASLib(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLogLinearModelASLib, self).__init__(*args, **kwargs)
        self.scenario = scen.ASRankingScenario()
        self.scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        self.scenario.compute_rankings()
        # preprocessing of data
        self.scaler = StandardScaler()
        self.scenario.feature_data[self.scenario.feature_data.columns] = self.scaler.fit_transform(self.scenario.feature_data[self.scenario.feature_data.columns])
        self.test_scen, self.train_scen = self.scenario.get_split(indx=5)
        self.test_scen.remove_duplicates()
        self.train_scen.remove_duplicates()

        self.train_inst = self.train_scen.feature_data
        self.test_inst = self.test_scen.feature_data
        self.train_performances = self.train_scen.performance_data
        self.test_performances = self.test_scen.performance_data
        self.train_ranking = self.train_scen.performance_rankings
        self.test_ranking = self.test_scen.performance_rankings
        self.train_ranking_inverse = self.train_scen.performance_rankings_inverse
        self.test_ranking_inverse = self.test_scen.performance_rankings_inverse

    # def test_fit(self):
    #     model = ll.LogLinearModel()
    #     model.fit(self.train_ranking,self.train_ranking_inverse,self.train_inst,self.train_performances,lambda_value=0,regression_loss="Squared")        
    #     for index, row in self.test_ranking.iterrows():
    #         instance_values = self.test_inst.loc[index].values
    #         print("True Performances", self.test_performances.loc[index].values)
    #         print("Predicted Performances", model.predict_performances(instance_values))
    #         print("\n")
    #         print("True Ranking", self.test_ranking.loc[index].values)
    #         print("Predicted Ranking", model.predict_ranking(instance_values))
    #         print("\n")

    def test_nll(self):
        model = ll.LogLinearModel()
        tensor = util.construct_ordered_tensor(self.train_inst,self.train_performances)
        print(tensor)
        print(model.tensor_nll(None, tensor))

if __name__ == "__main__":
    unittest.main()