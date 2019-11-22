import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.neural_net as nn
import Corras.
from sklearn.preprocessing import StandardScaler

class TestNeuralNetworkSynthetic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestNeuralNetworkSynthetic, self).__init__(*args, **kwargs)
        self.train_size = 250
        self.test_size = 10
        self.noise_factor = 0.0
        features_train = np.asarray(onp.random.randint(low=0, high=30, size=(self.train_size,4)))
        features_test = np.asarray(onp.random.randint(low=0, high=30, size=(self.test_size,4)))
        def create_performances(feature_list):
            performances = []
            for features in feature_list:
                # generate performances as functions linear in the features
                performance_1 = 5 * features[0] + 2 * features[1] + 7 * features[2] + 42
                performance_2 =  3 * features[1] + 5 * features[3] + 14
                performance_3 = 2 * features[0] + 4 * features[1] + 11 * features[3] + 77
                performance_4 = 7 * features[1] + 4 * features[0] + 11 * features[2] + features[3]
                performance_5 = 2 * features[1] + 9 * features[2] + 7 * features[3] + 12 + features[0]
                performances.append([performance_1, performance_2, performance_3, performance_4, performance_5])
            return performances

        performances_train = np.asarray(create_performances(features_train), dtype=np.float64)
        performances_test = np.asarray(create_performances(features_test), dtype=np.float64)

        features_train = np.asarray(features_train, dtype=np.float64)
        features_test = np.asarray(features_test, dtype=np.float64)
        
        rankings_train = np.argsort(np.argsort(np.asarray(performances_train))) + 1
        rankings_test = np.argsort(np.argsort(np.asarray(performances_test))) + 1

        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        self.train_inst = pd.DataFrame(data=features_train,columns=["a","b","c","d"])
        self.test_inst = pd.DataFrame(data=features_test,columns=["a","b","c","d"])
        self.train_performances = pd.DataFrame(data=performances_train,columns=["alg1","alg2","alg3","alg4","alg5"])
        self.test_performances = pd.DataFrame(data=performances_test,columns=["alg1","alg2","alg3","alg4","alg5"])
        self.train_ranking = pd.DataFrame(data=rankings_train,columns=["alg1","alg2","alg3","alg4","alg5"])
        self.test_ranking = pd.DataFrame(data=rankings_test,columns=["alg1","alg2","alg3","alg4","alg5"])

        print("train instances", self.train_inst)
        print("test instances", self.test_inst)
        print("train performances", self.train_performances)
        print("train rankings", self.train_ranking)
        print("test performances", self.test_performances)
        print("test rankings", self.test_ranking)


    def test_regression(self):
        model = nn.NeuralNetwork()
        model.fit(self.train_ranking,self.train_inst,self.train_performances,lambda_value=0,regression_loss="Squared")        
        rankings = util.ordering_to_ranking_list(self.train_ranking.values)
        model.fit_list(rankings,self.train_inst.values,self.train_performances.values,lambda_value=0,regression_loss="Squared")
        for index, row in self.test_inst.iterrows():
            print("True Performances", self.test_performances.loc[index].values)
            print("Predicted Performances", model.predict_performances(row.values))
            print("\n")
            print("True Ranking", self.test_ranking.loc[index].values)
            print("Predicted Ranking", model.predict_ranking(row.values))
            print("\n")


if __name__ == "__main__":
    unittest.main()