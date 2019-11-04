import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.neural_net as nn
from sklearn.preprocessing import StandardScaler

class TestNeuralNetworkSynthetic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestNeuralNetworkSynthetic, self).__init__(*args, **kwargs)
        self.train_size = 250
        self.test_size = 10
        self.noise_factor = 0.0
        features_train = np.random.randint(low=0, high=30, size=(self.train_size,4))
        features_test = np.random.randint(low=0, high=30, size=(self.test_size,4))
        def create_performances(feature_list):
            performances = []
            for features in feature_list:
                # generate performances as functions linear in the features
                performance_1 = 5 * features[0] + 2 * features[1] + 7 * features[2] + 42
                performance_2 =  3 * features[1] + 5 * features[3] + 14
                performance_3 = 2 * features[0] + 4 * features[1] + 11 * features[3] + 77
                performances.append([performance_1, performance_2, performance_3])
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
        self.train_performances = pd.DataFrame(data=performances_train,columns=["alg1","alg2","alg3"])
        self.test_performances = pd.DataFrame(data=performances_test,columns=["alg1","alg2","alg3"])
        self.train_ranking = pd.DataFrame(data=rankings_train,columns=["alg1","alg2","alg3"])
        self.test_ranking = pd.DataFrame(data=rankings_test,columns=["alg1","alg2","alg3"])
        self.train_ranking_inverse = pd.DataFrame(data=np.argsort(rankings_train)+1,columns=["alg1","alg2","alg3"])
        self.test_ranking_inverse = pd.DataFrame(data=np.argsort(rankings_test)+1,columns=["alg1","alg2","alg3"])

        print("train instances", self.train_inst)
        print("test instances", self.test_inst)
        print("train performances", self.train_performances)
        print("train rankings", self.train_ranking)
        print("test performances", self.test_performances)
        print("test rankings", self.test_ranking)
        print("train rankings inverse", self.train_ranking_inverse)
        print("test rankings inverse", self.test_ranking_inverse)

    def test_regression(self):
        model = nn.NeuralNetwork()
        model.fit(self.train_ranking,self.train_ranking_inverse,self.train_inst,self.train_performances,lambda_value=0,regression_loss="Squared")        
        for index, row in self.test_inst.iterrows():
            print("Instance Features", row)
            print("True Performances", self.test_performances.loc[index].values)
            print("Predicted Performances", model.predict_performances(row.values))
            print("\n")
        #     print("True Ranking", self.test_ranking.loc[index].values)
        #     print("Predicted Ranking", model.predict_ranking(row.values))
        #     print("\n")


if __name__ == "__main__":
    unittest.main()