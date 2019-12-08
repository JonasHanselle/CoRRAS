import unittest
import autograd.numpy as np
import numpy as onp
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.neural_net_hinge as nn_hinge
import Corras.Util.ranking_util as util
from sklearn.preprocessing import StandardScaler

class TestNeuralNetworkHingeSynthetic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestNeuralNetworkHingeSynthetic, self).__init__(*args, **kwargs)
        self.train_size = 2500
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
                # performances.append([performance_1, performance_5])
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
        # self.train_performances = pd.DataFrame(data=performances_train,columns=["alg1", "alg2"])
        # self.test_performances = pd.DataFrame(data=performances_test,columns=["alg1", "alg2"])
        # self.train_ranking = pd.DataFrame(data=rankings_train,columns=["alg1", "alg2"])
        # self.test_ranking = pd.DataFrame(data=rankings_test,columns=["alg1", "alg2"])

        print("train instances", self.train_inst)
        print("test instances", self.test_inst)
        print("train performances", self.train_performances)
        print("train rankings", self.train_ranking)
        print("test performances", self.test_performances)
        print("test rankings", self.test_ranking)

    def test_regression(self):
        model1 = nn_hinge.NeuralNetworkSquaredHinge()

        inst,perf,rank = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features(self.train_inst,self.train_performances,max_pairs_per_instance=15,seed=15)
        print(inst)
        print(perf)
        print(rank)
        rank = rank.astype("int32")
        model1.fit(5, rank, inst, perf,lambda_value=1, epsilon_value=1, regression_loss="Squared", num_epochs=100, learning_rate=1, batch_size=32, early_stop_interval=2, patience=8)

        for index, row in self.test_inst.iterrows():
            print("True Performances", self.test_performances.loc[index].values)
            print("Predicted Performances Model 1", model1.predict_performances(row.values))
            print("\n")
            print("True Ranking", self.test_ranking.loc[index].values)
            print("Predicted Ranking Model 1", model1.predict_ranking(row.values))
            print("\n")

        # print("loss hist", model1.loss_history)

if __name__ == "__main__":
    unittest.main()