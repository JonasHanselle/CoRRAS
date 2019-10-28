import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
from sklearn.preprocessing import StandardScaler

class TestLogLinearModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLogLinearModel, self).__init__(*args, **kwargs)
        self.scenario = scen.ASRankingScenario()
        self.scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        self.scenario.compute_rankings()
        # preprocessing of data
        self.scaler = StandardScaler()
        self.scenario.feature_data[self.scenario.feature_data.columns] = self.scaler.fit_transform(self.scenario.feature_data[self.scenario.feature_data.columns])
        self.test_scen, self.train_scen = self.scenario.get_split(indx=5)
        self.test_scen.remove_duplicates()
        self.train_scen.remove_duplicates()

    def test_fit(self):
        model = ll.LogLinearModel()
        model.fit(self.test_scen)
        assert(True)


    def test_nll(self):
        features_train = [[5,1,7], [1,8,7], [11,8,0],[4,2,3],[1,5,2],[6,4,9],[2,8,3]]
        features_test =  [[3,8,4], [8,4,1], [2,6,9]]
        def create_ranking(feature_list):
            rankings = []
            for features in feature_list:
                if features[0] > features[2] and features[1] > features[2]:
                    print("case 1")
                    rankings.append([2,1,3])
                elif features[0] < features[2] and features[1] < features[2]:
                    print("case 2")
                    rankings.append([1,2,3])
                else:
                    print("case 3")
                    rankings.append([3,2,1])
            return rankings
        print("train data")
        rankings_train = create_ranking(features_train)
        print("test data")
        rankings_test = create_ranking(features_test)
    
        train_inst = pd.DataFrame(data=features_train,columns=["a","b","c"])
        test_inst = pd.DataFrame(data=features_test,columns=["a","b","c"])
        train_ranking = pd.DataFrame(data=rankings_train,columns=["alg1","alg2","alg3"])
        test_ranking = pd.DataFrame(data=rankings_test,columns=["alg1","alg2","alg3"])

        print(train_inst)
        print(test_inst)
        print(train_ranking)
        print(test_ranking)

        model = ll.LogLinearModel()

        # TODO test model 


    def test_gradient(self):
        nll = ll.PLNegativeLogLikelihood() 
        num_labels = len(self.train_scen.algorithms)
        num_features = len(self.train_scen.features)
        self.weights = np.random.rand(num_labels, num_features)
        # take some direction
        d = np.eye(N=1, M=len(self.weights.flatten()), k=5)
        epsilon = 0.01
        gradient = nll.first_derivative(self.train_scen.performance_rankings,self.train_scen.performance_rankings_inverse,self.train_scen.feature_data, self.weights)
        print("gradient", gradient)
        gradient_step = np.dot(d,gradient.flatten())
        print("step", np.dot(d,gradient.flatten()))
        def f(w):
            return nll.negative_log_likelihood(self.train_scen.performance_rankings,self.train_scen.feature_data,w)
        w = self.weights
        print("w+e", w+epsilon*(np.reshape(d,(num_labels,num_features))))
        print("w-e", w-epsilon*(np.reshape(d,(num_labels,num_features))))
        print("f(w+e)", f(w+epsilon*(np.reshape(d,(num_labels,num_features)))))
        print("f(w-e)", f(w-epsilon*(np.reshape(d,(num_labels,num_features)))))
        local_finite_approx = (f(w+epsilon*(np.reshape(d,(num_labels,num_features)))) - f(w-epsilon*(np.reshape(d,(num_labels,num_features))))) / (2 * epsilon)
        print("local finite approximation", local_finite_approx)
        self.assertAlmostEqual(gradient_step[0], local_finite_approx)

if __name__ == "__main__":
    unittest.main()