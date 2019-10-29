import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
from sklearn.preprocessing import StandardScaler

class TestLogLinearModelSynthetic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLogLinearModelSynthetic, self).__init__(*args, **kwargs)
        features_train = np.random.randint(low=0, high=30, size=(250,3))
        features_test = np.random.randint(low=0, high=30, size=(10,3))
        def create_ranking(feature_list):
            rankings = []
            for features in feature_list:
                if features[0] > 15:
                    rankings.append([1,2,3])
                # elif features[0] < 15 and features[1] < 15 and features[2] < 15:
                #     rankings.append([1,3,2])
                # elif features[0] > 15 and features[1] < 15 and features[2] > 15:
                #     rankings.append([2,1,3])
                # elif features[0] > 15 and features[1] < 15 and features[2] < 15:
                #     rankings.append([2,3,1])
                # elif features[0] < 15 and features[1] > 15 and features[2] > 15:
                #     rankings.append([3,1,2])
                else:
                    rankings.append([3,2,1])
            return rankings
        print("train data")
        rankings_train = create_ranking(features_train)
        print("test data")
        rankings_test = create_ranking(features_test)
    
        features_train = np.asarray(features_train, dtype=np.float64)
        features_test = np.asarray(features_test, dtype=np.float64)
        rankings_train = np.asarray(rankings_train)
        rankings_test = np.asarray(rankings_test)

        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        self.train_inst = pd.DataFrame(data=features_train,columns=["a","b","c"])
        self.test_inst = pd.DataFrame(data=features_test,columns=["a","b","c"])
        self.train_ranking = pd.DataFrame(data=rankings_train,columns=["alg1","alg2","alg3"])
        self.test_ranking = pd.DataFrame(data=rankings_test,columns=["alg1","alg2","alg3"])
        self.train_ranking_inverse = pd.DataFrame(data=np.argsort(rankings_train)+1,columns=["alg1","alg2","alg3"])
        self.test_ranking_inverse = pd.DataFrame(data=np.argsort(rankings_test)+1,columns=["alg1","alg2","alg3"])

        print(self.train_inst)
        print(self.test_inst)
        print(self.train_ranking)
        print(self.test_ranking)
        print(self.train_ranking_inverse)
        print(self.test_ranking_inverse)


    # def test_fit(self):
    #     # test model 
    #     model = ll.LogLinearModel()
    #     model.fit(self.train_ranking, self.train_ranking_inverse, self.train_inst)        
    #     for index, row in self.test_inst.iterrows():
    #         print("True Ranking", self.test_ranking.loc[index].values)
    #         print("Prediction", model.predict(row.values))
    #         print("\n")

    def test_gradient(self):
        nll = ll.PLNegativeLogLikelihood() 
        num_labels = len(self.train_ranking.columns)
        num_features = len(self.train_inst.columns)+1
        # self.weights = np.random.rand(num_labels, num_features)
        self.weights = np.ones((num_labels, num_features))
        # take some direction
        d = np.eye(N=1, M=len(self.weights.flatten()), k=5)
        epsilon = 0.001
        gradient = nll.first_derivative(self.train_ranking,self.train_ranking_inverse,self.train_inst, self.weights)
        print("gradient", gradient)
        gradient_step = np.dot(d,gradient.flatten())
        print("step", np.dot(d,gradient.flatten()))
        def f(w):
            return nll.negative_log_likelihood(self.train_ranking,self.train_inst,w)
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