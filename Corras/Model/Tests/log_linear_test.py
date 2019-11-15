import unittest
import autograd.numpy as np
import pandas as pd
from autograd import grad
import sys
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
from sklearn.preprocessing import StandardScaler

class TestLogLinearModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLogLinearModel, self).__init__(*args, **kwargs)
        np.set_printoptions(threshold=sys.maxsize)
        self.scenario = scen.ASRankingScenario()
        self.scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        self.scenario.compute_rankings()
        # preprocessing of data
        self.scaler = StandardScaler()
        self.scenario.feature_data[self.scenario.feature_data.columns] = self.scaler.fit_transform(self.scenario.feature_data[self.scenario.feature_data.columns])
        self.test_scen, self.train_scen = self.scenario.get_split(indx=5)
        self.test_scen.remove_duplicates()
        self.train_scen.remove_duplicates()

    # def test_fit(self):
    #     model = ll.LogLinearModel()
    #     model.fit(self.test_scen)
    #     assert(True)

    def test_gradient(self):
        model = ll.LogLinearModel()
        num_labels = len(self.train_scen.algorithms)
        num_features = len(self.train_scen.features)+1
        self.weights = np.random.rand(num_labels, num_features)
        # take some direction
        d = np.eye(N=1, M=len(self.weights.flatten()), k=5)

        epsilon = 0.0001
        def f(w):
            return model.negative_log_likelihood(self.train_scen.performance_rankings,self.train_scen.feature_data,w)            
        def g(w):
            return model.vectorized_nll(self.train_scen.performance_rankings.values,self.train_scen.feature_data.loc[self.train_scen.performance_rankings.index].values,w)
        f_prime = grad(f)
        g_prime = grad(g)
        w = self.weights
        gradient_step_f = f_prime(w)
        # print("w+e", w+epsilon*(np.reshape(d,(num_labels,num_features))))
        # print("w-e", w-epsilon*(np.reshape(d,(num_labels,num_features))))
        # print("f(w+e)", f(w+epsilon*(np.reshape(d,(num_labels,num_features)))))
        # print("f(w-e)", f(w-epsilon*(np.reshape(d,(num_labels,num_features)))))
        local_finite_approx_f = (f(w+epsilon*(np.reshape(d,(num_labels,num_features)))) - f(w-epsilon*(np.reshape(d,(num_labels,num_features))))) / (2 * epsilon)
        print("local finite approximation f", local_finite_approx_f)
        # print("gradient step f", gradient_step_f.flatten()[5])
        gradient_step_g = g_prime(w)
        # print("w+e", w+epsilon*(np.reshape(d,(num_labels,num_features))))
        # print("w-e", w-epsilon*(np.reshape(d,(num_labels,num_features))))
        # print("g(w+e)", g(w+epsilon*(np.reshape(d,(num_labels,num_features)))))
        # print("g(w-e)", g(w-epsilon*(np.reshape(d,(num_labels,num_features)))))
        local_finite_approx_g = (g(w+epsilon*(np.reshape(d,(num_labels,num_features)))) - g(w-epsilon*(np.reshape(d,(num_labels,num_features))))) / (2 * epsilon)
        print("local finite approximation g", local_finite_approx_g)
        # print("gradient step g", gradient_step_g.flatten()[5])
        self.assertAlmostEqual(gradient_step_f.flatten()[5], local_finite_approx_f)        
        self.assertAlmostEqual(gradient_step_g.flatten()[5], local_finite_approx_g)

if __name__ == "__main__":
    unittest.main()