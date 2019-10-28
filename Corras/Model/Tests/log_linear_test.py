import unittest
import numpy as np
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

    def test_gradient(self):
        nll = ll.PLNegativeLogLikelihood()
        num_labels = len(self.train_scen.algorithms)
        num_features = len(self.train_scen.features)
        self.weights = np.random.rand(num_labels, num_features)
        # take some direction
        d = np.eye(N=1, M=len(self.weights.flatten()), k=5)
        epsilon = 0.0001
        gradient = nll.first_derivative(self.train_scen.performance_rankings,self.train_scen.performance_rankings_inverse,self.train_scen.feature_data, self.weights)
        print("gradient", gradient)
        gradient_step = np.dot(d,gradient.flatten())
        print("step", np.dot(d,gradient.flatten()))
        def f(w):
            return nll.negative_log_likelihood(self.train_scen.performance_rankings,self.train_scen.feature_data,w)
        w = self.weights
        local_finite_approx = (f(w+epsilon*(np.reshape(d,(num_labels,num_features)))) - f(w-epsilon*(np.reshape(d,(num_labels,num_features))))) / (2 * epsilon)
        print("local finite approximation", local_finite_approx)
        self.assertAlmostEqual(gradient_step[0], local_finite_approx)

if __name__ == "__main__":
    unittest.main()