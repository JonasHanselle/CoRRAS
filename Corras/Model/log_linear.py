import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.optimize import minimize


class PLNegativeLogLikelihood:

    def __init__(self):
        pass

    def negative_log_likelihood(self, rankings: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Compute NLL w.r.t. the data in the given batch and the given weights

        Arguments:
            rankings {pandas.DataFrame} -- Data sample for computing the NNL
            features {pandas.DataFrame} -- Feature values for computing the NNL
            weights {np.ndarray} -- Weight vector, i.e. model parameters

        Returns:
            [float64] -- Negative log-likelihood
        """
        outer_sum = 0
        for index, ranking in rankings.iterrows():
            # print("index", index)
            feature_values = features.loc[index].values
            # print("feature values",  feature_values)
            inner_sum = 0
            for m in range(0,len(ranking)):
                remaining_sum = 0
                for j in range(m,len(ranking)):
                    # compute utility of remaining labels
                    remaining_sum += np.exp(np.dot(weights[ranking[j]-1],feature_values))
                inner_sum += np.log(np.exp(np.dot(weights[ranking[m]-1],feature_values))) - np.log(remaining_sum)
            outer_sum += inner_sum
        return -outer_sum               
        
    def first_derivative(self, rankings: pd.DataFrame, inverse_rankings : pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Computes the the gradient vectors of the nll 
        
        Arguments:
            rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]
        
        Returns:
            [type] -- [description]
        """
        gradients = np.zeros_like(weights)
        for a in range(0,gradients.shape[0]):
            for index, ranking in rankings.iterrows():
                current_features = features.loc[index].values
                if inverse_rankings.loc[index][a] >= 1:
                    gradients[a] += current_features
            for index, ranking in rankings.iterrows():
                current_features = features.loc[index].values
                for m in range(0,len(ranking)):
                    if inverse_rankings.loc[index][a] >= m:
                        denominator = 0
                        for j in range(m,len(ranking)):
                            denominator += np.exp(np.dot(weights[ranking[j]-1],current_features))
                        numerator = np.exp(np.dot(weights[a],current_features)) * current_features
                        fraction = numerator / denominator
                        gradients[a] -= fraction
        return np.negative(gradients)
        
    def second_derivative(self, rankings: pd.DataFrame, inverse_rankings : pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Computes the the gradient vector of the nll 
        
        Arguments:
            rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]
        
        Returns:
            [type] -- [description]
        """
        hessian = np.zeros(shape=(weights.shape[1],weights.shape[1]))
        return hessian

class LogLinearModel:

    def __init__(self):
        self.weights = None
        self.dataset = None

    def fit(self, dataset: ASRankingScenario):
        """Fits a label ranking model based on a log-linear utility function.

        Arguments:
            data {ASRankingScenario} -- Training data set
        """
        self.dataset = dataset
        num_labels = len(dataset.algorithms)
        num_features = len(dataset.features)
        self.weights = np.zeros(shape=(num_labels, num_features))
        nll = PLNegativeLogLikelihood()
        
        # minimize nnl
        def f(x):
            x = np.reshape(x,(num_labels, num_features))
            return nll.negative_log_likelihood(dataset.performance_rankings,dataset.feature_data, x)

        def f_prime(x):
            x = np.reshape(x,(num_labels, num_features))
            return nll.first_derivative(dataset.performance_rankings, dataset.performance_rankings_inverse,dataset.feature_data,x).flatten()

        flat_weights = self.weights.flatten()
        print(flat_weights.shape)
        result = minimize(f, flat_weights, method="L-BFGS-B", jac=None, options={"maxiter" : 10, "disp" : True})
        print("Result", result)

    def predict(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        utility_scores = np.exp(np.dot(self.weights.transpose(), features))
        print(utility_scores)
        return None
