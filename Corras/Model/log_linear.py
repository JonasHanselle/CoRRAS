import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario


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
        
    def first_derivative(self, rankings: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Computes the first gradient 
        
        Arguments:
            rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return None

    def second_derivative(self):
        return None


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
        self.weights = np.ones(shape=(num_labels, num_features))

        # Start gradient descent
        minibatch_size = 1
        learning_rate = 0.1
        max_iter = 1
        nll = PLNegativeLogLikelihood()
        for i in range(0, max_iter):
            print(i)
            minibatch = dataset.performance_rankings.sample(minibatch_size)
            nll_value = nll.negative_log_likelihood(minibatch, dataset.feature_data, self.weights)
            print("nll", nll_value)

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
