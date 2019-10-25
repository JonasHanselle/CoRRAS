import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario

class PLNegativeLogLikelihood:
    
    def __init__(self, dataset=None, ):
        pass

    def negative_log_likelihood(self):
        return None

    def first_derivative(self):
        return None

    def second_derivative(self):
        return None

class LogLinearModel:

    def __init__(self, regression_penalty="absolute"):
        self.weights = None

    def fit(self, data : ASRankingScenario):
        """Fits a label ranking model based on a log-linear utility function.
        
        Arguments:
            data {ASRankingScenario} -- Training data set
        """
        num_labels = len(data.algorithms)
        num_features = len(data.features)
        self.weights = np.ones(shape=(num_features,num_labels))
        print(self.weights)

    def predict(self, features : np.ndarray):
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