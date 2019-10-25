import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario

class LogLinearModel:

    def __init__(self):
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
        utility_scores = np.dot(self.weights.transpose(), features)
        print(utility_scores)
        return None