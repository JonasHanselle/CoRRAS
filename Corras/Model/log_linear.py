import numpy as np
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario

class LogLinearModel:
    
    def __init__(self):
        self.weights = None

    def fit(self, data : ASRankingScenario):
        pass

    def predict(self, features : np.ndarray):
        return None