import numpy as np
import pandas as pd
import aslib_scenario.aslib_scenario as scenario

class ASRankingScenario(scenario.ASlibScenario):
    def __init__(self):
        super(ASRankingScenario, self).__init__()
        self.performance_rankings = None
    
    def compute_rankings(self):
        if self.performance_data is None:
            self.logger.error("Please read in performance data!")
        elif self.performance_measure != ["runtime"]:
            self.logger.error("Currently we only support runtime as a performance measure!")
        else:
            performances = self.performance_data
            # rankings = np.argsort(performances, axis=1)
            rankings = performances.rank(axis=1).astype(int)
            self.performance_rankings = rankings