import numpy as np
import pandas as pd
import aslib_scenario.aslib_scenario as scenario

class ASRankingScenario(scenario.ASlibScenario):
    """Extension of the ASlibScenario provided by Marius Lindauer. Allows to store rankings.
    """
    def __init__(self):
        super(ASRankingScenario, self).__init__()
        self.performance_rankings = None
    
    def compute_rankings(self):
        """Computes the rankings according to the performance an algorithm achieved.
        Currently only runtime is supported as a performance measure.
        """
        if self.performance_data is None:
            self.logger.error("Please read in performance data!")
        elif self.performance_measure != ["runtime"]:
            self.logger.error("Currently we only support runtime as a performance measure!")
        else:
            performances = self.performance_data
            rankings = performances.rank(axis=1).astype(int)
            self.performance_rankings = rankings