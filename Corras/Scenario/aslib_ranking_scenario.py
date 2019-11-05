import numpy as np
import pandas as pd
import aslib_scenario.aslib_scenario as scenario
import Corras.Util.ranking_util as util

class ASRankingScenario(scenario.ASlibScenario):
    """Extension of the ASlibScenario provided by Marius Lindauer. Allows to store rankings.
    """

    def __init__(self):
        super(ASRankingScenario, self).__init__()
        self.performance_rankings = None
        # TODO remove inverse rankings, not needed anymore
        self.performance_rankings_inverse = None

    def compute_rankings(self, break_up_ties : bool = False):
        """Computes the rankings according to the performance an algorithm achieved.
        Currently only runtime is supported as a performance measure.
        """
        if self.performance_data is None:
            self.logger.error("Please read in performance data!")
        elif self.performance_measure != ["runtime"]:
            self.logger.error(
                "Currently we only support runtime as a performance measure!")
        else:
            self.performance_rankings, self.performance_rankings_inverse = util.compute_rankings(self.performance_data)
        if break_up_ties:
            self.performance_rankings = util.break_ties_of_ranking(self.performance_rankings)
            self.performance_rankings_inverse = util.break_ties_of_ranking(self.performance_rankings_inverse)

    def get_split(self, indx=1):
        """Get split for cross-validation TODO inherit docstring

        Keyword Arguments:
            indx {int} -- Number of fold (default: {1})

        Returns:
            (ASRankingScenario, ASRankingScenario) -- A tupel of ASRankingScenario objects, the first is the test data, the second the training data
        """
        test, training = super().get_split(indx=indx)
        test_insts = self.cv_data[
            self.cv_data["fold"] == float(indx)].index.tolist()
        training_insts = self.cv_data[
            self.cv_data.fold != float(indx)].index.tolist()
        test.performance_rankings = test.performance_rankings.drop(
            training_insts).sort_index()
        training.performance_rankings = training.performance_rankings.drop(
            test_insts).sort_index()
        return test, training

    def remove_duplicates(self):
        """deprecated
        """
        if self.performance_rankings is None:
            self.logger.error("No rankings computed!")
        else:
            # remove duplicate rankings
            util.remove_duplicates(self.performance_rankings, self.performance_rankings_inverse)