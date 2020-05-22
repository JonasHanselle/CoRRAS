import numpy as np
import pandas as pd
import aslib_scenario.aslib_scenario as scenario


class ASRankingScenario(scenario.ASlibScenario):
    """Extension of the ASlibScenario provided by Marius Lindauer. Allows to store rankings.
    """

    def __init__(self):
        super(ASRankingScenario, self).__init__()
        self.performance_rankings = None
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
            performances = self.performance_data
            rankings = performances.rank(axis=1).astype(int)
            self.performance_rankings = rankings
            inverse_rankings = np.add(self.performance_rankings.values.argsort(),1) 
            self.performance_rankings_inverse = pd.DataFrame(data=inverse_rankings, index=self.performance_rankings.index, columns=self.performance_rankings.columns)

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
        if self.performance_rankings is None:
            self.logger.error("No rankings computed!")
            return None
        else:
            # remove duplicate rankings
            for index, row in self.performance_rankings.iterrows():
                if len(row) > len(set(row.tolist())):
                    self.performance_rankings.drop(index, inplace=True)
                    self.performance_rankings_inverse.drop(index, inplace=True)
