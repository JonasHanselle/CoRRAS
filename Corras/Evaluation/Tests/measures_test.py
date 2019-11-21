import unittest
import numpy as np
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
from Corras.Evaluation.evaluation import ndcg_at_k, compute_relevance_scores_equi_width

class EvaluationMeasuresTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(EvaluationMeasuresTest, self).__init__(*args, **kwargs)


    def test_ndcg(self):
        relevance_scores = np.asarray([0,2,2,1,0])
        prediction = np.asarray([1,5,4,2,3])
        print(ndcg_at_k(prediction,relevance_scores,5))

if __name__ == "__main__":
    unittest.main()