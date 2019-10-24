import unittest
import Corras.Util.aslib_ranking_scenario as scen
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestASLibRankingScenarios(unittest.TestCase):

    def test_computed_rankings(self):
        test_scen = scen.ASRankingScenario()
        test_scen.read_scenario("aslib_data/scenario_name")
        test_scen.compute_rankings()
        test_scen.performance_data.info()
        test_scen.performance_rankings.info()

        assert(True)

if __name__ == "__main__":
    unittest.main()