import unittest
import Corras.Util.ranking_util as util
import Corras.Scenario.aslib_ranking_scenario as scen

class UtilTests(unittest.TestCase):

    def test_rankings(self):
        scenario = scen.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        scenario.compute_rankings()
        # print(scenario.performance_rankings)
        # print(scenario.performance_rankings_inverse)
        print(util.break_ties_of_ranking(scenario.performance_rankings))

        pass

if __name__ == "__main__":
    unittest.main()