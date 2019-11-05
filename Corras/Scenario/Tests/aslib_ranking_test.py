import unittest
import Corras.Scenario.aslib_ranking_scenario as scen

class ASLibRankingTest(unittest.TestCase):

    def test_rankings(self):
        scenario = scen.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        scenario.compute_rankings()
        print(scenario.performance_rankings)
        # print(scenario.performance_rankings_inverse)
        scenario.compute_rankings(True)
        print(scenario.performance_rankings)
        # print(scenario.performance_rankings_inverse)

if __name__ == "__main__":
    unittest.main()