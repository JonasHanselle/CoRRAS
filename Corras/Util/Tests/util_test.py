import unittest
import Corras.Util.ranking_util as util
import Corras.Scenario.aslib_ranking_scenario as scen

class UtilTests(unittest.TestCase):

    # def test_rankings(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
    #     scenario.compute_rankings()
    #     # print(scenario.performance_rankings)
    #     # print(scenario.performance_rankings_inverse)
    #     print(util.break_ties_of_ranking(scenario.performance_rankings))

    # def test_tensor_construction(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
    #     scenario.compute_rankings()
    #     features = scenario.feature_data
    #     performances = scenario.performance_data
    #     print(util.construct_ordered_tensor(features, performances))

    def test_ranking_conversion(self):
        
        scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        scenario.compute_rankings()

    # def test_custom_tau(self):
    #     ranking_a = [1,3,5,7]
    #     ranking_b = [3,5,7]
    #     ranking_c = [0,4,8]
    #     ranking_d = [1,8,4]

    #     assert(util.custom_tau(ranking_a,ranking_b)==1)
    #     assert(util.custom_tau(ranking_a,ranking_c)==0)
    #     assert(util.custom_tau(ranking_c,ranking_d)==-1)
        
if __name__ == "__main__":
    unittest.main()