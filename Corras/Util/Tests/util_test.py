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
    #     scenario.read_scenario("aslib_data-aslib-v4.0/SAT12-RAND")
    #     features = scenario.feature_data
    #     performances = scenario.performance_data
    #     tensor = util.construct_ordered_tensor(features, performances)
    #     print(tensor)
    #     print(tensor.size * tensor.itemsize)

    # def test_np_representation_test(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/SAT12-RAND")
    #     features = scenario.feature_data
    #     performances = scenario.performance_data
    #     np_f, np_p, np_r = util.construct_numpy_representation(features,performances)
    #     print(np_f.size * np_f.itemsize + np_p.size * np_p.itemsize + np_r.size * np_r.itemsize)
    #     print("features shape", np_f.shape)
    #     print("performances shape", np_p.shape)
    #     print("rankings shape", np_r.shape)

    # def test_ordering_to_ranking_list(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
    #     features = scenario.feature_data
    #     performances = scenario.performance_data
    #     np_f, np_p, np_r = util.construct_numpy_representation(features,performances)
    #     print("performances",np_p)
    #     print("rankings",np_r)
    #     print("list",util.ordering_to_ranking_list(np_r))


    # def test_ordering_to_ranking_list(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/SAT11-INDU")
    #     features = scenario.feature_data
    #     performances = scenario.performance_data
    #     np_f, np_p, np_r = util.construct_numpy_representation_with_list_rankings(features,performances,max_rankings_per_instance=5,seed=15, pairs=True)
    #     print("performances",np_p)
    #     print("rankings",np_r)
    #     print("list",util.ordering_to_ranking_frame(np_r))

    # def test_ordering_to_ranking_frame(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/MIP-2016")
    #     features = scenario.feature_data
    #     performances = scenario.performance_data
    #     np_f, np_p, np_r = util.construct_numpy_representation_with_pairs_of_rankings(features,performances,max_rankings_per_instance=500,seed=15)

    def test_ordered_pairs(self):
        scenario = scen.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/SAT12-ALL")
        performances = scenario.performance_data
        performances = performances / (10*scenario.algorithm_cutoff_time)
        feat,perf,rank = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(scenario.feature_data,performances,max_pairs_per_instance=1,seed=2, skip_value = float(scenario.algorithm_cutoff_time*10))
        print(feat.shape)
        print(perf.shape)
        print(rank.shape)
        print(feat[:5,:5])
        print(perf[:5])
        print(rank[:5])
        print(weights[:5])
        feat,perf,rank = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(scenario.feature_data,performances,max_pairs_per_instance=1,seed=2, order="desc", skip_value = float(scenario.algorithm_cutoff_time*10))
        print("\n")
        print(feat[:5,:5])
        print(perf[:5])
        print(rank[:5])
        print(weights[:5])


    # def test_ranking_conversion(self):
    #     scenario = scen.ASRankingScenario()
    #     scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
    #     scenario.compute_rankings(True)
    #     # scenario.remove_duplicates()
    #     # print(scenario.performance_rankings)
    #     print(scenario.performance_rankings.iloc[2])
    #     print(util.ordering_to_ranking(scenario.performance_rankings.iloc[2]))

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