import unittest
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll

class TestLogLinearModel(unittest.TestCase):

    def test_fit(self):
        scenario = scen.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/CPMP-2015")
        scenario.compute_rankings()
        print("length before", len(scenario.performance_rankings))
        scenario.remove_duplicates()
        print("length after", len(scenario.performance_rankings))
        # test_scen, train_scen = scenario.get_split(indx=5)
        # model = ll.LogLinearModel()

        # model.fit(train_scen)

        # for index in range(1,10):
        #     split = scenario.get_split(indx=index)
        #     train_scenario = split[0]
        #     test_scenario = split[1]
        #     train_scenario.performance_rankings.to_csv("train_ranking_" + str(index) + ".csv")
        #     train_scenario.performance_data.to_csv("train_performance_" + str(index) + ".csv")
        #     test_scenario.performance_rankings.to_csv("test_ranking_" + str(index) + ".csv")
        #     test_scenario.performance_data.to_csv("test_performance_" + str(index) + ".csv")
        assert(True)

if __name__ == "__main__":
    unittest.main()