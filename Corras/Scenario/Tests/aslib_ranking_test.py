import unittest
import Corras.Scenario.aslib_ranking_scenario as scen
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ASLibRankingTest(unittest.TestCase):

    def test_rankings(self):
        scenario = scen.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/MIP-2016")
        scenario.compute_rankings()
        print(scenario.performance_rankings)
        # print(scenario.performance_rankings_inverse)
        scenario.compute_rankings(True)
        sns.distplot(scenario.performance_data.iloc[:,0],hist=False,rug=True,)
        
        plt.show()

if __name__ == "__main__":
    unittest.main()