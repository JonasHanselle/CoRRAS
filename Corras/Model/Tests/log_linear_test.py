import unittest
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.log_linear as ll
from sklearn.preprocessing import StandardScaler

class TestLogLinearModel(unittest.TestCase):

    def test_fit(self):
        test_scen = scen.ASRankingScenario()
        test_scen.read_scenario("aslib_data-aslib-v4.0/CSP-2010")
        test_scen.compute_rankings()
        model = ll.LogLinearModel()
        model.fit(test_scen)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(test_scen.feature_data)
        feature_vector = test_scen.feature_data.iloc[15].values
        feature_vector_scaled = features_scaled[15]
        print(feature_vector)
        print(feature_vector_scaled)
        model.predict(feature_vector)
        model.predict(feature_vector_scaled)
        assert(True)

if __name__ == "__main__":
    unittest.main()