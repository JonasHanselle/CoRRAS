import unittest
import autograd.numpy as np
import numpy as onp
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.neural_net_hinge as nn_hinge
import Corras.Util.ranking_util as util
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder


class TestNeuralNetworkHingeSynthetic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestNeuralNetworkHingeSynthetic, self).__init__(*args, **kwargs)
        self.train_size = 250
        self.test_size = 10
        self.noise_factor = 0.0
        features_train = np.asarray(onp.random.randint(
            low=0, high=30, size=(self.train_size, 4)))
        features_test = np.asarray(onp.random.randint(
            low=0, high=30, size=(self.test_size, 4)))

        def create_performances(feature_list):
            inst = 1
            performances = []
            for features in feature_list:
                # generate performances as functions linear in the features
                performance_1 = 5 * features[0] + 2 * \
                    features[1] + 7 * features[2] + 42
                performance_2 = 3 * features[1] + 5 * features[3] + 14
                performance_3 = 2 * features[0] + 4 * \
                    features[1] + 11 * features[3] + 77
                performance_4 = 7 * \
                    features[1] + 4 * features[0] + \
                    11 * features[2] + features[3]
                performance_5 = 2 * \
                    features[1] + 9 * features[2] + 7 * \
                    features[3] + 12 + features[0]
                performances.append(
                    [performance_1, performance_2, performance_3, performance_4, performance_5])
                # performances.append([performance_1, performance_5])
                inst += 1
            return performances

        performances_train = np.asarray(
            create_performances(features_train))
        performances_test = np.asarray(
            create_performances(features_test))

        features_train = np.asarray(features_train)
        features_test = np.asarray(features_test)

        rankings_train = np.argsort(np.argsort(
            np.asarray(performances_train))) + 1
        rankings_test = np.argsort(np.argsort(
            np.asarray(performances_test))) + 1

        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        self.train_inst = pd.DataFrame(
            data=features_train, columns=["a", "b", "c", "d"])
        self.test_inst = pd.DataFrame(
            data=features_test, columns=["a", "b", "c", "d"])
        self.train_performances = pd.DataFrame(data=performances_train, columns=[
                                               "alg1", "alg2", "alg3", "alg4", "alg5"])
        self.test_performances = pd.DataFrame(data=performances_test, columns=[
                                              "alg1", "alg2", "alg3", "alg4", "alg5"])
        self.train_ranking = pd.DataFrame(data=rankings_train, columns=[
                                          "alg1", "alg2", "alg3", "alg4", "alg5"])
        self.test_ranking = pd.DataFrame(data=rankings_test, columns=[
                                         "alg1", "alg2", "alg3", "alg4", "alg5"])

        print("train instances", self.train_inst)
        print("test instances", self.test_inst)
        print("train performances", self.train_performances)
        print("train rankings", self.train_ranking)
        print("test performances", self.test_performances)
        print("test rankings", self.test_ranking)

    def test_regression(self):
        train_performances = self.train_performances
        train_features = self.train_inst
        test_performances = self.test_performances
        test_features = self.test_inst
        print(train_performances)
        melted_train_performances = pd.melt(train_performances.reset_index(
        ), id_vars="index", value_name="performance")
        print(melted_train_performances)
        joined_train_data = train_features.join(
            melted_train_performances.set_index("index"))
        joined_train_data["algorithm"] = joined_train_data["algorithm"].astype(
            "category")
        encoder = OneHotEncoder()
        train_data = encoder.fit_transform(joined_train_data)

        melted_test_performances = pd.melt(test_performances.reset_index(
        ), id_vars="index", value_name="performance")
        joined_test_data = test_features.join(
            melted_test_performances.set_index("index"))
        joined_test_data["algorithm"] = joined_test_data["algorithm"].astype(
            "category")
        test_data = encoder.transform(joined_test_data)

        print("joined train data", joined_train_data)

        # preprocessing
        imputer = SimpleImputer()
        scaler = StandardScaler()

        scalable_columns = [col for (i, col) in enumerate(
            train_data.columns) if "algorithm=" not in col and "performance" not in col]

        train_data[scalable_columns] = imputer.fit_transform(
            train_data[scalable_columns])
        train_data[scalable_columns] = scaler.fit_transform(
            train_data[scalable_columns])
        print("train data", train_data)
        X_train = train_data.iloc[:, :-1]
        print("X_train", X_train)
        y_train = train_data.iloc[:, -1]
        print("y_train", y_train)
        # X_test = test_data.iloc[:, :-1]
        # y_test = test_data.iloc[:, -1]

        model = RandomSurvivalForest(n_estimators=50,
                                     max_depth=3,
                                     min_samples_split=10,
                                     min_samples_leaf=15,
                                     max_features="sqrt",
                                     n_jobs=1,
                                     random_state=seed)

        mask = y_train != scenario.algorithm_cutoff_time * 10

        timeouted_runs = ~mask

        # the time at which the observation ends is actually the cutoff, not the par10
        y_train[timeouted_runs] = scenario.algorithm_cutoff_time

        structured_y_train = np.rec.fromarrays([mask, y_train],
                                               names="terminated,runtime")

        print(structured_y_train)

        print("Starting to fit model")
        model.fit(X_train, structured_y_train)

        for index, row in self.test_inst.iterrows():
            print("True Performances",
                  self.test_performances.loc[index].values)
            print("Predicted Performances Model 1",
                  model1.predict_performances(row.values))
            print("\n")


if __name__ == "__main__":
    unittest.main()
