import unittest
import numpy as np
import numpy as onp
import pandas as pd
import Corras.Scenario.aslib_ranking_scenario as scen
import Corras.Model.neural_net_hinge as nn_hinge
import Corras.Util.ranking_util as util
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

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
        self.train_size = 2500
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

        self.algorithms = ["alg1", "alg2", "alg3", "alg4", "alg5"]

        self.train_inst = pd.DataFrame(
            data=features_train, columns=["a", "b", "c", "d"])
        self.test_inst = pd.DataFrame(
            data=features_test, columns=["a", "b", "c", "d"])
        self.train_performances = pd.DataFrame(
            data=performances_train, columns=self.algorithms)
        self.test_performances = pd.DataFrame(
            data=performances_test, columns=self.algorithms)
        self.train_ranking = pd.DataFrame(
            data=rankings_train, columns=self.algorithms)
        self.test_ranking = pd.DataFrame(
            data=rankings_test, columns=self.algorithms)

        print("train instances", self.train_inst)
        print("test instances", self.test_inst)
        print("train performances", self.train_performances)
        print("train rankings", self.train_ranking)
        print("test performances", self.test_performances)
        print("test rankings", self.test_ranking)

    def test_regression(self):
        seed = 15
        train_performances = self.train_performances
        train_features = self.train_inst
        test_performances = self.test_performances
        test_features = self.test_inst
        # print(train_performances)
        # melted_train_performances = pd.melt(train_performances.reset_index(
        # ), id_vars="index", value_name="performance", var_name="algorithm")
        # print(melted_train_performances)
        # joined_train_data = train_features.join(
        #     melted_train_performances.set_index("index"))
        # joined_train_data["algorithm"] = joined_train_data["algorithm"].astype(
        #     "category")
        # for index, algorithm in enumerate(self.algorithms):

        # train_data = encoder.fit_transform(joined_train_data)

        dataset = []
        indices = []
        for inst_index, row in train_performances.iterrows():
            for alg_index, algorithm in enumerate(self.algorithms):
                cur_features = self.train_inst.loc[inst_index]
                alg_enc = len(self.algorithms) * [0]
                alg_enc[alg_index] = 1
                alg_one_hot = pd.Series(alg_enc, index=self.algorithms)
                cur_performance = row.loc[algorithm]
                new_row = cur_features.append(alg_one_hot).append(
                    pd.Series(data=[cur_performance], index=["performance"]))
                dataset.append(new_row)
                indices.append(inst_index)

        df_train = pd.DataFrame(dataset, index=indices)
        print(df_train)

        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        # X_test = test_data.iloc[:, :-1]
        # y_test = test_data.iloc[:, -1]

        X_train = X_train.to_numpy().astype(np.float64)
        y_train = y_train.to_numpy().astype(np.float64)

        mask = y_train <= 2500.0
        timeouted_runs = ~mask

        # the time at which the observation ends is actually the cutoff, not the par10
        y_train[timeouted_runs] = 700.0
        structured_y_train = np.rec.fromarrays([mask, y_train],
                                               names="terminated,runtime")
        print(structured_y_train)

        model = RandomSurvivalForest(n_estimators=100,
                                     n_jobs=1,
                                     random_state=seed)
        print("Starting to fit model")
        model.fit(X_train, structured_y_train)

        # evaluate model
        result_data_rsf = []
        for index, row in test_features.iterrows():
            predicted_performances = []
            # predicted_performances = [-1] * len(
            #     test_scenario.performance_data.columns)
            for alg_index, algorithm in enumerate(self.algorithms):
                # print(algorithm)
                cur_features = row
                alg_enc = len(self.algorithms) * [0]
                alg_enc[alg_index] = 1
                alg_one_hot = pd.Series(alg_enc, index=self.algorithms)
                new_row = cur_features.append(alg_one_hot)
                new_row_np = new_row.to_numpy().astype(np.float64).reshape(1, -1)
                predicted_performance = model.predict(new_row_np)
                predicted_performances.append(predicted_performance[0])
            result_data_rsf.append(
                [index, *predicted_performances])

        performance_cols = [
            x + "_performance" for x in self.algorithms
        ]
        result_columns_rsf = ["problem_instance"]
        result_columns_rsf += performance_cols
        results_rsf = pd.DataFrame(data=result_data_rsf,
                                   columns=result_columns_rsf)
        print(results_rsf)

        taus = []
        for index, row in self.test_performances.iterrows():
            true_performances = row.to_numpy()
            true_ranking = np.argsort(true_performances)
            predicted_scores = results_rsf.loc[index].to_numpy()[1:]
            predicted_ranking = np.argsort(predicted_scores)[::-1]
            print("true", true_performances[true_ranking])
            print("predicted", predicted_scores[predicted_ranking])
            print(predicted_scores)
            print()
            print("argmax", np.argmax(predicted_scores))
            print("true ranking", true_ranking)
            print("predicted ranking", predicted_ranking)
            print("\n")
            # print("predicted scores", predicted_scores)
            taus.append(kendalltau(true_ranking, predicted_ranking)[0])

        print("taus", taus)
        print("Average Kendalls tau", np.mean(taus))

if __name__ == "__main__":
    unittest.main()
