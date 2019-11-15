import autograd.numpy as np
from autograd import grad, hessian
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.optimize import minimize


class LogLinearModel:

    def __init__(self):
        self.weights = None

    def absolute_error(self, performances: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Compute absolute error for regression

        Arguments:
            performances {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]

        Returns:
            [type] -- [description]
        """
        loss = 0
        for row in performances.iterrows():
            current_performances = row.values
            feature_values = np.hstack((features.loc[index].values, [1]))
            utilities = np.exp(np.dot(weights, feature_values))
            inverse_utilities = np.reciprocal(utilities)
            loss += np.sum(np.absolute(np.subtract(current_performances, inverse_utilities)))
        return loss

    def squared_error(self, performances: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Compute squared error for regression

        Arguments:
            performances {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]

        Returns:
            [type] -- [description]
        """
        loss = 0
        for index, row in performances.iterrows():
            current_performances = row.values
            feature_values = np.hstack((features.loc[index].values, [1]))
            utilities = np.exp(np.dot(weights, feature_values))
            inverse_utilities = np.reciprocal(utilities)
            loss += np.sum(np.square(np.subtract(current_performances,
                                                 inverse_utilities)))
        return loss

    def negative_log_likelihood(self, rankings: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Compute NLL w.r.t. the data in the given batch and the given weights
        Arguments:
            rankings {pandas.DataFrame} -- Data sample for computing the NNL
            features {pandas.DataFrame} -- Feature values for computing the NNL
            weights {np.ndarray} -- Weight vector, i.e. model parameters
        Returns:
            [float64] -- Negative log-likelihood
        """
        outer_sum = 0
        outer_sum_first = 0
        sum0 = 0
        sum1 = 0
        sum2 = 0
        for index, ranking in rankings.iterrows():
            # add one column for bias
            feature_values = np.hstack((features.loc[index].values, [1]))
            inner_sum = 0
            ranking = np.argsort(ranking.values)
            for m in range(0, len(ranking)):
            # if ranking[m] > 0:
                remaining_sum = 0
                for j in range(m, len(ranking)):
                    # compute utility of remaining labels
                    # if ranking[j] > 0:
                        # print("utility", np.exp(
                        #     np.dot(weights[ranking[j]], feature_values)))
                        remaining_sum += np.exp(
                            np.dot(weights[ranking[j]], feature_values))
                        print("summand", np.exp(
                            np.dot(weights[ranking[j]], feature_values)))
                print("remaining_sum", remaining_sum)
                print("remaining log", m, np.log(remaining_sum))
                sum0 += np.log(remaining_sum)
                sum1 += remaining_sum
                print("remaining cumsum", sum1)
                sum2 += np.dot(weights[ranking[m]], feature_values)
                # print("weighted:", np.dot(weights[ranking[m]], feature_values))
                inner_sum += np.log(remaining_sum) - \
                    np.dot(weights[ranking[m]], feature_values)
                # print("inner_summands", np.log(remaining_sum), np.dot(weights[ranking[m]], feature_values))
            # print("inner_sum", inner_sum)
            outer_sum += inner_sum
        print("weighted feature sum", sum2)
        print("remaining logs sum", sum0)
        print("difference", sum0 - sum2)
        print("outer_sum", outer_sum)
        return outer_sum

    def vectorized_nll(self, rankings, features, weights):
        """Compute NLL w.r.t. the data in the given batch and the given weights

        Arguments:
            rankings {np.ndarray} -- Data sample for computing the NNL
            features {np.ndarray} -- Feature values for computing the NNL
            weights {np.ndarray} -- Weight vector, i.e. model parameters

        Returns:
            [float64] -- Negative log-likelihood
        """
        nll = 0
        features = np.hstack((features,np.ones((features.shape[0],1))))
        ordered_weights = weights[np.argsort(rankings)]
        weighted_features = np.tensordot(ordered_weights,features, (2,1))
        sum1 = np.sum(weighted_features,1)
        utilities = np.exp(weighted_features)
        new_logs = []
        for m in range(0,rankings.shape[1]):
            new_logs.append(np.log(np.sum(utilities[0,m:],0)))
        new_logs = np.array(new_logs)
        print("logs", new_logs)
        print("utilitiesum1", np.sum(weighted_features[0]))
        print("shape", utilities.shape)
        print("remainer sum1", np.sum(new_logs[0]))
        outer_nll = sum1[0,0]
        print("outer_nll", outer_nll)
        return outer_nll

    def fit(self, rankings: pd.DataFrame, inverse_rankings: pd.DataFrame, features: pd.DataFrame, performances: pd.DataFrame, lambda_value=0.5, regression_loss="Absolute", maxiter=1000):
        """[summary]

        Arguments:
            rankings {pd.DataFrame} -- [description]
            inverse_rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]

        Returns:
            [type] -- [description]
        """
        num_labels = len(rankings.columns)
        # add one column for bias
        num_features = len(features.columns)+1

        self.weights = np.ones(
            shape=(num_labels, num_features)) / num_features * num_labels
        print("shape",self.weights.shape)
        nll = self.negative_log_likelihood
        reg_loss = None
        if regression_loss == "Absolute":
            reg_loss = self.absolute_error
        elif regression_loss == "Squared":
            reg_loss = self.squared_error

        # minimize loss function
        def f(x):
            x = np.reshape(x, (num_labels, num_features))
            if lambda_value == 0:
                return reg_loss(performances, features, x)
            elif lambda_value == 1:
                return nll(rankings, features, x)
            return lambda_value * nll(rankings, features, x) + (1 - lambda_value) * reg_loss(performances, features, x)

        jac = grad(f)

        flat_weights = self.weights.flatten()
        result = minimize(f, flat_weights, method="L-BFGS-B",
                          jac=jac, options={"maxiter": maxiter, "disp": True})

        print("Result old", result)
        self.weights = np.reshape(result.x, (num_labels, num_features))
        print("Weights old", self.weights)


    def fit_np(self, rankings, inverse_rankings, features, performances, lambda_value=0.5, regression_loss="Absolute", maxiter=1000):
        """[summary]

        Arguments:
            rankings {np.ndarray} -- [description]
            inverse_rankings {np.ndarray} -- [description]
            features {np.ndarray} -- [description]

        Returns:
            [type] -- [description]
        """
        num_labels = rankings.shape[1]
        # add one column for bias
        num_features = features.shape[1]+1
        print("labels", num_labels, "features", num_features)
        self.weights = np.random.rand(num_labels, num_features)
        nll = self.vectorized_nll
        reg_loss = None
        if regression_loss == "Absolute":
            reg_loss = self.absolute_error
        elif regression_loss == "Squared":
            reg_loss = self.squared_error

        # minimize loss function
        def g(x):
            x = np.reshape(x, (num_labels, num_features))
            return nll(rankings, features, x)
            # if lambda_value == 0:
            #     return reg_loss(performances, features, x)
            # elif lambda_value == 1:
            #     return nll(rankings, features, x)
            # return lambda_value * nll(rankings, features, x) + (1 - lambda_value) * reg_loss(performances, features, x)

        jac = grad(g)

        flat_weights = self.weights.flatten()
        result = minimize(g, flat_weights, method="TNC",
                          jac=jac, options={"maxiter": maxiter, "disp": True})

        print("Result new", result)
        self.weights = np.reshape(result.x, (num_labels, num_features))
        print("Weights new", self.weights)

    def predict_performances(self, features: np.ndarray):
        """Predict a vector of performance values.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        utility_scores = np.exp(np.dot(self.weights, features))
        return np.reciprocal(utility_scores)

    def predict_ranking(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        utility_scores = np.exp(np.dot(self.weights, features))
        return np.argsort(np.argsort(utility_scores)[::-1]) + 1