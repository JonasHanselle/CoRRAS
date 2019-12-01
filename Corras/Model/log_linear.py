import autograd.numpy as np
from autograd import grad, hessian
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.optimize import minimize


class LogLinearModel:

    def __init__(self):
        self.weights = None

    def squared_error(self, performances: np.ndarray, features: np.ndarray, weights: np.ndarray):
        """Compute squared error for regression

        Arguments:
            performances {np.ndarray} -- [description]
            features {np.ndarray} -- [description]
            weights {np.ndarray} -- [description]

        Returns:
            [type] -- [description]
        """
        loss = 0
        feature_values = np.hstack((features, np.ones((features.shape[0], 1))))
        utilities = np.exp(np.dot(weights, feature_values.T))
        # utilities = np.dot(weights, feature_values.T)
        inverse_utilities = np.reciprocal(utilities)
        # inverse_utilities = utilities
        loss += np.mean(np.square(np.subtract(performances.T,
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
        for i, (index, ranking) in enumerate(rankings.iterrows()):
            # add one column for bias
            feature_values = np.hstack((features.loc[index].values, [1]))
            inner_sum = 0
            ranking = np.argsort(ranking.values)
            for m in range(0, len(ranking)):
                remaining_sum = 0
                for j in range(m, len(ranking)):
                    # compute utility of remaining labels
                    remaining_sum += np.exp(
                        np.dot(weights[ranking[j]], feature_values))
                inner_sum += np.log(remaining_sum) - \
                    np.dot(weights[ranking[m]], feature_values)
            outer_sum += inner_sum
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
        features = np.hstack((features, np.ones((features.shape[0], 1))))
        ordered_weights = weights[rankings-1]
        rows = []
        for i in range(0, ordered_weights.shape[0]):
            result = np.dot(ordered_weights[i], features[i])
            rows.append(result)
        weighted_features = np.vstack(rows)
        sum1 = np.sum(np.sum(weighted_features, 0))
        utilities = np.exp(weighted_features)
        new_logs = []
        for m in range(0, rankings.shape[1]):
            new_logs.append(np.log(np.sum(utilities[:, m:], axis=1)))
        new_logs = np.asarray(new_logs)
        outer_nll = np.sum(np.sum(new_logs)) - sum1
        return outer_nll


    def list_nll(self, rankings, features, weights):
        """Compute NLL w.r.t. the data in the given batch and the given weights
        This version is assuming rankings are given as a list of python lists,
        allowing for rankings of different lengths.

        Arguments:
            rankings {np.ndarray} -- Data sample for computing the NNL
            features {np.ndarray} -- Feature values for computing the NNL
            weights {np.ndarray} -- Weight vector, i.e. model parameters

        Returns:
            [float64] -- Negative log-likelihood
        """
        nll = 0
        g1 = 0
        g2 = 0
        features = np.hstack((features, np.ones((features.shape[0], 1))))
        for index, ranking in enumerate(rankings):
            sum1 = 0
            sum2 = 0
            for m in range(0, len(ranking)):
                sum1 += np.sum(np.dot(weights[ranking[m]-1], features[index]))
                summand = 0
                for j in range(m, len(ranking)):
                    summand += np.exp(np.dot(weights[ranking[j]-1], features[index]))
                sum2 += np.log(summand)
                nll += sum2 - sum1
            g1 += sum1
            g2 += sum2
        return g2-g1

    def fit_np(self, rankings, features, performances, lambda_value=0.5, regression_loss="Squared", maxiter=1000):
        """[summary]

        Arguments:
            rankings {np.ndarray} -- [Rankings]
            features {np.ndarray} -- [Feature Data]
            performances {np.ndarray} -- [Performance Data]
            lambda_value {float} -- Lambda 


        Returns:
            [type] -- [description]
        """
        num_labels = rankings.shape[1]
        # add one column for bias
        num_features = features.shape[1]+1
        # self.weights = np.random.rand(num_labels, num_features)
        self.weights = np.ones((num_labels, num_features)) / (num_features * num_labels)
        nll = self.vectorized_nll
        reg_loss = self.squared_error

        # minimize loss function
        def g(x):
            x = np.reshape(x, (num_labels, num_features))
            if lambda_value == 0:
                return reg_loss(performances, features, x)
            elif lambda_value == 1:
                return nll(rankings, features, x)
            return lambda_value * nll(rankings, features, x) + (1 - lambda_value) * reg_loss(performances, features, x)

        jac = grad(g)

        flat_weights = self.weights.flatten()
        result = minimize(g, flat_weights, method="L-BFGS-B",
                          jac=jac, options={"maxiter": maxiter, "disp": True})

        self.weights = np.reshape(result.x, (num_labels, num_features))

    def fit_list(self, num_labels, rankings : list, features, performances, lambda_value=0.5, regression_loss="Squared", maxiter=1000):
        """[summary]

        Arguments:
            num_labels {int} -- [Total number of labels]
            rankings {np.ndarray} -- [Rankings]
            features {np.ndarray} -- [Feature Data]
            performances {np.ndarray} -- [Performance Data]
            lambda_value {float} -- Lambda 


        Returns:
            [type] -- [description]
        """
        # add one column for bias
        num_features = features.shape[1]+1
        # self.weights = np.random.rand(num_labels, num_features)
        self.weights = np.ones((num_labels, num_features)) / (num_features * num_labels)
        nll = self.list_nll
        reg_loss = self.squared_error

        # minimize loss function
        def g(x):
            x = np.reshape(x, (num_labels, num_features))
            if lambda_value == 0:
                return reg_loss(performances, features, x)
            elif lambda_value == 1:
                return nll(rankings, features, x)
            return lambda_value * nll(rankings, features, x) + (1 - lambda_value) * reg_loss(performances, features, x)

        jac = grad(g)

        flat_weights = self.weights.flatten()
        result = minimize(g, flat_weights, method="L-BFGS-B",
                          jac=jac, options={"maxiter": maxiter, "disp": True})

        self.weights = np.reshape(result.x, (num_labels, num_features))

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
        # utility_scores = np.dot(self.weights, features)
        # return utility_scores
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
