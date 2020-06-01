import autograd.numpy as np
from autograd import grad, hessian
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.optimize import minimize

class LogLinearModel:
    def __init__(self,
                 use_exp_for_regression=False,
                 use_reciprocal_for_regression=False):
        self.weights = None
        self.loss_history = []
        self.use_exp_for_regression = use_exp_for_regression
        self.use_reciprocal_for_regression = use_reciprocal_for_regression

    def squared_error(self, performances: np.ndarray, features: np.ndarray,
                      labels: np.ndarray, weights: np.ndarray,
                      sample_weights: np.ndarray):
        """Compute squared error for regression

        Arguments:
            performances {np.ndarray} -- [description]
            features {np.ndarray} -- [description]
            weights {np.ndarray} -- [description]
            sample_weights {np.ndarray} -- weights of the individual samples

        Returns:
            [type] -- [description]
        """
        loss = 0
        # add one column for bias
        feature_values = np.hstack((features, np.ones((features.shape[0], 1))))
        utilities = None
        if self.use_exp_for_regression:
            utilities = np.exp(np.dot(weights, feature_values.T))
        else:
            utilities = np.dot(weights, feature_values.T)
        inverse_utilities = utilities
        if self.use_reciprocal_for_regression:
            inverse_utilities = np.reciprocal(utilities)
        indices = labels.T - 1
        loss += np.mean(sample_weights[:, np.newaxis] * np.square(
            np.subtract(performances,
                        inverse_utilities.T[np.arange(len(labels)),
                                            indices].T)))
        return loss

    def negative_log_likelihood(self, rankings: pd.DataFrame,
                                features: pd.DataFrame, weights: np.ndarray):
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

    def vectorized_nll(self, rankings, features, weights, sample_weights):
        """Compute NLL w.r.t. the data in the given batch and the given weights

        Arguments:
            rankings {np.ndarray} -- Data sample for computing the NNL
            features {np.ndarray} -- Feature values for computing the NNL
            weights {np.ndarray} -- Weight vector, i.e. model parameters

        Returns:
            [float64] -- Negative log-likelihood
        """
        nll = 0
        # one column for bias
        features = np.hstack((features, np.ones((features.shape[0], 1))))
        ordered_weights = weights[rankings - 1]
        rows = []
        for i in range(0, ordered_weights.shape[0]):
            result = np.dot(ordered_weights[i], features[i])
            rows.append(result)
        weighted_features = np.vstack(rows)
        # sum1 = np.sum(
        #     np.sum(sample_weights[:, np.newaxis] * weighted_features, 0))
        sum1 = np.sum(np.sum(weighted_features, 0))
        utilities = np.exp(weighted_features)
        new_logs = []
        for m in range(0, rankings.shape[1]):
            new_logs.append(np.log(np.sum(utilities[:, m:], axis=1)))
        new_logs = np.asarray(new_logs)
        # outer_nll = np.sum(np.sum(
        #     sample_weights[:, np.newaxis] * new_logs.T)) - sum1
        outer_nll = np.sum(np.sum(new_logs)) - sum1
        outer_nll = outer_nll / rankings.shape[0]
        # print(sum1/rankings.shape[0])
        return outer_nll

    def fit_np(self,
               num_labels,
               rankings,
               features,
               performances,
               sample_weights=None,
               lambda_value=0.5,
               regression_loss="Squared",
               maxiter=1000,
               print_output=False,
               log_losses=True,
               reg_param=0.0):
        """[summary]

        Arguments:
            rankings {np.ndarray} -- [Rankings]
            features {np.ndarray} -- [Feature Data]
            performances {np.ndarray} -- [Performance Data]
            lambda_value {float} -- Lambda 


        Returns:
            [type] -- [description]
        """
        # num_labels = rankings.shape[1]
        # add one column for bias
        if sample_weights is None:
            sample_weights = np.ones(rankings.shape[0])
        num_features = features.shape[1] + 1
        self.weights = np.ones(
            (num_labels, num_features)) / (num_features * num_labels)
        nll = self.vectorized_nll
        reg_loss = self.squared_error
        self.loss_history = []

        # minimize loss function

        def callback(x):
            x = np.reshape(x, (num_labels, num_features))
            se_value = reg_loss(performances, features, rankings, x,
                                sample_weights)
            nll_value = nll(rankings, features, x, sample_weights)
            self.loss_history.append([se_value, nll_value])

        def g(x):
            x = np.reshape(x, (num_labels, num_features))
            if lambda_value == 0:
                reg_loss_value = (1 - lambda_value) * reg_loss(
                    performances, features, rankings, x,
                    sample_weights) + reg_param * np.sum(x**2)
                return reg_loss(performances, features, rankings, x,
                                sample_weights)
            elif lambda_value == 1:
                nll_value = lambda_value * nll(
                    rankings, features, x,
                    sample_weights) + reg_param * np.sum(x**2)
                return nll(rankings, features, x, sample_weights)
            nll_value = lambda_value * nll(rankings, features, x,
                                           sample_weights)
            reg_loss_value = (1 - lambda_value) * reg_loss(
                performances, features, rankings, x, sample_weights)
            return nll_value + reg_loss_value + reg_param * np.sum(x**2)

        jac = grad(g)

        cb = None
        if log_losses:
            cb = callback

        flat_weights = self.weights.flatten()
        result = minimize(g,
                          flat_weights,
                          method="L-BFGS-B",
                          jac=jac,
                          callback=cb,
                          options={
                              "maxiter": maxiter,
                              "disp": print_output
                          })

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
        utility_scores = None
        if self.use_exp_for_regression:
            utility_scores = np.exp(np.dot(self.weights, features))
        else:
            utility_scores = np.dot(self.weights, features)
        if self.use_reciprocal_for_regression:
            return np.reciprocal(utility_scores)
        return utility_scores

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
        return np.argsort(np.argsort(utility_scores)) + 1

    def save_loss_history(self, filepath: str):
        """Saves the history of losses after the model has been fit
        
        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.loss_history,
                             index=None,
                             columns=["MSE", "NLL"])
        frame.to_csv(path_or_buf=filepath, index_label="iter")

    def get_loss_history_frame(self):
        """Saves the history of losses after the model has been fit
        
        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.loss_history,
                             index=None,
                             columns=["MSE", "NLL"])
        frame.insert(0, "iteration", range(0, len(frame)))
        return frame