import autograd.numpy as np
from autograd import grad, hessian
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.optimize import minimize

class LinearHingeModel:

    def __init__(self):
        self.weights = None
        self.loss_history = []

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
        utilities = None
        if self.use_exp_for_regression:
            utilities = np.exp(np.dot(weights, feature_values.T))
        else:
            utilities = np.dot(weights, feature_values.T)
        inverse_utilities = utilities
        if self.use_reciprocal_for_regression:
            inverse_utilities = np.reciprocal(utilities)
        loss += np.mean(np.square(np.subtract(performances.T,
                                                inverse_utilities)))
        return loss

    def fit_np(self, num_labels, labels, features, performances, sample_weights=None, lambda_value=0.5, epsilon_value=1, regression_loss="Squared", maxiter=100, print_output=False, log_losses=True, reg_param = 0.0):
        """[summary]

        Arguments:
            features {np.ndarray} -- [Feature Data]
            performances {np.ndarray} -- [Performance Data]
            lambda_value {float} -- Lambda 


        Returns:
            [type] -- [description]
        """
        if sample_weights is None:
            sample_weights = np.ones(features.shape[0])
        # add one column for bias
        feature_values = np.hstack((features, np.ones((features.shape[0], 1))))
        num_features = feature_values.shape[1]
        self.weights = np.ones((num_labels, num_features)) / (num_features * num_labels)
        self.loss_history = []


        def callback(w):
            w = np.reshape(w, (num_labels, num_features))
            squared_error = 0
            hinge_loss = 0
            for cur_labels, cur_features, cur_performances in zip(labels,feature_values,performances):
                y_hats = np.dot(w[[cur_labels-1]],cur_features)
                squared_error = squared_error + np.sum(np.square(y_hats-cur_performances))
                hinge_loss = hinge_loss + max(0, epsilon_value - (y_hats[1] - y_hats[0]))**2
            squared_error = squared_error  / labels.shape[0]
            hinge_loss = hinge_loss / labels.shape[0]
            self.loss_history.append([squared_error, hinge_loss])
            squared_error = lambda_value * squared_error
            hinge_loss = (1 - lambda_value) * hinge_loss
            total_error = squared_error + hinge_loss
 

        # minimize loss function
        def g(w):
            w = np.reshape(w, (num_labels, num_features))
            squared_error = 0
            hinge_loss = 0
            for cur_labels, cur_features, cur_performances, sample_weight in zip(labels,feature_values,performances, sample_weights):
                y_hats = np.dot(w[[cur_labels-1]],cur_features)
                squared_error = squared_error + sample_weight * np.sum(np.square(y_hats-cur_performances))
                hinge_loss = hinge_loss + max(0, epsilon_value - (y_hats[1] - y_hats[0]))**2
                hinge_loss = sample_weight * hinge_loss
            squared_error = squared_error  / labels.shape[0]
            hinge_loss = hinge_loss / labels.shape[0]
            squared_error = lambda_value * squared_error
            hinge_loss = (1 - lambda_value) * hinge_loss
            # TODO check whether whole matrix or only current weight vectors should be taken into account
            total_error = squared_error + hinge_loss + reg_param * w**2
            return np.mean(total_error)

        jac = grad(g)
        cb = None
        if log_losses:
            cb = callback

        flat_weights = self.weights.flatten()
        result = minimize(g, flat_weights, method="L-BFGS-B",
                          jac=jac, callback=cb, options={"maxiter": maxiter, "disp": print_output})

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
        scores = np.dot(self.weights, features)
        return scores

    def predict_ranking(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        scores = np.dot(self.weights, features)
        return np.argsort(np.argsort(scores)) + 1

    def save_loss_history(self, filepath : str):
        """Saves the history of losses after the model has been fit
        
        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.loss_history, index=None, columns=["MEAN_SQUARED_ERROR", "SQUARED_HINGE"])
        frame.to_csv(path_or_buf=filepath, index_label="call")