import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from scipy.optimize import minimize


class RegressionSquaredError:

    def __init__(self):
        pass

    def squared_error(self, performances : pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
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
            # TODO change to inverse for use in combined regression and ranking
            # print("utilities", utilities)
            # print("performances", current_performances)
            loss += np.sum(np.square(np.subtract(current_performances,inverse_utilities)))
        return loss
        
    def first_derivative(self, rankings: pd.DataFrame, inverse_rankings : pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """[summary]
        
        Arguments:
            rankings {pd.DataFrame} -- [description]
            inverse_rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]
        """
        pass

class PLNegativeLogLikelihood:

    def __init__(self):
        pass

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
        for index, ranking in rankings.iterrows():
            # print("index", index)
            # add one column for bias
            feature_values = np.hstack((features.loc[index].values, [1]))
            # print("features", feature_values)
            # print("feature values",  feature_values)
            inner_sum = 0
            for m in range(0, len(ranking)):
                remaining_sum = 0
                for j in range(m, len(ranking)):
                    # compute utility of remaining labels
                    remaining_sum += np.exp(
                        np.dot(weights[ranking[j]-1], feature_values))
                inner_sum += np.log(
                    np.exp(np.dot(weights[ranking[m]-1], feature_values))) - np.log(remaining_sum)
            outer_sum += inner_sum
        return -outer_sum

    def first_derivative(self, rankings: pd.DataFrame, inverse_rankings: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Computes the the gradient vectors of the nll 

        Arguments:
            rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]

        Returns:
            [type] -- [description]
        """
        gradient = np.zeros_like(weights)
        for a in range(0, gradient.shape[0]):
            for index, ranking in rankings.iterrows():
                current_features = np.hstack((features.loc[index].values, [1]))
                # print("ranking",ranking.values)
                # print("inverse ranking", inverse_rankings.loc[index].values,"\n") 
                if inverse_rankings.loc[index].values[a] >= 1:
                    gradient[a] += current_features
            for index, ranking in rankings.iterrows():
                current_features = np.hstack((features.loc[index].values, [1]))
                for m in range(0, len(ranking)):
                    if inverse_rankings.loc[index].values[a] >= m:
                        denominator = 0
                        for j in range(m, len(ranking)):
                            denominator += np.exp(
                                np.dot(weights[ranking[j]-1], current_features))
                        numerator = np.exp(
                            np.dot(weights[a], current_features)) * current_features
                        fraction = numerator / denominator
                        # print("numerator", numerator)
                        # print("denominator", denominator)
                        # print("fraction", fraction, "\n\n")
                        gradient[a] -= fraction
        return -gradient

    def second_derivative(self, rankings: pd.DataFrame, inverse_rankings: pd.DataFrame, features: pd.DataFrame, weights: np.ndarray):
        """Computes the the gradient vector of the nll 

        Arguments:
            rankings {pd.DataFrame} -- [description]
            features {pd.DataFrame} -- [description]
            weights {np.ndarray} -- [description]

        Returns:
            [type] -- [description]
        """
        hessian = np.zeros(shape=(weights.shape[1], weights.shape[1]))
        return hessian


class LogLinearModel:

    def __init__(self):
        self.weights = None
        self.dataset = None

    # def fit(self, dataset: ASRankingScenario):
    #     """Fits a label ranking model based on a log-linear utility function.

    #     Arguments:
    #         data {ASRankingScenario} -- Training data set
    #     """
    #     self.dataset = dataset
    #     num_labels = len(dataset.algorithms)
    #     num_features = len(dataset.features)
    #     self.weights = np.zeros(shape=(num_labels, num_features))
    #     nll = PLNegativeLogLikelihood()

    #     # minimize nnl
    #     def f(x):
    #         x = np.reshape(x,(num_labels, num_features))
    #         return nll.negative_log_likelihood(dataset.performance_rankings,dataset.feature_data, x)

    #     def f_prime(x):
    #         x = np.reshape(x,(num_labels, num_features))
    #         return nll.first_derivative(dataset.performance_rankings, dataset.performance_rankings_inverse,dataset.feature_data,x).flatten()

    #     flat_weights = self.weights.flatten()
    #     print(flat_weights.shape)
    #     result = minimize(f, flat_weights, method="L-BFGS-B", jac=None, options={"maxiter" : 10, "disp" : True})
    #     print("Result", result)

    def fit(self, rankings: pd.DataFrame, inverse_rankings: pd.DataFrame, features: pd.DataFrame, performances : pd.DataFrame, lambda_value = 0.5):
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

        self.weights = np.zeros(shape=(num_labels, num_features))
        nll = PLNegativeLogLikelihood()
        se = RegressionSquaredError()

        # minimize nnl
        def f(x):
            x = np.reshape(x, (num_labels, num_features))
            if lambda_value == 0:
                return se.squared_error(performances,features,x)
            elif lambda_value == 1:
                return lambda_value * nll.negative_log_likelihood(rankings, features, x)   
            return lambda_value * nll.negative_log_likelihood(rankings, features, x) + (1 - lambda_value) * se.squared_error(performances,features,x)

        flat_weights = self.weights.flatten()
        print(flat_weights.shape)
        result = minimize(f, flat_weights, method="L-BFGS-B",
                          jac=None, options={"maxiter": 100, "disp": True})
        print("Result", result)
        self.weights = np.reshape(result.x, (num_labels, num_features))
        print("Weights", self.weights)

    def predict(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        utility_scores = np.exp(np.dot(self.weights, features))
        ranking = np.argsort(np.argsort(utility_scores))+1
        ranking = ranking[::-1]
        return ranking

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
