import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario


class NeuralNetwork:
    def __init__(self):
        self.network = None

    def fit(self, rankings: pd.DataFrame, inverse_rankings: pd.DataFrame, features: pd.DataFrame, performances : pd.DataFrame, lambda_value = 0.5, regression_loss="Absolute"):
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
        self.network = keras.Sequential([layers.Dense(4, activation="sigmoid", input_shape=[num_features]), layers.Dense(4,activation="linear"),layers.Dense(3)])

        optimizer = tf.keras.optimizers.Adam()

        self.network.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # add constant 1 for bias
        feature_values = np.hstack((features.values,np.ones((features.shape[0],1))))
        self.network.fit(feature_values, performances.values, epochs = 1000, validation_split = 0.2, verbose = 0, callbacks=[early_stop])
        

    def predict_performances(self, features: np.ndarray):
        """Predict a vector of performance values.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            np.ndarray -- Estimation of performance values
        """
        # add constant 1 for bias
        features = np.hstack((features, [1]))
        # keras expects a 2 dimensional input
        features = np.expand_dims(features, axis=0)
         # compute utility scores
        utility_scores = self.network.predict(features)
        # return np.reciprocal(utility_scores)
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
        return np.argsort(np.argsort(utility_scores)[::-1]) + 1