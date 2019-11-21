import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario


class NeuralNetwork:
    def __init__(self):
        self.network = None

    def optimize(self, rankings: pd.DataFrame, inverse_rankings: pd.DataFrame, features: pd.DataFrame, performances : pd.DataFrame, lambda_value = 0.5, regression_loss="Absolute"):
        outputs = self.network.output
        print(outputs)


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
        # self.network = keras.Functional([layers.Dense(8, activation="relu", input_shape=[num_features]), layers.Dense(8, activation="relu", input_shape=[num_features]), layers.Dense(4,activation="linear"),layers.Dense(3)])
        
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = keras.layers.Dense(8, activation="relu")(input_layer)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        output_layers = []
        for i in range(0, num_labels):
            output_layers.append(keras.layers.Dense(1, name="output_layer"+str(i))(hidden_layers))

        self.network = keras.Model(inputs=input_layer, outputs=output_layers)
        optimizer = tf.keras.optimizers.Adam()

        def reg_squared_error(y_true, y_pred):
            return tf.reduce_mean(tf.square(tf.subtract(y_true,tf.exp(y_pred))))

        def reg_absolute_error(y_true, y_pred):
            return tf.reduce_mean(tf.abs(tf.subtract(y_true,tf.exp(y_pred))))

        self.network.compile(loss=[reg_squared_error, reg_squared_error, reg_squared_error], optimizer=optimizer, metrics=["mse", "mae"])

        self.network._make_predict_function()

        self.network.summary()

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # # add constant 1 for bias
        feature_values = np.hstack((features.values,np.ones((features.shape[0],1))))
        self.optimize(rankings,inverse_rankings,features,performances)
        self.network.fit(feature_values, [performances.values[:,0],performances.values[:,1],performances.values[:,2]], epochs = 10000, validation_split = 0.2, verbose = 0, callbacks=[early_stop])
        

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
        utility_scores = np.exp(self.network.predict(features))
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