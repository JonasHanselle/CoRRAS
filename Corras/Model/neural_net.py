import numpy as np
np.random.seed(15)
import pandas as pd
import tensorflow_core as tf
from tensorflow_core import keras
from tensorflow_core.python.keras import layers
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.data import Dataset
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
import logging


class NeuralNetwork:
    def __init__(self):
        self.network = None
        self.logger = logging.getLogger("CorrasNeuralNet")
        K.set_floatx("float64")


    # def custom_optimizer(self, learning_rate=0.01):

        # if self.network is None:
        #     self.logger.error("No model build so far!")
        # predictions = self.network.outputs()
        # print(predictions)
        # optimizer = Adam(lr=learning_rate)
        # loss_val = 0
        # updates = optimizer.get_updates(self.network.trainable_weights, [], loss_val)
        # train = K.function([self.network.input, *predictions])
        # return train


    def build_network(self, num_labels, num_features ):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = keras.layers.Dense(8, activation="relu")(input_layer)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        output_layers = []
        for i in range(0, num_labels):
            output_layers.append(keras.layers.Dense(1, name="output_layer"+str(i))(hidden_layers))
        return keras.Model(inputs=input_layer, outputs=output_layers)

    def fit(self, num_labels: int, rankings: np.ndarray, features: np.ndarray, performances : np.ndarray, lambda_value = 0.5, regression_loss="Absolute", num_epochs=100, learning_rate=0.1, batch_size=32):
        """Fit the network to the given data.

        Arguments:
            num_labels {int} -- Number of labels in the ranking
            rankings {np.ndarray} -- Ranking of performances
            features {np.ndarray} -- Features
            performances {np.ndarray} -- Performances
            lambda_value {float} -- Lambda
            regression_loss {String} -- Which regression loss
            should be applied, "Squared" and "Absolute" are
            supported
        """
        # add one column for bias
        num_features = features.shape[1]+1
        self.network = self.build_network(num_labels, num_features)

        self.network._make_predict_function()
        self.network.summary()
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # # add constant 1 for bias and create tf dataset
        feature_values = np.hstack((features,np.ones((features.shape[0],1))))
        print(feature_values.shape)
        print(performances.shape)
        dataset = Dataset.from_tensor_slices((feature_values, performances))
        dataset.batch(batch_size)
        print("dataset", dataset)

        print("output", self.network(feature_values[0, None]))
        
        # define custom loss function
        def custom_loss(model, x, y, i):
            y_ = model(x)[i]
            # compute MSE
            # print("x",x)
            # print("y",y)
            # print("y_",y_)

            loss = tf.reduce_mean(tf.square(tf.subtract(y,y_)))
            # print("loss", loss)
            return loss
        
        # l = custom_loss(self.network, feature_values, performances, i)

        # define gradient of custom loss function
        def grad(model, inputs, targets, i):
            with tf.GradientTape() as tape:
                loss_value = custom_loss(model, inputs, targets, i)
            return loss_value, tape.gradient(loss_value, model.trainable_weights)
        
        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        train_loss_results = []

        for epoch in range(num_epochs):
            # for x, y in zip(feature_values, performances):
            for x,y in dataset:
                # print(x,y)
                tvs = self.network.trainable_weights
                accum_tvs = [tf.Variable(tf.zeros_like(tv.initialized_value()),trainable=False) for tv in tvs]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvs]
                # TODO finish gradient accumulation
                cumulative_grads = None
                for i in range(num_labels):
                    loss_value, grads = grad(self.network,feature_values,performances, i)
                    print("loss", loss_value)

                # print(loss_value)
                optimizer.apply_gradients(zip(grads, self.network.trainable_weights))

    
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
        # features = np.expand_dims(features, axis=0)
         # compute utility scores
        # utility_scores = np.exp(self.network(features[:,None].T))
        # return np.reciprocal(utility_scores)
        return self.network(features[:,None].T)

    def predict_ranking(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        utility_scores = np.exp(self.network(features[:,None].T))
        return np.argsort(np.argsort(utility_scores)[::-1]) + 1