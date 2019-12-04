import logging
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from sklearn.utils import shuffle
from tensorflow_core.python.data import Dataset
from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras import layers
from tensorflow_core import keras
import tensorflow_core as tf
import pandas as pd
import numpy as np
np.random.seed(15)


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

    def build_network(self, num_labels, num_features):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = keras.layers.Dense(8, activation="relu")(input_layer)
        # hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        # hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        # output_layers = []
        # for i in range(0, num_labels):
        #     output_layers.append(keras.layers.Dense(
        #         1, activation="linear", name="output_layer"+str(i))(hidden_layers))
        output_layer = keras.layers.Dense(
            num_labels, activation="linear", name="output_layer")(hidden_layers)
        return keras.Model(inputs=input_layer, outputs=output_layer)

    def fit(self, num_labels: int, rankings: np.ndarray, features: np.ndarray, performances: np.ndarray, lambda_value=0, regression_loss="Absolute", num_epochs=1000, learning_rate=0.1, batch_size=32, seed=1, patience=16, es_val_ratio=0.3, reshuffle_buffer_size=1000, early_stop_interval=5):
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
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10)

        # add constant 1 for bias and create tf dataset
        feature_values = np.hstack((features, np.ones((features.shape[0], 1))))
        print(feature_values.shape)
        # print(performances.shape)

        # split feature and performance data
        feature_values, performances = shuffle(
            feature_values, performances, random_state=seed)
        val_data = Dataset.from_tensor_slices((feature_values[: int(
            es_val_ratio * feature_values.shape[0])], performances[: int(es_val_ratio * performances.shape[0])], rankings[: int(es_val_ratio * rankings.shape[0])]))
        train_data = Dataset.from_tensor_slices((feature_values[int(
            es_val_ratio * feature_values.shape[0]):], performances[int(es_val_ratio * performances.shape[0]):], rankings[int(es_val_ratio * rankings.shape[0]):]))
        print(val_data)
        print(train_data)
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(1)
        # define custom loss function

        def custom_loss(model, x, y_perf, y_rank, i):
            """Compute loss for i-th label
            
            Arguments:
                model {[type]} -- [Neural network]
                x {[type]} -- [Feature vector]
                y_perf {[type]} -- [Performances]
                y_rank {[type]} -- [Rankings]
                i {[type]} -- [Label]
            
            Returns:
                [float64] -- [Loss]
            """
            output = model(x)
            # compute MSE
            reg_loss = tf.reduce_mean(tf.square(tf.subtract(output[:,i], y_perf[:,i])))
            exp_utils = np.exp(output)
            print(exp_utils)
            exp_utils_ordered = exp_utils[np.arange(exp_utils.shape[0])[:,np.newaxis],y_rank-1]
            inv_rank = np.argsort(y_rank)
            rank_loss = 0
            for k in range(num_labels):
                indicator = inv_rank[:,i] >= k
                # exp_utils_indicator = exp_utils[indicator]
                indicator = np.repeat(indicator[:,None], num_labels, axis=1)
                # if inv_rank[i] >= k:
                # if indicator.any():
                exp_utils_indicator = np.where(indicator, exp_utils, np.zeros_like(exp_utils))
                # print(indicator)
                # print("exp utils", exp_utils)
                # print("exp utils ind", exp_utils_indicator)
                # print("numerator" + str(k), exp_utils_indicator[:,i])
                rank_loss += np.divide(exp_utils_indicator[:,i], np.sum(exp_utils_ordered[:,k:], axis=1))
                # print("exp ut ind", exp_utils_indicator[:,i])
                # print("rank_loss " + str(k), rank_loss)
            if i < (num_labels - 1):
                rank_loss -= 1
            rank_loss = tf.reduce_sum(rank_loss)
            # print("reg_loss", reg_loss)
            # print("rank_loss", rank_loss)
            return lambda_value * rank_loss + (1 - lambda_value) * reg_loss
        # define gradient of custom loss function

        def grad(model, x, y_perf, y_rank, i):
            with tf.GradientTape() as tape:
                loss_value = custom_loss(model, x, y_perf, y_rank, i)
            return loss_value, tape.gradient(loss_value, model.trainable_weights)

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        best_val_loss = float("inf")
        current_best_model = keras.models.clone_model(self.network)
        patience_cnt = 0

        for epoch in range(num_epochs):

            for x, y_perf, y_rank in train_data:
                tvs = self.network.trainable_weights
                accum_tvs = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in tvs]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvs]

                for i in range(num_labels):
                    loss_value, grads = grad(self.network, x, y_perf, y_rank, i)
                    for j in range(len(accum_tvs)):
                        accum_tvs[j].assign_add(grads[j])

                # print(loss_value)
                optimizer.apply_gradients(
                    zip(accum_tvs, self.network.trainable_weights))
            # if epoch % early_stop_interval == 0:
            #     losses = []
            #     for x, y_perf, y_rank in val_data:
            #         losses.append(custom_loss(self.network, x, y_perf, y_rank))
            #     loss_tensor = np.average(losses)
            #     current_val_loss = tf.reduce_mean(loss_tensor)
            #     if current_val_loss < best_val_loss:
            #         best_val_loss = current_val_loss
            #         current_best_model = keras.models.clone_model(self.network)
            #         print("new best validation loss", best_val_loss)
            #         patience_cnt = 0
            #     else:
            #         patience_cnt += 1
            #     if patience_cnt >= patience:
            #         print("early stopping")
            #         break

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
        return self.network(features[:, None].T)

    def predict_ranking(self, features: np.ndarray):
        """Predict a label ranking.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            pd.DataFrame -- Ranking of algorithms
        """
        # compute utility scores
        features = np.hstack((features, [1]))
        utility_scores = np.exp(self.network(features[:, None].T))
        return np.argsort(np.argsort(utility_scores)[::-1]) + 1