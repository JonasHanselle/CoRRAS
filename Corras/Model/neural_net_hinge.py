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


class NeuralNetworkSquaredHinge:
    def __init__(self):
        self.network = None
        self.logger = logging.getLogger("CorrasNeuralNet")
        K.set_floatx("float64")

    def build_network(self, num_labels, num_features):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = keras.layers.Dense(8, activation="relu")(input_layer)
        hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        # hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        output_layer = keras.layers.Dense(
            num_labels, activation="linear", name="output_layer")(hidden_layers)
        return keras.Model(inputs=input_layer, outputs=output_layer)

    def fit(self, num_labels: int, rankings: np.ndarray, features: np.ndarray, performances: np.ndarray, lambda_value=0.5, epsilon_value=1, regression_loss="Absolute", num_epochs=1000, learning_rate=0.1, batch_size=32, seed=1, patience=16, es_val_ratio=0.3, reshuffle_buffer_size=1000, early_stop_interval=5):
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
        # print(val_data)
        print("train data", train_data)
        train_data = train_data.batch(batch_size)
        # val_data = val_data.batch(1)
        # define custom loss function

        def custom_loss(model, x, y_perf, y_rank):
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
            # print("x", x)
            # print("y_perf", y_perf)
            # print("y_rank", y_rank)
            output = model(x)
            row_indices = tf.range(tf.shape(y_rank)[0])
            y_ind = y_rank - 1 
            added_indices_0 = tf.stack([row_indices,y_ind[:,0]], axis=1)
            added_indices_1 = tf.stack([row_indices,y_ind[:,1]], axis=1)
            # print(added_indices_0.shape, added_indices_1.shape, x.shape, y_perf.shape, y_rank.shape, output.shape, row_indices.shape)
            y_hat_0 = tf.gather_nd(output, added_indices_0) 
            y_hat_1 = tf.gather_nd(output, added_indices_1)
            reg_loss = tf.square(tf.subtract(y_hat_0, y_perf[:,0])) + tf.square(tf.subtract(y_hat_1, y_perf[:,1]))
            rank_loss = 0
            return lambda_value * rank_loss + (1 - lambda_value) * reg_loss
        # define gradient of custom loss function

        def grad(model, x, y_perf, y_rank):
            with tf.GradientTape() as tape:
                loss_value = custom_loss(model, x, y_perf, y_rank)
            return loss_value, tape.gradient(loss_value, model.trainable_weights)

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        best_val_loss = float("inf")
        current_best_model = keras.models.clone_model(self.network)
        patience_cnt = 0

        for epoch in range(num_epochs):

            for x, y_perf, y_rank in train_data:
                # tvs = self.network.trainable_weights
                # accum_tvs = [tf.Variable(tf.zeros_like(
                #     tv.initialized_value()), trainable=False) for tv in tvs]
                # zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvs]

                loss_value, grads = grad(self.network, x, y_perf, y_rank)
                # for j in range(len(accum_tvs)):
                #     accum_tvs[j].assign_add(grads[j])

                # print(loss_value)
                optimizer.apply_gradients(
                    zip(grads, self.network.trainable_weights))
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
        predictions = self.network(features[:,None].T)
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
        # features = np.expand_dims(features, axis=0)
        predictions = self.network(features[:,None].T)
        return np.argsort(np.argsort(predictions)) + 1