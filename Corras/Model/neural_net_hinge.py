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


class NeuralNetworkSquaredHinge:
    def __init__(self):
        self.network = None
        self.logger = logging.getLogger("CorrasNeuralNet")
        self.loss_history = []
        self.es_val_history = []
        K.set_floatx("float64")

    def build_network(self,
                      num_labels,
                      num_features,
                      hidden_layer_sizes=None,
                      activation_function="relu"):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = input_layer
        if hidden_layer_sizes is None:
            hidden_layers = keras.layers.Dense(
                num_features, activation=activation_function)(hidden_layers)
            hidden_layers = keras.layers.Dense(
                num_features, activation=activation_function)(hidden_layers)
        else:
            for layer_size in hidden_layer_sizes:
                hidden_layers = keras.layers.Dense(
                    layer_size,
                    activation=activation_function)(hidden_layers)

        # hidden_layers = keras.layers.Dense(8, activation="relu")(hidden_layers)
        output_layer = keras.layers.Dense(num_labels,
                                          activation="linear",
                                          name="output_layer")(hidden_layers)
        return keras.Model(inputs=input_layer, outputs=output_layer)

    def fit(self,
            num_labels: int,
            rankings: np.ndarray,
            features: np.ndarray,
            performances: np.ndarray,
            sample_weights: np.ndarray,
            lambda_value=0.5,
            epsilon_value=1,
            regression_loss="Absolute",
            num_epochs=1000,
            learning_rate=0.001,
            batch_size=32,
            seed=1,
            patience=16,
            es_val_ratio=0.3,
            reshuffle_buffer_size=1000,
            early_stop_interval=5,
            log_losses=True,
            hidden_layer_sizes=None,
            activation_function="relu"):
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
        num_features = features.shape[1] + 1
        self.network = self.build_network(
            num_labels,
            num_features,
            hidden_layer_sizes=hidden_layer_sizes,
            activation_function=activation_function)

        self.network._make_predict_function()
        self.network.summary()

        self.loss_history = []
        self.es_val_history = []
        # add constant 1 for bias and create tf dataset
        feature_values = np.hstack((features, np.ones((features.shape[0], 1))))
        # print(feature_values.shape)
        # print(performances.shape)

        # split feature and performance data
        feature_values, performances, rankings, sample_weights = shuffle(feature_values,
                                                                         performances,
                                                                         rankings, sample_weights,
                                                                         random_state=seed)
        val_data = Dataset.from_tensor_slices(
            (feature_values[:int(es_val_ratio * feature_values.shape[0])],
             performances[:int(es_val_ratio * performances.shape[0])],
             rankings[:int(es_val_ratio * rankings.shape[0])],
             sample_weights[:int(es_val_ratio * sample_weights.shape[0])]))
        train_data = Dataset.from_tensor_slices(
            (feature_values[int(es_val_ratio * feature_values.shape[0]):],
             performances[int(es_val_ratio * performances.shape[0]):],
             rankings[int(es_val_ratio * rankings.shape[0]):],
             sample_weights[int(es_val_ratio * sample_weights.shape[0]):]))
        # print(val_data)
        # print("train data", train_data)
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(1)

        # define custom loss function

        def custom_loss(model, x, y_perf, y_rank, sample_weight):
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
            row_indices = tf.range(tf.shape(y_rank)[0])
            y_ind = y_rank - 1
            added_indices_0 = tf.stack([row_indices, y_ind[:, 0]], axis=1)
            added_indices_1 = tf.stack([row_indices, y_ind[:, 1]], axis=1)
            y_hat_0 = tf.gather_nd(output, added_indices_0)
            y_hat_1 = tf.gather_nd(output, added_indices_1)
            reg_loss = tf.reduce_mean(tf.multiply(sample_weight,
                                                  (tf.square(tf.subtract(y_hat_0, y_perf[:, 0])))))
            reg_loss += tf.reduce_mean(
                (tf.square(tf.subtract(y_hat_1, y_perf[:, 1]))))
            rank_loss = tf.reduce_mean(tf.multiply(sample_weight,
                                                   tf.square(tf.maximum(0, epsilon_value - (y_hat_1 - y_hat_0)))))
            return lambda_value * reg_loss + (
                1 - lambda_value) * rank_loss, reg_loss, rank_loss

        # define gradient of custom loss function

        def grad(model, x, y_perf, y_rank, sample_weight):
            with tf.GradientTape() as tape:
                loss_value, reg_loss, rank_loss = custom_loss(
                    model, x, y_perf, y_rank, sample_weight)
            return loss_value, tape.gradient(
                loss_value, model.trainable_weights), reg_loss, rank_loss

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        best_val_loss = float("inf")
        current_best_weights = self.network.get_weights()
        patience_cnt = 0

        for epoch in range(num_epochs):
            epoch_reg_loss_avg = tf.keras.metrics.Mean()
            epoch_rank_loss_avg = tf.keras.metrics.Mean()
            for x, y_perf, y_rank, sample_weight in train_data:
                loss_value, grads, reg_loss, rank_loss = grad(
                    self.network, x, y_perf, y_rank, sample_weight)
                optimizer.apply_gradients(
                    zip(grads, self.network.trainable_weights))
                epoch_reg_loss_avg(reg_loss)
                epoch_rank_loss_avg(rank_loss)
            if log_losses:
                self.loss_history.append([
                    float(epoch_reg_loss_avg.result()),
                    float(epoch_rank_loss_avg.result())
                ])

            if epoch % early_stop_interval == 0:
                losses = []
                for x, y_perf, y_rank, sample_weight in val_data:
                    losses.append(custom_loss(self.network, x, y_perf, y_rank, sample_weight))
                loss_tensor = np.average(losses)
                current_val_loss = tf.reduce_mean(loss_tensor)
                self.es_val_history.append(current_val_loss)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    current_best_weights = self.network.get_weights()
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt >= patience:
                    print("early stopping")
                    break
        self.network.set_weights(current_best_weights)

    def predict_performances(self, features: np.ndarray):
        """Predict a vector of performance values.

        Arguments:
            features {np.ndarray} -- Instance feature values

        Returns:
            np.ndarray -- Estimation of performance values
        """
        # add constant 1 for bias
        features = np.hstack((features, [1]))
        predictions = self.network(features[:, None].T)

        return predictions.numpy()[0]

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
        predictions = self.network(features[:, None].T)
        return np.argsort(np.argsort(predictions)) + 1

    def save_loss_history(self, filepath: str):
        """Saves the history of losses after the model has been fit

        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.loss_history,
                             index=None,
                             columns=["MSE", "SQH"])
        frame.to_csv(path_or_buf=filepath, index_label="iter")

    def get_loss_history_frame(self):
        """Saves the history of losses after the model has been fit

        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.loss_history,
                             index=None,
                             columns=["MSE", "SQH"])
        frame.insert(0, "epoch", range(0, len(frame)))
        return frame

    def save_es_val_history(self, filepath: str):
        """Saves the history of losses after the model has been fit

        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.es_val_history,
                             index=None,
                             columns=["ES_VAL_LOSS"])
        frame.to_csv(path_or_buf=filepath, index_label="es_call")

    def get_es_val_history_frame(self):
        """Saves the history of losses after the model has been fit

        Arguments:
            filepath {str} -- Path of the csv file
        """
        frame = pd.DataFrame(data=self.es_val_history,
                             index=None,
                             columns=["ES_VAL_LOSS"])
        frame.insert(0, "es_call", range(0, len(frame)))
        return frame
