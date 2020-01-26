import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from Corras.Model import log_linear
from Corras.Model import neural_net
from Corras.Model import neural_net_hinge
from Corras.Util import ranking_util as util
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

title = "glass"
df = pd.read_table(sep="\t",
                   filepath_or_buffer="LabelRankingData/" + title +
                   "_dense.txt")
df = df.iloc[1:]
feature_columns = [x for x in df.columns if x[0] == "A"]
ranking_columns = [x for x in df.columns if x[0] == "L"]
features = df[feature_columns]
rankings = df[ranking_columns]  
print(features)
print(rankings)

kf = KFold(n_splits=2, shuffle=True, random_state=5)
split = next(kf.split(df), None)

# training_portions = np.linspace(start=0, stop=1, num=5)
training_portions = [1]
result_data = []

# for split_num, split in enumerate(kf.split(df)):

for portion in training_portions:

    train_features = features.iloc[split[0]].astype('float64')
    train_rankings = rankings.iloc[split[0]].astype('int32')
    test_features = features.iloc[split[1]].astype('float64')
    test_rankings = rankings.iloc[split[1]].astype('int32')

    train_features = train_features[:int(portion * len(train_features))]
    train_rankings = train_rankings[:int(portion * len(train_rankings))]

    train_features_np = train_features.values
    train_rankings_np = train_rankings.values

    train_rankings_np_rank = util.ordering_to_ranking_matrix(train_rankings_np)

    inst, rank = util.sample_ranking_pairs_with_features_from_rankings(
        train_features_np, train_rankings_np_rank, 10, 15)

    print(train_features_np)
    print(inst)
    print(train_rankings_np_rank)
    print(rank)
    perf = np.zeros_like(rank)

    # inst_, perf_, rank_ = util.construct_numpy_representation_with_ordered_pairs_of_rankings_and_features(
    #         self.train_features,
    #         np.argsort(train_rank),
    #         max_pairs_per_instance=15,
    #         seed=15)
    # rank_ = rank_.astype("int32")

    if (len(train_rankings) == 0):
        continue

    model1 = log_linear.LogLinearModel()
    model2 = neural_net_hinge.NeuralNetworkSquaredHinge()
    model3 = neural_net.NeuralNetwork()

    # train_pairs = util.sample_pairs(train_rankings, 20, seed=15)
    # print(train_pairs.shape)
    # print(train_features.shape)

    # model1.fit_list(train_rankings_np.shape[1],train_rankings_np_rank.tolist(),train_features_np,None,lambda_value=1,regression_loss="Squared", maxiter=50)
    model1.fit_np(train_rankings.shape[1],
                  rank,
                  inst,
                  perf,
                  lambda_value=1,
                  regression_loss="Squared",
                  maxiter=100)
    train_rankings_np = train_rankings_np.astype("int32")
    model2.fit(train_rankings.shape[1],
               rank.astype("int32"),
               inst.astype("float64"),
               perf.astype("float64"),
               lambda_value=0.0,
               epsilon_value=1.0,
               regression_loss="Squared",
               num_epochs=500,
               learning_rate=0.001,
               hidden_layer_sizes=[16,16],
               activation_function="relu",
               batch_size=64)
    model3.fit(train_rankings.shape[1],
               rank.astype("int32"),
               inst.astype("float64"),
               perf.astype("float64"),
               lambda_value=1.0,
               regression_loss="Squared",
               num_epochs=500,
               learning_rate=0.001,
               hidden_layer_sizes=[16,16],
               activation_function="relu",
               batch_size=64)
    # model1.weights = model2.weights
    # test_weights = np.random.rand(train_rankings.shape[1], train_features.shape[1]+1)
    # print("test weights shape",test_weights.shape)
    # nll = model1.negative_log_likelihood(train_rankings[:4],train_features[:4],test_weights)
    # print("nll1", nll)
    # nll = model1.vectorized_nll(train_rankings_np[:4],train_features_np[:4],test_weights)
    # print("nll2", nll)

    #         current_taus = []

    for index, row in test_features.iterrows():
        predicted_ranking1 = model1.predict_ranking(row)
        predicted_ranking2 = model2.predict_ranking(row)
        predicted_ranking3 = model3.predict_ranking(row)
        true_ranking = test_rankings.loc[index].values
        tau1 = kendalltau(predicted_ranking1[::-1],true_ranking).correlation
        tau2 = kendalltau(predicted_ranking2, true_ranking).correlation
        tau3 = kendalltau(predicted_ranking3, true_ranking).correlation
        result_data.append([1, portion, tau1, tau2, tau3])

results = pd.DataFrame(
    data=result_data,
    columns=["split", "train_portion", "tau1", "tau2", "tau3"])
sns.set_style("darkgrid")
df = model2.get_loss_history_frame()
print(df)
print("avg kendalls tau model1:", results["tau1"].mean())
print("avg kendalls tau model2:", results["tau2"].mean())
print("avg kendalls tau model3:", results["tau3"].mean())
