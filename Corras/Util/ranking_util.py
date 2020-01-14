import autograd.numpy as np
import pandas as pd
import itertools as it
import random
from scipy.stats import kendalltau


def compute_rankings(performances):
    """Computes the rankings according to the performance an algorithm achieved.
    Currently only runtime is supported as a performance measure.
    """
    rankings = performances.rank(axis=1, method="min").astype(int)
    return rankings


def remove_duplicates(rankings: pd.DataFrame):
    """Removes duplicate entries from rankings inplace

    Arguments:
        rankings {pd.DataFrame} -- Rankings
    """
    if rankings is None:
        return
    else:
        # remove duplicate rankings
        for index, row in rankings.iterrows():
            if len(row) > len(set(row.tolist())):
                rankings.drop(index, inplace=True)


def break_ties_of_ranking(ranking: pd.DataFrame,
                          max_rankings_per_instance=15,
                          seed=15):
    """Breaks up ties in a ranking. A ranking 1 2 3 3 
    will be replaces by two rankings 1 2 3 -1 and 1 2 -1 3
    Entries -1 indicate that the label is absent in the 
    ranking

    Arguments:
        ranking {pd.DataFrame} -- [description]
    """
    random.seed(seed)
    new_frame = pd.DataFrame()
    for index, row in ranking.iterrows():
        if len(set(row.values)) == len(row.values):
            row = row.rename(index)
            new_frame = new_frame.append(row)
        else:
            # remove the row and insert new rankings
            ranks = [[-1] for i in range(0, len(row.values))]
            for i, k in enumerate(row.values, 1):
                if ranks[k - 1] == [-1]:
                    ranks[k - 1] = []
                ranks[k - 1].append(i)
            ranking.drop(index, inplace=True)
            new_rankings = []
            for new_ranking in it.product(*ranks):
                new_row = pd.Series(index=row.index)
                new_row[:] = -1
                for i, r in enumerate(new_ranking, 1):
                    if r >= 0:
                        new_row.iloc[r - 1] = i
                new_row = new_row.rename(index)
                new_rankings.append(new_row)
            new_rankings_sample = random.sample(
                new_rankings,
                min((max_rankings_per_instance, len(new_rankings))))
            for sample in new_rankings_sample:
                new_frame = new_frame.append(sample)
    return new_frame.astype("int16")


# def break_ties_of_ranking(ranking : pd.DataFrame, max_rankings_per_instance = 15, seed = 15):
#     """Breaks up ties in a ranking. A ranking 1 2 3 3
#     will be replaces by two rankings 1 2 3 -1 and 1 2 -1 3
#     Entries -1 indicate that the label is absent in the
#     ranking

#     Arguments:
#         ranking {pd.DataFrame} -- [description]
#     """
# new_frame = pd.DataFrame()
# for index, row in ranking.iterrows():
#     if len(set(row.values)) == len(row.values):
#         row = row.rename(index)
#         new_frame = new_frame.append(row)
#     else:
#         # remove the row and insert new rankings
#         ranks = [[-1] for i in range(0,len(row.values))]
#         for i, k in enumerate(row.values, 1):
#             if ranks[k-1] == [-1]:
#                 ranks[k-1] = []
#             ranks[k-1].append(i)
#         ranking.drop(index, inplace=True)
#         for new_ranking in it.product(*ranks):
#             new_row = pd.Series(index=row.index)
#             new_row[:] = -1
#             for i, r in enumerate(new_ranking, 1):
#                 if r >= 0:
#                     new_row.iloc[r-1] = i
#             new_row = new_row.rename(index)
#             new_frame = new_frame.append(new_row)
# return new_frame.astype("int16")


def ordering_to_ranking_frame(ordering_frame: pd.DataFrame):
    rankings = []
    for instance, series in ordering_frame.iterrows():
        ranking = [-1] * len(ordering_frame.columns)
        for count, (index, item) in enumerate(series.iteritems()):
            ranking[item - 1] = count
        rankings.append(ranking)
    new_frame = pd.DataFrame(
        data=rankings,
        columns=[str(i + 1) for i in range(len(ordering_frame.columns))],
        index=ordering_frame.index)
    return new_frame


def ordering_to_ranking(ranking_series: pd.Series):
    """Create a ranking from a DataFrame that has one 
    column for each label and each entry corresponds to
    the rank of that label in the ranking. -1 indicates
    that the label is absent in the ranking. 

    Arguments:
        rankings {pd.DataFrame} -- [description]

    Returns:
        [type] -- [description]
    """
    ranking = [-1] * len(ranking_series)
    for index, item in ranking_series.iteritems():
        # set index of
        ranking[item - 1] = index
    short_ranking = [x for x in ranking if x != -1]
    return short_ranking


def ordering_to_ranking_matrix(ordering_matrix: np.ndarray):
    """Create a ranking from a DataFrame that has one 
    column for each label and each entry corresponds to
    the rank of that label in the ranking. -1 indicates
    that the label is absent in the ranking. 

    Arguments:
        rankings {pd.DataFrame} -- [description]

    Returns:
        [type] -- [description]
    """
    rankings = []
    for ordering_index in range(0, ordering_matrix.shape[0]):
        ranking = [-1] * ordering_matrix.shape[1]
        for label_index in range(0, ordering_matrix.shape[1]):
            # set index of
            label = ordering_matrix[ordering_index, label_index]
            ranking[label - 1] = label_index + 1
        # short_ranking = [x for x in ranking if x != -1]
        rankings.append(np.asarray(ranking))
    return np.asarray(rankings)


def ordering_to_ranking_list(ordering_matrix: np.ndarray):
    """Create a ranking from a DataFrame that has one 
    column for each label and each entry corresponds to
    the rank of that label in the ranking. -1 indicates
    that the label is absent in the ranking. 

    Arguments:
        rankings {pd.DataFrame} -- [description]

    Returns:
        [type] -- [description]
    """
    rankings = []
    for ordering_index in range(0, ordering_matrix.shape[0]):
        ranking = [-1] * ordering_matrix.shape[1]
        for label_index in range(0, ordering_matrix.shape[1]):
            # set index of
            label = ordering_matrix[ordering_index, label_index]
            ranking[label - 1] = label_index + 1
        short_ranking = [x for x in ranking if x != -1]
        rankings.append((short_ranking))
    return rankings


def construct_ordered_tensor(features: pd.DataFrame,
                             performances: pd.DataFrame):
    """Constructs a N x M x (d + 2) tensor which is ordered according to the second dimension.
       N is the number of training examples
       M is the number of labels
       in the last dimension, the instnace feautres, the algorithm label and the associated true performance are stored


    Arguments:
        features {pd.DataFrame} -- Feature data
        performances {pd.DataFrame} -- Performance data
    """
    rankings = compute_rankings(performances)
    rankings = break_ties_of_ranking(rankings)
    tensor = np.zeros((*rankings.shape, len(features.columns) + 2))
    for i, (index, row) in enumerate(rankings.iterrows()):
        row_features = features.loc[index].values
        row_performances = performances.loc[index].values
        for l in range(0, len(row.values)):
            tensor[i, l, :-2] = row_features
            tensor[i, l, -2] = row_performances[l]
            tensor[i, l, -1] = row.iloc[l]
    return tensor


def construct_numpy_representation(features: pd.DataFrame,
                                   performances: pd.DataFrame,
                                   max_rankings_per_instance=5,
                                   seed=15):
    """Get numpy representation of features, performances and rankings

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms

    Returns:
        [type] -- Triple of numpy ndarrays, first stores the feature
        values, the second stores the algirhtm performances and the
        third stores the algorithm rankings
    """
    rankings = compute_rankings(performances)
    rankings = break_ties_of_ranking(
        rankings, max_rankings_per_instance=max_rankings_per_instance)

    joined = rankings.join(features).join(performances,
                                          lsuffix="_rank",
                                          rsuffix="_performance")
    np_features = joined[features.columns.values].values
    np_performances = joined[[
        x + "_performance" for x in performances.columns
    ]].values
    np_rankings = joined[[x + "_rank" for x in performances.columns]].values
    return np_features, np_performances, np_rankings


def construct_numpy_representation_only_performances(
    features: pd.DataFrame, performances: pd.DataFrame):
    """Get numpy representation of features, performances

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms

    Returns:
        [type] -- Triple of numpy ndarrays, first stores the feature
        values, the second stores the algirhtm performances and the
        third stores the algorithm rankings
    """

    joined = features.join(performances)
    np_features = joined[features.columns.values].values
    np_performances = joined[[x for x in performances.columns]].values
    return np_features, np_performances


def construct_numpy_representation_with_pairs_of_rankings(
    features: pd.DataFrame,
    performances: pd.DataFrame,
    max_pairs_per_instance=100,
    seed=15,
    order="asc",
    skip_value=None):
    """Get numpy representation of features, performances and rankings

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms

    Returns:
        [type] -- Triple of numpy ndarrays, first stores the feature
        values, the second stores the algorithm performances and the
        third stores the algorithm rankings
    """
    labels, ranks = sample_pairs(performances,
                            pairs_per_instance=max_pairs_per_instance,
                            seed=seed,
                            skip_value=skip_value)
    joined = labels.join(features).join(performances,
                                          lsuffix="_rank",
                                          rsuffix="_performance")
    np_features = joined[features.columns.values].values
    np_performances = joined[[x for x in performances.columns]].values
    np_labels = joined[[x for x in labels.columns]].values + 1
    if order == "desc":
        np_labels = np.flip(np_labels, axis=1)

    return np_features, np_performances, np_labels


def construct_numpy_representation_with_ordered_pairs_of_rankings_and_features(
    features: pd.DataFrame,
    performances: pd.DataFrame,
    max_pairs_per_instance=100,
    seed=15,
    order="asc",
    skip_value=None):
    """Get numpy representation of features, performances and rankings

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms
        max_pairs_per_instance {pd.DataFrame} -- Upper bound for sampled pairs per instance
        seed {pd.DataFrame} -- Seed used for sampling randomly
        order {pd.DataFrame} -- Takes values "asc" and "desc", to decide whether pairs are in ascending or descending order
        skip_value {pd.DataFrame} -- Pairs containing this value are being skipped during sampling, the intended use is ignoring timed out algorithm runs in the context of algorithm selection

    Returns:
        [type] -- Triple of numpy ndarrays, first stores the feature
        values, the second stores the algirhtm performances and the
        third stores the algorithm rankings
    """
    rankings, weights = sample_pairs(performances,
                            pairs_per_instance=max_pairs_per_instance,
                            seed=seed,
                            skip_value=skip_value)
    joined = rankings.join(features).join(performances,
                                          lsuffix="_rank",
                                          rsuffix="_performance")
    np_features = joined[features.columns.values].values
    np_performances = joined[[x for x in performances.columns]].values
    np_rankings = joined[[x for x in rankings.columns]].values + 1
    np_performances = np_performances[
        np.arange(np_performances.shape[0])[:, np.newaxis], np_rankings - 1]
    # TODO check for maximization problems
    # if order == "desc":
    #     np_rankings = np.flip(np_rankings,axis=1)
    return np_features, np_performances, np_rankings


def construct_numpy_representation_with_ordered_pairs_of_rankings_and_features_and_weights(
    features: pd.DataFrame,
    performances: pd.DataFrame,
    max_pairs_per_instance=100,
    seed=15,
    order="asc", skip_value=None):
    """Get numpy representation of features, performances and rankings

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms

    Returns:
        [type] -- Triple of numpy ndarrays, first stores the feature
        values, the second stores the algirhtm performances and the
        third stores the algorithm rankings
    """
    rankings, weights = sample_pairs(performances,
                            pairs_per_instance=max_pairs_per_instance,
                            seed=seed, skip_value=skip_value)
    joined = rankings.join(features).join(performances,
                                          lsuffix="_rank",
                                          rsuffix="_performance")
    np_features = joined[features.columns.values].values
    np_performances = joined[[x for x in performances.columns]].values
    np_rankings = joined[[x for x in rankings.columns]].values + 1
    np_performances = np_performances[
        np.arange(np_performances.shape[0])[:, np.newaxis], np_rankings - 1]
    max_len = len(performances.columns)
    np_weights = weights.to_numpy()
    np_weights = np.amax(np_weights, axis=1)
    # print("np_weights", np_weights)
    np_weights = np.exp2(np_weights)
    # print("exp np_weights", np_weights)

    # TODO check for maximization problems
    # if order == "desc":
    #     np_rankings = np.flip(np_rankings,axis=1)

    return np_features, np_performances, np_rankings, np_weights


def enumerate_pairs(k):
    """Enumerates ordered pairs

    Arguments:
        k {[type]} -- Up to which index ordered pairs should be enumerated

    Returns:
        [type] -- List of tuples of indices
    """
    result = []
    for i in range(1, k):
        for j in range(0, i):
            result.append((j, i))
    return result


def sample_pairs(performances: pd.DataFrame,
                 pairs_per_instance: int,
                 seed: int,
                 skip_value=None):
    """Sample ordered pairs from a Pandas DataFrame containing. Pairs containing the skip_value are being skipped. 

    Arguments:
        performances {pd.DataFrame} -- [description]
        pairs_per_instance {int} -- [description]
        seed {int} -- [description]

    Keyword Arguments:
        skip_value {[type]} -- [Pairs containing the skip value will be skipped during sampling. The intended use is skipping timed out algorithm runs in the context of algorithm selection.] (default: {None})

    Returns:
        [type] -- [description]
    """
    pairs_result = []
    ranks_result = []
    indices = []
    random.seed(seed)
    pairs = enumerate_pairs(len(performances.columns))
    for index, row in performances.iterrows():
        ranks = row.rank(method="max")
        random.shuffle(pairs)
        candidates = pairs[:]
        i = 0
        while i < pairs_per_instance and candidates:
            pair = candidates.pop(0)
            if skip_value is not None:
                if row.iloc[pair[0]] == skip_value or row.iloc[
                        pair[1]] == skip_value:
                    print("skipping skip value: " + str(skip_value))
                    continue
            if row.iloc[pair[0]] > row.iloc[pair[1]]:
                pairs_result.append([pair[1], pair[0]])
                ranks_result.append([ranks[pair[1]], ranks[pair[0]]])
                indices.append(index)
                i += 1
            elif row.iloc[pair[0]] < row.iloc[pair[1]]:
                pairs_result.append([pair[0], pair[1]])
                ranks_result.append([ranks[pair[0]], ranks[pair[1]]])
                indices.append(index)
                i += 1
        if not candidates:
            pass
    return pd.DataFrame(data=pairs_result, index=indices,
                        columns=[1, 2]), pd.DataFrame(data=ranks_result,
                                                      index=indices,
                                                      columns=[1, 2])


def sample_ranking_pairs_with_features(performances: pd.DataFrame,
                                       features: pd.DataFrame,
                                       pairs_per_instance: int,
                                       seed: int,
                                       skip_value=None):
    pairs_result = []
    indices = []
    random.seed(seed)
    pairs = enumerate_pairs(len(performances.columns))
    for index, row in performances.iterrows():
        random.shuffle(pairs)
        candidates = pairs[:]
        i = 0
        while i < pairs_per_instance and candidates:
            pair = candidates.pop(0)
            if skip_value is not None:
                if row.iloc[pair[0]] == skip_value or row.iloc[
                        pair[1]] == skip_value:
                    print("skipping skip value: " + str(skip_value))
                    continue
            if row.iloc[pair[0]] > row.iloc[pair[1]]:
                pairs_result.append([pair[1], pair[0]])
                indices.append(index)
                i += 1
            elif row.iloc[pair[0]] < row.iloc[pair[1]]:
                pairs_result.append([pair[0], pair[1]])
                indices.append(index)
                i += 1
        if not candidates:
            pass
    return pd.DataFrame(data=pairs_result, index=indices, columns=[1, 2])


def sample_ranking_pairs_with_features_from_rankings(features: np.ndarray,
                                                     rankings: np.ndarray,
                                                     pairs_per_instance: int,
                                                     seed: int):
    pairs_result = []
    indices = []
    random.seed(seed)
    pairs = enumerate_pairs(rankings.shape[1])
    new_features = []
    new_rankings = []
    for feature_vec, ranking in zip(features, rankings):
        random.shuffle(pairs)
        candidates = pairs[:]
        i = 0
        while i < pairs_per_instance and candidates:
            pair = candidates.pop(0)
            new_features.append(feature_vec)
            new_rankings.append([ranking[pair[0]], ranking[pair[1]]])
            i += 1
        if not candidates:
            pass
    return np.asarray(new_features), np.asarray(new_rankings)


def construct_numpy_representation_with_list_rankings(
    features: pd.DataFrame,
    performances: pd.DataFrame,
    max_rankings_per_instance=5,
    seed=15,
    pairs=False):
    """Get numpy representation of features and performances. Rankings
    are constructed as nested python lists, such that rankings of 
    heterogenous length are possible.

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms

    Returns:
        [type] -- Tupel of numpy ndarrays, first stores the feature
        values, the second stores the algirhtm performances. The 
        third return is a list of python lists containing the 
        rankings
    """
    rankings = compute_rankings(performances)
    if pairs:
        rankings = break_ties_of_ranking_pairs(
            rankings, max_rankings_per_instance=max_rankings_per_instance)
    else:
        rankings = break_ties_of_ranking(
            rankings, max_rankings_per_instance=max_rankings_per_instance)
    joined = rankings.join(features).join(performances,
                                          lsuffix="_rank",
                                          rsuffix="_performance")
    np_features = joined[features.columns.values].values
    np_performances = joined[[
        x + "_performance" for x in performances.columns
    ]].values
    np_rankings = joined[[x + "_rank" for x in performances.columns]].values
    return np_features, np_performances, np_rankings


# def construct_numpy_representation_with_list_rankings(features : pd.DataFrame, performances : pd.DataFrame, max_rankings_per_instance = 5, seed = 15):
#     """Get numpy representation of features and performances. Rankings
#     are constructed as nested python lists, such that rankings of
#     heterogenous length are possible.

#     Arguments:
#         features {pd.DataFrame} -- Feature values
#         performances {pd.DataFrame} -- Performances of algorithms

#     Returns:
#         [type] -- Tupel of numpy ndarrays, first stores the feature
#         values, the second stores the algirhtm performances. The
#         third return is a list of python lists containing the
#         rankings
#     """
#     rankings = compute_rankings(performances)
#     rankings = break_ties_of_ranking(rankings, max_rankings_per_instance=max_rankings_per_instance)

#     joined = rankings.join(features).join(performances, lsuffix="_rank", rsuffix="_performance")
#     np_features = joined[features.columns.values].values
#     np_performances = joined[[x + "_performance" for x in performances.columns]].values
#     np_rankings = joined[[x + "_rank" for x in performances.columns]].values
#     return np_features, np_performances, np_rankings

# def construct_numpy_representation_with_list_of_n_klets(features : pd.DataFrame, performances : pd.DataFrame, n : int, k : int, seed : int):
#     """Get numpy representation of features and performances. Rankings
#     are constructed as nested python lists, such that rankings of
#     heterogenous length are possible.

#     Arguments:
#         features {pd.DataFrame} -- Feature values
#         performances {pd.DataFrame} -- Performances of algorithms
#         n {int} -- Number of partial rankings per instance
#         k {int} -- Length of

#     Returns:
#         [type] -- Tupel of numpy ndarrays, first stores the feature
#         values, the second stores the algirhtm performances. The
#         third return is a list of python lists containing the
#         rankings
#     """
#     rankings = compute_rankings(performances)
#     rankings = break_ties_of_ranking(rankings)

#     joined = rankings.join(features).join(performances, lsuffix="_rank", rsuffix="_performance")
#     np_features = joined[features.columns.values].values
#     np_performances = joined[[x + "_performance" for x in performances.columns]].values
#     np_rankings = joined[[x + "_rank" for x in performances.columns]].values
#     return np_features, np_performances, np_rankings


def custom_tau(ranking_a, ranking_b):
    """Custom implementaion of the kendalls tau rank correlation coefficient
    that allows computation on partial rankings. Elements that are not
    present in both of the rankings are not considered for the computation
    of kendalls tau

    Arguments:
        ranking_a {[type]} -- [description]
        ranking_b {[type]} -- [description]
    """
    elements = set(ranking_a).intersection(ranking_b)
    if len(elements) < 2:
        return 0
    ranking_a = [x for x in ranking_a if x in elements]
    ranking_b = [x for x in ranking_b if x in elements]
    return kendalltau(ranking_a, ranking_b).correlation
