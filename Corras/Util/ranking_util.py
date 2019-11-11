import numpy as np
import pandas as pd
import itertools as it
from scipy.stats import kendalltau

def compute_rankings(performances):
    """Computes the rankings according to the performance an algorithm achieved.
    Currently only runtime is supported as a performance measure.
    """
    rankings = performances.rank(axis=1, method="min").astype(int)
    return rankings

def remove_duplicates(rankings : pd.DataFrame):
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

def break_ties_of_ranking(ranking : pd.DataFrame):
    """Breaks up ties in a ranking. A ranking 1 2 3 3 
    will be replaces by two rankings 1 2 3 -1 and 1 2 -1 3
    Entries -1 indicate that the label is absent in the 
    ranking
    
    Arguments:
        ranking {pd.DataFrame} -- [description]
    """
    new_frame = pd.DataFrame()
    for index, row in ranking.iterrows():
        if len(set(row.values)) == len(row.values):
            row = row.rename(index)
            new_frame = new_frame.append(row)
        else:
            # remove the row and insert new rankings
            ranks = [[-1] for i in range(0,len(row.values))]
            for i, k in enumerate(row.values, 1):
                if ranks[k-1] == [-1]:
                    ranks[k-1] = []
                ranks[k-1].append(i)
            ranking.drop(index, inplace=True)
            for new_ranking in it.product(*ranks):
                new_row = pd.Series(index=row.index)
                new_row[:] = -1
                for i, r in enumerate(new_ranking, 1):
                    if r >= 0:
                        new_row.iloc[r-1] = i
                new_row = new_row.rename(index)
                new_frame = new_frame.append(new_row)
    return new_frame.astype("int16")

def ordering_to_ranking(ranking_series : pd.Series):
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
        ranking[item-1] = index
    short_ranking = [x for x in ranking if x != -1]
    return short_ranking


def construct_ordered_tensor(features : pd.DataFrame, performances : pd.DataFrame):
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
    tensor = np.zeros((*rankings.shape,len(features.columns)+2))
    for i, (index, row) in enumerate(rankings.iterrows()):
        row_features = features.loc[index].values
        row_performances = performances.loc[index].values
        for l in range(0, len(row.values)):
            tensor[i,l,:-2] = row_features
            tensor[i,l,-2] = row_performances[l]
            tensor[i,l,-1] = row.iloc[l]
    return tensor

def construct_numpy_representation(features : pd.DataFrame, performances : pd.DataFrame):
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
    rankings = break_ties_of_ranking(rankings)
    joined = rankings.join(features).join(performances, lsuffix="_rank", rsuffix="_performance")
    print(features.columns)
    np_features = joined[features.columns.values].values
    np_performances = joined[[x + "_performance" for x in performances.columns]].values
    np_rankings = joined[[x + "_rank" for x in performances.columns]].values
    print(np_features)
    print(np_performances)
    print(np_rankings)
    return np_features, np_performances, np_rankings

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