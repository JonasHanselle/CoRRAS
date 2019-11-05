import numpy as np
import pandas as pd
import itertools as it

def compute_rankings(performances):
    """Computes the rankings according to the performance an algorithm achieved.
    Currently only runtime is supported as a performance measure.
    """
    rankings = performances.rank(axis=1, method="min").astype(int)
    inverse_rankings_array = np.add(rankings.values.argsort(),1) 
    inverse_rankings = pd.DataFrame(data=inverse_rankings_array, index=rankings.index, columns=rankings.columns)
    return rankings, inverse_rankings

def remove_duplicates(rankings : pd.DataFrame, inverse_rankings : pd.DataFrame):
    """Removes duplicate entries from rankings and inverse_rankings inplace
    
    Arguments:
        rankings {pd.DataFrame} -- Rankings
        inverse_rankings {pd.DataFrame} -- Inverse Rankings
    """
    if rankings is None or inverse_rankings is None:
        return 
    else:
        # remove duplicate rankings
        for index, row in rankings.iterrows():
            if len(row) > len(set(row.tolist())):
                rankings.drop(index, inplace=True)
                inverse_rankings.drop(index, inplace=True)

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

def construct_ordered_tensor(features : pd.DataFrame, performances : pd.DataFrame):
    """Constructs a N x M x (d + 2) tensor which is ordered according to the second dimension.
       N is the number of training examples
       M is the number of labels
       in the last dimension, the instnace feautres, the algorithm label and the associated true performance are stored

    
    Arguments:
        features {pd.DataFrame} -- Feature data
        performances {pd.DataFrame} -- Performance data
    """
    rankings, fnord = compute_rankings(performances)
    rankings = break_ties_of_ranking(rankings)
    tensor = np.zeros((*rankings.shape,len(features.columns)+2))
    for i, (index, row) in enumerate(rankings.iterrows()):
        row_features = features.loc[index].values
        row_performances = performances.loc[index].values
        for l in range(0, len(row.values)):
            tensor[i,l,:-2] = row_features
            tensor[i,l,-2] = row_performances[l]
            tensor[i,l,-1] = row.iloc[l]
    print(tensor.shape)
    return tensor