import numpy as np
import pandas as pd
import itertools as it

def break_ties_of_ranking(ranking : pd.DataFrame):
    """Breaks up ties in a ranking. A ranking 1 2 3 3 
    will be replaces by two rankings 1 2 3 and 1 2 3
    
    Arguments:
        ranking {pd.DataFrame} -- [description]
    """
    print(ranking)
    for index, row in ranking.iterrows():
        if len(set(row.values)) < len(row.values):
            # remove the row and insert new rankings
            ranks = [[-1] for i in range(0,len(row.values))]
            for i, k in enumerate(row.values, 1):
                if ranks[k-1] == [-1]:
                    ranks[k-1] = []
                ranks[k-1].append(i)
            ranking.drop(index, inplace=True)
            for new_ranking in it.product(*ranks):
                new_row = pd.Series(index=row.index)
                for i, r in enumerate(new_ranking, 1):
                    new_row.iloc[r-1] = i
                ranking.append(new_row, ignore_index=True, inplace=True)
    print(ranking)
    

def construct_ordered_tensor(features : pd.DataFrame, performances : pd.DataFrame):
    """Constructs a N x M x 2 tensor which is ordered according to the second dimension.
       N is the number of training examples
       M is the number of labels
       in the last dimension, the label and the associated true performance are stored
    
    Arguments:
        features {pd.DataFrame} -- Feature data
        performances {pd.DataFrame} -- Performance data
    """
    pass