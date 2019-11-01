import pandas as pd
import itertools as it

def break_ties_of_ranking(ranking : pd.DataFrame):
    """Breaks up ties in a ranking. A ranking 1 2 3 3 
    will be replaces by two rankings 1 2 3 and 1 2 3
    
    Arguments:
        ranking {pd.DataFrame} -- [description]
    """
    for index, row in ranking.iterrows():
        print(ranking)
        if len(set(row.values)) < len(row.values):
            # remove the row and insert new rankings
            # print(row.values)
            ranks = [[-1] for i in range(0,len(row.values))]
            for i, k in enumerate(row.values, 1):
                if ranks[k-1] == [-1]:
                    ranks[k-1] = []
                ranks[k-1].append(i)
            # print(ranks)
            ranking.drop(index)
            for new_ranking in it.product(*ranks):
                new_row = pd.Series(data=new_ranking,index=index)
                ranking.append(new_row)
        print(ranking)
        
            