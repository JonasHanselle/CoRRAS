import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario

def compute_relevance_scores_equi_width(scen, num_bins=5):
    """Compute graded relevance scores for use e.g. in 
    (noramlized) discounted cumulative gain based on an
    equi-width binning of the achieved runtime. There are
    num_bins bins, starting at 0 and ending at the algorithm
    runtime cutoff. Algorithm runs above this cutoff get a
    relevance score of zero.
    
    Arguments:
        scen {ASRankingScenario} -- AS Scenario
    
    Keyword Arguments:
        num_bins {int} -- Number of bins (default: {5})
    
    Returns:
        {pd.DataFrame} -- DataFrame containing the graded 
        relevance score for each algorithm run
    """
    performances = scen.performance_data.to_numpy()
    bins = np.linspace(start=0, stop=scen.algorithm_cutoff_time, num=num_bins)[::-1]
    binned_performances = np.digitize(performances,bins)
    return pd.DataFrame(data=binned_performances,index=scen.performance_data.index,columns=scen.performance_data.columns)
    
def ndcg_at_k(predicted_ranking, relevance_scores, k):
    """Computes the normalized discounted cumulative 
    gain at rank k. For further details refer to
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    
    Arguments:
        predicted_ranking {[type]} -- [description]
        relevance_scores {[type]} -- [description]
        k {[type]} -- [description]
    """
    #TODO real values relevance score
    ordering = np.argsort(predicted_ranking)
    predicted = relevance_scores[ordering]
    best = np.sort(relevance_scores)[::-1]
    discounts = np.log2(np.arange(k)+2)
    # dcg at k
    dcg = np.sum(predicted[:k]/discounts[:k])
    # ideal dcg at k
    idcg = np.sum(best[:k]/discounts[:k])
    ndcg = dcg/idcg
    return ndcg