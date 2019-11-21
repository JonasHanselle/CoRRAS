import numpy as np

def dcg_at_k(truth, prediction, k=5):
    """DCG at rank k
    
    Arguments:
        truth {np.ndarray} -- Ground truth ranking
        prediction {np.ndarray} -- Predicted ranking
    
    Keyword Arguments:
        k {int} -- Rank (default: {5})
    """