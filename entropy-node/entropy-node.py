import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    pro = counts/len(y)
    shanon = -np.sum(pro * np.log2(pro))
    return float(shanon)
    