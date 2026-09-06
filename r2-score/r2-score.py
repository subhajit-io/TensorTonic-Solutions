import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    men = np.mean(y_true)
    ssot = np.sum((y_true - men)**2)
    ssres = np.sum((y_true - y_pred)**2)
    if ssot == 0 :
        return 1.0 if ssres==0 else 0.0
    rsq = 1 - (ssres/ssot)
    return float(rsq)