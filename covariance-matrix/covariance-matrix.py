import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    cent  = X-np.mean(X, axis=0)
    return cent.T @ cent / (X.shape[0]-1)