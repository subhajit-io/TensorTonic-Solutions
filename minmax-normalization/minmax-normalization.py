import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.asarray(X, dtype = float)
    X_min = np.min(X, axis=axis, keepdims=True)
    X_max = np.max(X, axis=axis, keepdims=True)
    range = X_max-X_min
    X_bar = ((X-X_min)/X_max-X_min)
    s_range = (np.where(range>eps, range, 1.0))
    return (X - X_min)/s_range