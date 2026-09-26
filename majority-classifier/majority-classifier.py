import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    y_train = np.asarray(y_train, dtype=int)
    samples = np.array(X_test, dtype=float)
    values, first_positions, counts = np.unique(y_train, return_index=True, return_counts=True)
    candidates = np.flatnonzero(counts == counts.max())
    majority = values[candidates[np.argmin(first_positions[candidates])]]
    sample_count = samples.shape[0]
    return np.full(sample_count, majority, dtype=int)
