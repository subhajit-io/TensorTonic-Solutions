import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x = np.sort(np.asarray(x, dtype=float))
    q = np.asarray(q, dtype=float)
    positions = q / 100.0 * (x.size - 1)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = positions - lower
    return x[lower] * (1.0 - weight) + x[upper] * weight