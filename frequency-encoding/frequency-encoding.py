import numpy as np

def frequency_encoding (values):
    """
    Returns the relative frequency of every input value using NumPy.
    Preserves the order of the input list.
    """
    arr = np.asarray(values)
    ar, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    freqs = counts[inv] / arr.size
    return freqs.tolist()