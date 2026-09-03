import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    e_v_d = np.dot(x,p)
    return e_v_d