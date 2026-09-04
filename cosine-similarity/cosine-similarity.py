import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.dot(a,b)
    m_a = np.sqrt(a**2 + b**2)
    m_b = np.sqrt(b**2 + a**2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a ==0 or norm_b ==0:
        return 0.0
    cos_theta = (dot/(norm_a*norm_b))
    return float(cos_theta)