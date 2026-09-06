import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    probabilities = [
        math.comb(n, i) * p ** i * (1.0 - p) ** (n - i)
        for i in range(k + 1)
    ]
    return {"pmf": float(probabilities[k]), "cdf": float(sum(probabilities))}