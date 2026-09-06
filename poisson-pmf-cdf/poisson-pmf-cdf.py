import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k + 1):
        term *= lam / i
        cdf += term
    return {"pmf": float(term), "cdf": float(cdf)}