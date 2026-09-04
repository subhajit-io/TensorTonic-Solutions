def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    num = 0.0
    den = 0.0
    for sim, rating in zip(similarities, ratings):
        if sim > 0:
            num += sim * rating
            den += sim
    if den == 0:
        return 0.0
    return num / den