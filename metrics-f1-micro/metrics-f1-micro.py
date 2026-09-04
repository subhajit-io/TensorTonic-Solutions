def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    true_positives = sum(
        actual == predicted
        for actual, predicted in zip(y_true, y_pred)
    )
    errors = len(y_true) - true_positives
    denominator = 2 * true_positives + 2 * errors
    return round(2 * true_positives / denominator, 4)