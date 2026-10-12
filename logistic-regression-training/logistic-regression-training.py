import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(x: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    for s in range(steps):
        logits = x @ w + b
        preds = _sigmoid(logits)
        grad_w = x.T @ (preds - y) / n
        grad_b = np.mean(preds - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b