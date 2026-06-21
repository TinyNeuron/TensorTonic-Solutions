import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    n = len(X)
    b = 0.0
    yArr = np.array(y)
    xArr = np.array(X)
    w = np.zeros(xArr.shape[1])
    for _ in range(n_iters):
        yhat = 1 / (1 + np.exp(-(xArr @ w + b)))
        w = w - lr * 1 / n * xArr.T @ (yhat - y)
        b = b - lr * 1 / n * np.sum(yhat -y)
    return (w, b)