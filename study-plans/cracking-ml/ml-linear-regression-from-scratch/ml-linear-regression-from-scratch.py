import numpy as np

def linear_regression(X, y, lr, epochs):
    n = len(X)
    bias = 0.0
    xArr = np.array(X)
    yArr = np.array(y)
    w = np.zeros(xArr.shape[1])
    for _ in range(epochs):
        yhat = xArr @ w + bias
        wGrad = 2 / n * (xArr.T @ (yhat - yArr))
        biasGrad = 2 / n * np.sum(yhat - yArr)
        w = w - lr * wGrad
        bias = bias - lr * biasGrad
    return (w, bias)