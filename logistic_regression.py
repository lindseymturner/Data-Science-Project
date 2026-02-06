"""
Description: logistic regression implementation using SGD.
- sigmoid prob func
- cost func
- sgd update
- training loop (w/ shuffling)
- prediction func
- accuracy
- conf matrix

Name: Max Lovinger
Date: 12/2/25
"""

# Can grab this from Lab7 with minor modifications

import numpy as np

# Sigmoid / Probability p(y = 1 | x, w)

def prob_pos(x, w):
    """Compute p(y=1|x) using the logistic (sigmoid) function."""
    z = np.dot(x, w)
    return 1 / (1 + np.exp(-z))


# Cost function

def cost_function(X, y, w):
    """Compute negative log-likelihood cost."""
    m = len(y)
    z = X @ w
    h = 1 / (1 + np.exp(-z))
    # Add 1e-9 to prevent log(0)
    cost = -(1/m) * np.sum(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))
    return cost


# Stochastic Gradient Descent update
def SGD(xi, yi, w, alpha):
    """Perform one SGD update: w := w - alpha * gradient."""
    h = prob_pos(xi, w)
    g = (h - yi) * xi
    return w - alpha * g


# Training loop 
# Note we will have a max amout of iterations to prevent infinite loops (This was occuring). 
def train_logistic_regression(X, y, alpha=0.01, tolerance=1e-6, max_iter = 1000):
    """
    Train logistic regression using SGD.
    Stop when |prev_cost - current_cost| < tolerance or max_iter reached.
    """
    m, n = X.shape
    w = np.zeros(n)

    prev_cost = float('inf')
    curr_cost = cost_function(X, y, w)
    
    iteration = 0
    while abs(prev_cost - curr_cost) > tolerance and iteration < max_iter:
        # Shuffle data
        idx = np.arange(m)
        np.random.shuffle(idx)

        X_shuffled = X[idx]
        y_shuffled = y[idx]

        for xi, yi in zip(X_shuffled, y_shuffled):
            w = SGD(xi, yi, w, alpha)

        prev_cost = curr_cost
        curr_cost = cost_function(X, y, w)
        iteration += 1

    return w


# Prediction (0/1)
def pred(X, w, threshold):
    y_hat = []
    for xi in X:
        h = prob_pos(xi, w)
        # IMPORTANT:
        # Adjust this for out threshold : 
        y_hat.append(1 if h >= threshold else 0)
    return np.array(y_hat)


# Accuracy
def acc(y, y_hat):
    correct = np.sum(y == y_hat)
    total = len(y)
    return correct / total, correct, total


# Confusion Matrix
def confusion_matrix(y, y_hat):
    tp = tn = fp = fn = 0

    for yi, yhi in zip(y, y_hat):
        if yi == 1 and yhi == 1:
            tp += 1
        elif yi == 0 and yhi == 0:
            tn += 1
        elif yi == 0 and yhi == 1:
            fp += 1
        elif yi == 1 and yhi == 0:
            fn += 1

    return tp, tn, fp, fn


