import numpy as np


def check_param(x, theta):
    if any(not isinstance(p, np.ndarray) for p in (x, theta)):
        return False
    print(f"shapes: {x.shape}, {theta.shape}")
    if any(not np.size(p) for p in (x, theta)):
        return False
    if len(x.shape) == 2 and x.shape[1] != theta.shape[0] - 1:
        return False
    if len(x.shape) == 1 and x.shape[0] != theta.shape[0] - 1:
        return False
    return True


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not matching.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, theta):
        return None
    y_hat = []
    for i, t in enumerate(x):
        res = 0
        for j, th in enumerate(theta):
            if j == 0:
                res += th
            else:
                res += t[j - 1] * th
        y_hat = np.append(y_hat, res)
    return y_hat.reshape(theta.shape)
