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


def add_ones(x):
    """Insert ones in the first column of the matrix x
    Args:
        x (np.ndarray): ndarray of size m * n
    Returns:
        _ (np.ndarray): ndarray of size m * (n + 1)
    """
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimensions m * n.
    theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimensions m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, theta):
        return None
    x_p = add_ones(x)
    print(x_p.shape)
    y_hat = x_p.dot(theta)
    print(y_hat)
    return y_hat.reshape(theta.shape)
