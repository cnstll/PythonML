import numpy as np


def add_ones(x):
    """Insert ones in the first column of the matrix x
    Args:
        x (np.ndarray): ndarray of size m * n
    Returns:
        _ (np.ndarray): ndarray of size m * (n + 1)
    """
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)


def check_param(x, y, theta):
    if any(not isinstance(p, np.ndarray) for p in (x, y, theta)):
        return False
    if any(not np.size(p) for p in (x, y, theta)):
        return False
    if x.shape[0] != y.shape[0]:
        return False
    if len(y.shape) == 2 and y.shape[1] != 1:
        return False
    if len(x.shape) == 2 and x.shape[1] != theta.shape[0] - 1:
        return False
    return True


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,
    without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, y, theta):
        return None
    m = np.size(y)
    x_p = add_ones(x)
    x_t = np.transpose(x_p)
    h_0 = x_p.dot(theta)
    diff_outputs = h_0 - y
    vectorized_gradient = x_t.dot(diff_outputs)
    return (1/m) * (vectorized_gradient)


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, y, theta):
        return None
    if alpha < 0 or alpha > 1:
        return None
    if not isinstance(max_iter, int) or max_iter < 0:
        return None
    new_theta = np.array(theta)
    while max_iter >= 0:
        g = gradient(x, y, new_theta)
        new_theta = new_theta - alpha * g
        max_iter -= 1
    return new_theta