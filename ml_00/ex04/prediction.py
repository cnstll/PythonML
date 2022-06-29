import numpy as np


def check_x(x):
    if not isinstance(x, np.ndarray) or \
       x.shape[0] == 0 or \
       (len(x.shape) == 2 and x.shape[1] == 0):
        return False
    else:
        return True


def check_thetas(th):
    if isinstance(th, np.ndarray) and th.shape == (2, 1):
        return True
    else:
        return False


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
        Args:
        x: has to be an numpy.array, a vector or a matrix.
        Returns:
        x as a numpy.array, a vector of shape m * 2.
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
        Raises:
        This function should not raise any Exception.
    """
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_x(x) or not check_thetas(theta):
        return None
    if len(x.shape) != 2 or x.shape[1] > 1:
        return None
    x_ = add_intercept(x)
    return x_.dot(theta)
