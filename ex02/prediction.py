import numpy as np


def check_x(x):
    if isinstance(x, np.ndarray) and \
       len(x.shape) == 2 and x.shape[0] > 1 and x.shape[1] == 1:
        return True
    else:
        return False


def check_thetas(th):
    if isinstance(th, np.ndarray) and th.shape == (2, 1):
        return True
    else:
        return False


def simple_predict(x, theta):
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
    y = np.empty_like(x)
    for i, c in enumerate(x):
        y[i] = float(theta[0] + theta[1] * c)
    return y
