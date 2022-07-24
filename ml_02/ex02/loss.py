import numpy as np


def check_param(y, y_hat):
    if any(not isinstance(p, np.ndarray) for p in (y, y_hat)):
        return False
    if any(not np.size(p) for p in (y, y_hat)):
        return False
    if y.shape != y_hat.shape:
        return False
    return True


def loss_(y, y_hat):
    """Computes the half mean squared error of
    two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(y, y_hat):
        return None
    diff = y_hat - y
    squared_error = sum(diff * diff)
    half_mean_divisor = 1 / (2 * len(y))
    return half_mean_divisor * float(squared_error)
