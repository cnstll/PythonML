import numpy as np


def check_param(x, power):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return False
    if not isinstance(power, int) or power < 0:
        return False
    return True


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising
    its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the
    components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array,
    of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, power):
        return None
    if power == 0:
        return np.array(x**0)
    mat = np.array(x)
    for n in range(2, power + 1):
        vec = np.array(x**n)
        mat = np.append(mat, vec, axis=1)
    return mat
