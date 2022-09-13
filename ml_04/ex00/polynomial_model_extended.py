import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns
        to every power in the range of 1 up to the power giveArgs:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    power: has to be an int, the power up to which the columns of
    matrix x are going to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of shape m * (np),
    containg the polynomial feature vaNone if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(power, int) or \
            power <= 0:
        return None
    if power == 1:
        return x
    y = x
    for n in range(2, power + 1):
        tmp = np.power(x, n)
        y = np.append(y, tmp, axis=1)
    return y


if __name__ == '__main__':
    x = np.arange(1, 11).reshape(5, 2)
    print(x)
    print(add_polynomial_features(x, 3))
    print(add_polynomial_features(x, 4))
