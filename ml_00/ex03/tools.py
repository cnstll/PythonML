import numpy as np


def check_x(x):
    if not isinstance(x, np.ndarray) or \
       x.shape[0] == 0 or \
       (len(x.shape) == 2 and x.shape[1] == 0):
        return False
    else:
        return True


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
    if isinstance(x, np.ndarray):
        print(x.shape)
    if not check_x(x):
        return None
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)
