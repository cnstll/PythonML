import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array
        using the min-max standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x_p as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    if len(x.shape) == 1 and x.shape[0] <= 1:
        return None
    if len(x.shape) == 2 and x.shape[0] > 1 and x.shape[1] > 1:
        return None
    x_p = (x - min(x)) / (max(x) - min(x))
    return x_p


X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
print(minmax(X))

Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print(minmax(Y))
