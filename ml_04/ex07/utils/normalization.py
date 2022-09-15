
import numpy as np


def check_param(x):
    if not isinstance(x, np.ndarray):
        return False
    if not x.size:
        return False
    return True


def zscore(x):
    """Computes the normalized version of a non-empty numpy.array
        using the z-score standardization.
    Args:
    x: has to be an numpy.array.
    Return:
    x_p as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if not check_param(x):
        return None
    x_p = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x_p


if __name__ == "__main__":
    arr1 = np.array([[1, 4, 3], [2, 3, 10], [3, 5, 450]])
    print(zscore(arr1))
