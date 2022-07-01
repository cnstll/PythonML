import numpy as np
import math

def check_param(x):
    return True


def mean(x):
    if check_param(x):
        sum = 0
        for m, n in enumerate(x):
            sum += n
        else:
            return sum / (m + 1)
    else:
        return None


def var(x):
    if check_param(x):
        avg = mean(x)
        sum = 0
        for m, n in enumerate(x):
            sum += pow(n - avg, 2)
        else:
            return sum / m
    else:
        return None


def std(x):
    if check_param(x):
        return math.sqrt(var(x))
    else:
        return None

def zscore(x):
    """Computes the normalized version of a non-empty numpy.array using the z-score standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x_p as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    x_p = (x - mean(x)) / std(x)
    return x_p

X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
print(zscore(X))

Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(zscore(Y))