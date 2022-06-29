import numpy as np


def check_x(x):
    if not isinstance(x, np.ndarray) or \
       x.shape[0] == 0 or \
       (len(x.shape) == 2 and x.shape[1] == 0):
        return False
    else:
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
    if not check_x(y_hat) and not check_x(y):
        return None
    if y_hat.shape != y.shape:
        return None
    diff = y_hat - y
    squared_error = sum(diff * diff)
    half_mean_divisor = 1 / (2 * len(y))
    return half_mean_divisor * float(squared_error)


X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
# Example 1:
print(loss_(X, Y))
# Output:
# 2.142857142857143
# Example 2:
loss_(X, X)
# Output
# 0.0
