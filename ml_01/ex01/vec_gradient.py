import numpy as np


def param_checked(x, y, theta):
    if not all(isinstance(t, np.ndarray) for t in (x, y, theta)):
        return False
    if not all(t.shape[0] >= 2 for t in (x, y, theta)):
        return False
    if not all(len(t.shape) == 2 and t.shape[1] == 1 for t in (x, y, theta)):
        return False
    if x.shape != y.shape:
        return False
    if theta.shape[0] != 2:
        return False
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
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)


def gradient(x, y, theta):
    """Computes a gradient vector from three
    non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not param_checked(x, y, theta):
        return None
    m = x.shape[0]
    x_p = add_intercept(x)
    x_t = np.transpose(x_p)
    hypothesis = x_p @ theta
    j = (1 / m) * x_t @ (hypothesis - y)
    return j


x = np.array([[12.4956442], [21.5007972],
             [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236],
             [45.7655287], [46.6793434], [59.5585554]])

theta1 = np.array([[2], [0.7]])
print(gradient(x, y, theta1))

theta2 = np.array([[1], [-0.4]])
print(gradient(x, y, theta2))
