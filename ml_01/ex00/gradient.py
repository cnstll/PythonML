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


def simple_gradient(x, y, theta):
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
    hypothesis = theta[0] + theta[1] * x
    j_0 = 1 / m * sum(hypothesis - y)
    j_1 = 1 / m * sum((hypothesis - y) * x)
    return np.array([j_0, j_1])


x = np.array([[12.4956442], [21.5007972],
             [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236],
             [45.7655287], [46.6793434], [59.5585554]])

theta1 = np.array([[2], [0.7]])
print(simple_gradient(x, y, theta1))

theta2 = np.array([[1], [-0.4]])
print(simple_gradient(x, y, theta2))
