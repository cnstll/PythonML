import numpy as np


def check_param(x, theta):
    if any(not isinstance(p, np.ndarray) for p in (x, theta)):
        return False
    if any(not np.size(p) for p in (x, theta)):
        return False
    if len(x.shape) == 2 and x.shape[1] != theta.shape[0] - 1:
        return False
    return True


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    return 1 / (1 + np.exp(-x))


def add_ones(x):
    """Insert ones in the first column of the matrix x
    Args:
        x (np.ndarray): ndarray of size m * n
    Returns:
        _ (np.ndarray): ndarray of size m * (n + 1)
    """
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty
    numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, theta):
        return None
    z = add_ones(x) @ theta
    y_hat = sigmoid_(z)
    return y_hat


# Example 1
x = np.array([4]).reshape((-1, 1))
theta = np.array([[2], [0.5]])
print(logistic_predict_(x, theta))
# Output:
# array([[0.98201379]])
# Example 1
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(logistic_predict_(x2, theta2))
# Output:
# array([[0.98201379],
# [0.99624161],
# [0.97340301],
# [0.99875204],
# [0.90720705]])
# Example 3
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(logistic_predict_(x3, theta3))
# Output:
# array([[0.03916572],
# [0.00045262],
# [0.2890505 ]])
