import numpy as np


def check_param(x):
    if not isinstance(x, np.ndarray):
        return False
    if not np.size(x):
        return False
    if len(x.shape) == 2 and x.shape[1] != 1:
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
    if not check_param(x):
        return None
    return 1 / (1 + np.exp(-x))


# Example 1:
x = np.array([[-4]])
np.set_printoptions(precision=17)
print(sigmoid_(x))
# Output:
# array([[0.01798620996209156]])
# Example 2:
x = np.array([[2]])
print(sigmoid_(x))
# Output:
# array([[0.8807970779778823]])
# Example 3:
x = np.array([[-4], [2], [0]])
print(sigmoid_(x))
# Output:
# array([[0.01798620996209156], [0.8807970779778823], [0.5]])
