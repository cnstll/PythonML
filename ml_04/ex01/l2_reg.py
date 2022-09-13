
import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray) or \
            theta.size == 0:
        return None
    if len(theta.shape) == 2 and theta.shape[1] != 1:
        return None
    sum = 0
    for n in range(1, len(theta)):
        sum += float(pow(theta[n], 2))
    return sum


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray) or \
            theta.size == 0:
        return None
    if len(theta.shape) == 2 and theta.shape[1] != 1:
        return None
    p_theta = np.copy(theta)
    p_theta[0] = 0
    return np.sum(np.dot(p_theta.T, p_theta), dtype=float)


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(iterative_l2(x))
    print(l2(x))
    y = np.array([3, 0.5, -6]).reshape((-1, 1))
    print(iterative_l2(y))
    print(l2(y))
