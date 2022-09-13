import os
import sys

import numpy as np

# Importing logistic prediction function from ex01
working_dir = os.path.dirname(__file__)
module_path = os.path.join(working_dir, '..', 'ex01')
sys.path.insert(0, module_path)
from l2_reg import l2  # Noqa


def check_param(y, y_hat, theta, lambda_):
    if any(not isinstance(p, np.ndarray) for p in (y, y_hat)):
        return False
    if any(not np.size(p) for p in (y, y_hat)):
        return False
    if y.shape != y_hat.shape:
        return False
    if len(y.shape) == 2 and y.shape[1] != 1:
        return False
    if len(theta.shape) == 2 and theta.shape[1] != 1:
        return False
    if not isinstance(lambda_, float):
        return False
    return True


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model
    from two non-empty numpy.array, without any for loop.Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(y, y_hat, theta, lambda_):
        return None
    diff = y_hat - y
    squared_error = np.sum(diff.T @ diff, dtype=float)
    m = len(y_hat)
    reg_term = l2(theta)
    return (1 / (2 * m)) * (squared_error + lambda_ * (reg_term))


if __name__ == '__main__':
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print(reg_loss_(y, y_hat, theta, .5))
    print(reg_loss_(y, y_hat, theta, .05))
    print(reg_loss_(y, y_hat, theta, .9))
