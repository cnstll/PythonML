import os
import sys

import numpy as np

working_dir = os.path.dirname(__file__)
module_path = os.path.join(working_dir, '..', 'ex01')
sys.path.insert(0, module_path)
from l2_reg import l2  # Noqa


def regularize(func):
    def inner(*args, **kwargs):
        m = len(args[0])
        theta = args[2]
        lambda_ = args[3]
        return func(*args) + (lambda_ / (2 * m)) * l2(theta)
    return inner


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


@regularize
def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model
    from two non-empty numpy.ndarray, without any for lArgs:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(y, y_hat, theta, lambda_):
        return None
    eps = 1e-15
    m = y.shape[0]
    ones = np.ones_like(y, dtype=float)
    loss = y * np.log(y_hat + eps)
    loss += (ones - y) * np.log(ones - y_hat + eps)
    l2(theta)
    return - float(1.0 / m) * np.sum(loss, dtype=float)


if __name__ == '__main__':
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print(reg_log_loss_(y, y_hat, theta, .5))
    print(reg_log_loss_(y, y_hat, theta, .05))
    print(reg_log_loss_(y, y_hat, theta, .9))
