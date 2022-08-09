import os
import sys

import numpy as np

# Importing logistic prediction function from ex01
dir_executed_file = os.path.dirname(__file__)
file_path = os.path.join(dir_executed_file, '..', 'ex01')
sys.path.insert(0, file_path)
from log_pred import add_ones, logistic_predict_  # Noqa


def check_param(x, theta):
    if any(not isinstance(p, np.ndarray) for p in (x, theta)):
        return False
    if any(not np.size(p) for p in (x, theta)):
        return False
    if len(x.shape) == 2 and x.shape[1] != theta.shape[0] - 1:
        return False
    return True


def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
       without any for-loop. The three arrays must have comp Args:
           x: has to be an numpy.ndarray, a matrix of shape m * n.
           y: has to be an numpy.ndarray, a vector of shape m * 1.
           theta: has to be an numpy.ndarray, a vector (n +1) * 1.
         Returns:
           The gradient as a numpy.ndarray, a vector of shape n * 1,
           containg the result of the formula for all j.
           None if x, y, or theta are empty numpy.ndarray.
           None if x, y and theta do not have compatible shapes.
         Raises:
           This function should not raise any Exception.
    """
    if not check_param(x, theta):
        return None
    m = x.shape[0]
    hypothesis = logistic_predict_(x, theta)
    diff = (hypothesis - y)
    x_t = np.transpose(add_ones(x))
    grad = 1 / m * (x_t @ diff)
    return grad


if __name__ == '__main__':
    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(vec_log_gradient(x1, y1, theta1))
    # Output:
    # array([[-0.01798621],
    #       [-0.07194484]])
    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(vec_log_gradient(x2, y2, theta2))
    # Output:
    # array([[0.3715235 ],
    #        [3.25647547]])
    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(vec_log_gradient(x3, y3, theta3))
    # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])
