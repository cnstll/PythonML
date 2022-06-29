import numpy as np
import sys
sys.path.insert(1, '../ex01')
from vec_gradient import gradient
sys.path.insert(1, '../../ml_00/ex04')
from prediction import predict_


def param_checked(x, y, theta, alpha, max_iter):
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
    if not isinstance(alpha, float):
        return None
    if not isinstance(max_iter, int):
        return None
    return True


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a vector of shape m * 1:
     (number of training examples, 1).
    y: has to be a numpy.array, a vector of shape m * 1:
     (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int,
     the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape 2 * 1.
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not param_checked(x, y, theta, alpha, max_iter):
        return None
    new_theta = theta
    for i in range(0, max_iter):
        tmp_theta = new_theta
        new_theta = tmp_theta - alpha * gradient(x, y, tmp_theta)
        if all(a == b for a, b in zip(tmp_theta, new_theta)):
            print(f"Iteration nbr {i} vs max_iter {max_iter}")
            return new_theta
    return new_theta


x = np.array([[12.4956442], [21.5007972],
             [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236],
             [45.7655287], [46.6793434], [59.5585554]])
theta = np.array([[1], [1]])
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-6, max_iter=15000)
print(f"New theta: \n{theta1}")
print()
print(f"Prediction: \n{predict_(x, theta1)}")
