from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np
import sys
sys.path.insert(1, '../ex06')
from loss import loss_, check_x


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    mse = loss_(y, y_hat) * 2
    return mse


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    rmse = sqrt(loss_(y, y_hat) * 2)
    return rmse


def mae_elem_(y, y_hat):
    j_elem = abs(y_hat - y)
    return j_elem


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    if not check_x(y) or not check_x(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    m = len(y)
    y_sum = sum(mae_elem_(y, y_hat))
    mae = float(y_sum / m)
    return mae


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    if not check_x(y) or not check_x(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    mean = np.mean(y)
    sum_residual = sum(pow(py_hat - py, 2) for py_hat, py in zip(y_hat, y))
    sum_mean_residual = sum(pow(py_hat - mean, 2) for py_hat in y_hat)
    r2_score = 1.0 - float(sum_residual) / float(sum_mean_residual)
    return r2_score


x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
# Mean squared error
# your implementation
print(mse_(x, y))
# sklearn implementation
print(mean_squared_error(x, y))
# Root mean squared error
# your implementation
print(rmse_(x, y))
# sklearn implementation not available: take the square root of MSE
print(sqrt(mean_squared_error(x, y)))
# Mean absolute error
# your implementation
print(mae_(x, y))
# sklearn implementation
print(mean_absolute_error(x, y))
# R2-score
# your implementation
print(r2score_(x, y))
# sklearn implementation
print(r2_score(x, y))
