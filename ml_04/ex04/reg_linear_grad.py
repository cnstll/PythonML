import numpy as np


def check_param(x, y, theta, lambda_):
    if any(not isinstance(p, np.ndarray) for p in (x, y, theta)):
        return False
    if not isinstance(lambda_, float):
        return None
    if any(not np.size(p) for p in (x, y, theta)):
        return False
    if x.shape[0] != y.shape[0]:
        return False
    if len(y.shape) == 2 and y.shape[1] != 1:
        return False
    if len(x.shape) == 2 and x.shape[1] != theta.shape[0] - 1:
        return False
    return True


def add_ones(x):
    """Insert ones in the first column of the matrix x
    Args:
        x (np.ndarray): ndarray of size m * n
    Returns:
        _ (np.ndarray): ndarray of size m * (n + 1)
    """
    if len(x.shape) == 1:
        col_of_ones = np.ones(1)
        return np.insert(x, 0, col_of_ones, axis=0)
    else:
        col_of_ones = np.ones(x.shape[0])
        return np.insert(x, 0, col_of_ones, axis=1)


def predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimensions m * (n + 1).
    theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimensions m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = x.dot(theta)
    return y_hat


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty
    numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing
    the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, y, theta, lambda_):
        return None
    m = np.size(y)
    n = np.size(theta, axis=0)
    x_p = add_ones(x)
    gradient = np.zeros_like(theta)
    y_hat = predict(x_p, theta)
    gradient[0] = np.sum([(y_hat[i] - y[i])
                         for i in range(0, m)], dtype=float)
    for j in range(1, n):
        sum = 0
        for i in range(0, m):
            sum += (y_hat[i] - y[i]) * \
                x_p[i][j]
        gradient[j] = sum + lambda_ * theta[j]
    return (1 / m) * (gradient)


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three
    non-empty numpy.ndarray,
    without any for-loop. The three arrays must have
    compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1,
    containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, y, theta, lambda_):
        return None
    m = np.size(y)
    x_p = add_ones(x)
    x_t = np.transpose(x_p)
    h_0 = x_p.dot(theta)
    diff_outputs = h_0 - y
    vectorized_gradient = x_t.dot(diff_outputs)
    reg_term = np.copy(theta)
    reg_term[0] = 0
    return (1 / m) * (vectorized_gradient + lambda_ * reg_term)


if __name__ == '__main__':
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    print(reg_linear_grad(y, x, theta, 1))
    print(vec_reg_linear_grad(y, x, theta, 1))
    print(reg_linear_grad(y, x, theta, 0.5))
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    print(reg_linear_grad(y, x, theta, 0.0))
    print(vec_reg_linear_grad(y, x, theta, 0.0))
