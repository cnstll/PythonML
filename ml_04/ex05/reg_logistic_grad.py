import numpy as np


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


def logistic_predict(x, theta):
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
    z = x @ theta
    sigmoid = 1 / (1 + np.exp(-z))
    y_hat = sigmoid
    return y_hat


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


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty
    numpy.ndarray, with two for-loops. The three arrayArgs:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results
    of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, y, theta, lambda_):
        return None
    m = y.shape[0]
    n = theta.shape[0]
    x_p = add_ones(x)
    y_hat = logistic_predict(x_p, theta)
    gradient = np.zeros_like(theta)
    gradient[0] = np.sum([y_hat[i] - y[i] for i in range(0, m)], dtype=float)
    for j in range(1, n):
        loss = 0
        for i in range(0, m):
            loss += (y_hat[i] - y[i]) * x_p[i][j]
        reg_term = lambda_ * theta[j]
        gradient[j] = loss + reg_term
    return (1.0 / m) * gradient


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty
    numpy.ndarray,
    without any for-loop. The three arrArgs:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of shape m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of
    the formula
    for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    if not check_param(x, y, theta, lambda_):
        return None
    m = y.shape[0]
    theta.shape[0]
    x_p = add_ones(x)
    y_hat = logistic_predict(x_p, theta)
    theta_p = np.copy(theta)
    theta_p[0] = 0.0
    return (1 / m) * (x_p.T @ (y_hat - y) + lambda_ * theta_p)


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(reg_logistic_grad(y, x, theta, 1.0))
    print(vec_reg_logistic_grad(y, x, theta, 1.0))
    print(reg_logistic_grad(y, x, theta, 0.5))
    print(vec_reg_logistic_grad(y, x, theta, 0.5))
    print(reg_logistic_grad(y, x, theta, 0.0))
    print(vec_reg_logistic_grad(y, x, theta, 0.0))
