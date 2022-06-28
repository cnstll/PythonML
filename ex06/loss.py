import numpy as np


def check_x(x):
    if not isinstance(x, np.ndarray) or \
       x.shape[0] == 0 or \
       (len(x.shape) == 2 and x.shape[1] == 0):
        return False
    else:
        return True


def check_thetas(x, th):
    if not isinstance(th, np.ndarray):
        return False
    elif th.shape[0] == 0:
        return False
    elif (len(th.shape) == 2 and th.shape[1] == 0):
        return False
    elif (x.shape[1] + 1 != th.shape[0]):
        return False
    else:
        return True


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
        Args:
        x: has to be an numpy.array, a vector or a matrix.
        Returns:
        x as a numpy.array, a vector of shape m * 2.
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
        Raises:
        This function should not raise any Exception.
    """
    col_of_ones = np.ones((x.shape[0]))
    return np.insert(x, 0, col_of_ones, axis=1)


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * n.
    theta: has to be an numpy.array, a vector of shape n * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not check_x(x) or not check_thetas(x, theta):
        return None
    x_ = add_intercept(x)
    return x_.dot(theta)

def loss_elem_(y, y_hat):
    """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
    """
    if not check_x(y) or not check_x(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    j_elem = pow(y_hat - y, 2)
    return j_elem

def loss_(y, y_hat):
    """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
    """
    if not check_x(y) or not check_x(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    m = len(y)
    y_sum = sum(loss_elem_(y, y_hat))
    j_value = float(y_sum / (2 * m))
    return j_value

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# Example 1:
print(loss_elem_(y1, y_hat1))
# Output:
# array([[0.], [1], [4], [9], [16]])
# Example 2:
print(loss_(y1, y_hat1))
# Output:
# 3.0

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])

# Example 3:
print(loss_elem_(y2, y_hat2))
# Output:
# array([[10.5625], [6.0025], [0.1225], [17.2225]])

# Example 4:
print(loss_(y2, y_hat2))
# Output:
# 4.238750000000004
x3 = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
# Example 5:
print(loss_elem_(y3, y_hat3))
print(loss_(y3, y_hat3))
# Output:
# 2.142857142857143
# Example 6:
print(loss_(y3, y3))
# Output:
# 0.0