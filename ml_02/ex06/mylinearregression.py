from os import stat
import numpy as np


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.0001, max_iter=200000):
        if not isinstance(thetas, (np.ndarray, list)):
            raise TypeError("Parameter thetas expect ndarray or list")
        length = len(thetas)
        if isinstance(thetas, list):
            thetas = np.array(thetas).reshape(length, 1)
        if not isinstance(alpha, float):
            raise TypeError("Parameter alpha expect float")
        if not isinstance(max_iter, int):
            raise TypeError("Parameter max_iter expect int")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def check_param(self, x, y, theta):
        if any(not isinstance(p, np.ndarray) for p in (x, theta)):
            return False
        if any(not np.size(p) for p in (x, theta)):
            return False
        if len(x.shape) == 2 and x.shape[1] != theta.shape[0] - 1:
            return False
        if y is not None:
            if not isinstance(y, np.ndarray):
                return False
            if x.shape[0] != y.shape[0]:
                return False
            if len(y.shape) == 2 and y.shape[1] != 1:
                return False
        return True

    @staticmethod
    def add_ones(x):
        """Insert ones in the first column of the matrix x
        Args:
            x (np.ndarray): ndarray of size m * n
        Returns:
            _ (np.ndarray): ndarray of size m * (n + 1)
        """
        col_of_ones = np.ones((x.shape[0]))
        return np.insert(x, 0, col_of_ones, axis=1)

    def gradient(self, x, y, current_thetas):
        """Computes a gradient vector from three non-empty numpy.array,
        without any for-loop.
        The three arrays must have the compatible dimensions.
        Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        m = np.size(y)
        x_p = self.add_ones(x)
        x_t = np.transpose(x_p)
        h_0 = x_p @ current_thetas
        diff_outputs = h_0 - y
        vectorized_gradient = x_t @ diff_outputs
        return (1/m) * (vectorized_gradient)

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a vector of shape m * 1:
         (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1:
         (number of training examples, 1).
        Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not self.check_param(x, y, self.thetas):
            return None
        if self.alpha < 0 or self.alpha > 1:
            return None
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            return None
        new_theta = np.array(self.thetas)
        max = self.max_iter
        while max >= 0:
            g = self.gradient(x, y, new_theta)
            new_theta = new_theta - self.alpha * g
            max -= 1
        self.thetas = new_theta

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
        Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not self.check_param(x, None, self.thetas):
            return None
        x_p = self.add_ones(x)
        y_hat = x_p @ self.thetas
        return y_hat

    @staticmethod
    def loss_elem_(y, y_hat):
        """
            Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
            Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
            Returns:
            J_elem: numpy.array, a vector of dimension
                (number of the training examples,1).
            None if there is a dimension matching problem between y and y_hat.
            None if y or y_hat is not of the expected type.
            Raises:
            This function should not raise any Exception.
        """
        if any(not isinstance(i, np.ndarray) for i in (y, y_hat)):
            return None
        if y.shape != y_hat.shape:
            return None
        j_elem = pow(y_hat - y, 2)
        return j_elem

    @staticmethod
    def loss_(y, y_hat):
        """Computes the half mean squared error of
        two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if any(not isinstance(i, np.ndarray) for i in (y, y_hat)):
            return None
        if y_hat.shape != y.shape:
            return None
        diff = y_hat - y
        squared_error = float(sum(diff * diff))
        half_mean_divisor = float(1 / (2 * len(y)))
        return half_mean_divisor * squared_error

    @staticmethod
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
        mse = MyLinearRegression.loss_(y, y_hat) * 2
        return mse
