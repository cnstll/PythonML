import numpy as np


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(thetas, (np.ndarray, list)):
            raise TypeError("Parameter thetas expect ndarray")
        if isinstance(thetas, list):
            thetas = np.array(thetas).reshape(2, 1)
        if thetas.shape != (2, 1):
            e_msg = f"Param thetas shape is {thetas.shape} should be (2, 1)"
            raise ValueError()
        if not isinstance(alpha, float):
            raise TypeError("Parameter alpha expect float")
        if not isinstance(max_iter, int):
            raise TypeError("Parameter max_iter expect int")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def param_checked(self, x, y):
        if not all(isinstance(t, np.ndarray) for t in (x, y)):
            return False
        if not all(t.shape[0] >= 2 for t in (x, y)):
            return False
        if not all(len(t.shape) == 2 and t.shape[1] == 1 for t in (x, y)):
            return False
        if x.shape != y.shape:
            return False
        return True

    @staticmethod
    def check_x(x):
        if not isinstance(x, np.ndarray) or \
           x.shape[0] == 0 or \
           (len(x.shape) == 2 and x.shape[1] == 0):
            return False
        else:
            return True

    def check_thetas(self):
        if not isinstance(self.thetas, np.ndarray):
            return False
        if not self.thetas.shape == (2, 1):
            return False
        else:
            return True

    def add_intercept(self, x):
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

    def gradient(self, x, y, tmp_thetas):
        """Computes a gradient vector from three
        non-empty numpy.array, without any for-loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not self.param_checked(x, y):
            return None
        m = x.shape[0]
        x_p = self.add_intercept(x)
        x_t = np.transpose(x_p)
        hypothesis = x_p @ tmp_thetas
        j = (1 / m) * x_t @ (hypothesis - y)
        return j

    def fit_(self, x, y):
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
        if not self.param_checked(x, y):
            return None
        new_theta = self.thetas
        for i in range(0, self.max_iter):
            tmp_theta = new_theta
            new_theta = tmp_theta - self.alpha * self.gradient(x, y, tmp_theta)
            if all(a == b for a, b in zip(tmp_theta, new_theta)):
                print(f"Iteration nbr {i} vs max_iter {self.max_iter}")
                break
        self.thetas = new_theta

    def predict_(self, x):
        """Computes the vector of prediction y_hat
        from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
        Returns:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.check_x(x):
            return None
        if not self.check_thetas():
            return None
        if len(x.shape) != 2 or x.shape[1] > 1:
            return None
        x_ = self.add_intercept(x)
        return x_.dot(self.thetas)

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
        if not MyLinearRegression.check_x(y):
            return None
        if not MyLinearRegression.check_x(y_hat):
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
        if not MyLinearRegression.check_x(y):
            return None
        if not MyLinearRegression.check_x(y_hat):
            return None
        if y_hat.shape != y.shape:
            return None
        diff = y_hat - y
        squared_error = float(sum(diff * diff))
        half_mean_divisor = float(1 / (2 * len(y)))
        return half_mean_divisor * squared_error
