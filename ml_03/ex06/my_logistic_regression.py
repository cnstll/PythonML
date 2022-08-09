
import matplotlib.pyplot as plt
import numpy as np


class MyLogisticRegression():
    """
        Description:
                My personnal logistic regression to classify things.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=100000):
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
        self.eps = 1e-15

    @staticmethod
    def check_param(x, y, theta):
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

    def vec_log_gradient(self, x, y, theta):
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
        if not MyLogisticRegression.check_param(x, None, theta):
            return None
        m = x.shape[0]
        z = MyLogisticRegression.add_ones(x) @ theta
        y_hat = 1 / (1 + np.exp(-z))
        diff = (y_hat - y)
        x_t = np.transpose(MyLogisticRegression.add_ones(x))
        grad = 1 / m * (x_t @ diff)
        return grad

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
        if not MyLogisticRegression.check_param(x, y, self.thetas):
            return None
        if self.alpha < 0 or self.alpha > 1:
            return None
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            return None
        new_theta = np.array(self.thetas)
        max = self.max_iter
        grad_values = np.array([])
        while max >= 0:
            g = self.vec_log_gradient(x, y, new_theta)
            new_theta = new_theta - self.alpha * g
            if max % 100 == 0:
                if np.size(grad_values) == 0:
                    grad_values = np.array(g).reshape(1, -1)
                else:
                    g_r = g.reshape(1, -1)
                    grad_values = np.append(grad_values, g_r, axis=0)
            max -= 1
        self.thetas = new_theta
        return grad_values

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
        if not MyLogisticRegression.check_param(x, None, self.thetas):
            return None
        z = MyLogisticRegression.add_ones(x) @ self.thetas
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def loss_elem_(self, y, y_hat):
        """
            Description:
            Calculates all the sub elements of the sum of the loss function.
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
        if y.shape[0] != y_hat.shape[0] \
           or y.shape[1] != 1 \
           or y_hat.shape[1] != 1:
            return None
        y.shape[0]
        ones = np.ones_like(y, dtype=float)
        j_elem = y * np.log(y_hat + self.eps)
        j_elem += (ones - y) * np.log(ones - y_hat + self.eps)
        return j_elem

    def loss_(self, y, y_hat):
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
        j_elem = self.loss_elem_(y, y_hat)
        if j_elem is None:
            return None
        else:
            m = y.shape[0]
            return - float(1.0 / m) * sum(j_elem.flatten())


def plot_gradient(gradient):
    n_cols = 2
    n_rows = int(gradient.shape[1] / n_cols)
    win_title = 'Evolution of gradients for each parameter'
    fig, axs = plt.subplots(n_rows, n_cols, label=win_title)
    x = np.arange(0, len(gradient), 1).reshape(-1, 1)
    for i in range(0, gradient.shape[1]):
        axs[i].plot(x, gradient[:, i], 'o', label=f"theta_{i}", ms=1)
        axs[i].grid(color='lightgray', linestyle='-', linewidth=1)
        axs[i].legend()
    plt.show()


if __name__ == '__main__':
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])
    # Example 0:
    Y_hat = mylr.predict_(X)
    print(Y_hat)
    # Output:
    # array([[0.99930437],
    #        [1.        ],
    #        [1. ]])
    # Example 1:
    print(mylr.loss_(Y, Y_hat))
    # Output:
    # 11.513157421577004
    # Example 2:
    grad = mylr.fit_(X, Y)
    # plot_gradient(grad)
    print(mylr.thetas)
    # Output:
    # array([[ 1.04565272],
    #        [ 0.62555148],
    #        [ 0.38387466],
    #        [ 0.15622435],
    #        [-0.45990099]])
    # Example 3:
    Y_hat = mylr.predict_(X)
    print(Y_hat)
    # Output:
    # array([[0.72865802],
    #       [0.40550072],
    #       [0.45241588]])
    # Example 4:
    print(mylr.loss_(Y, Y_hat))
    # Output:
    # 0.5432466580663214
