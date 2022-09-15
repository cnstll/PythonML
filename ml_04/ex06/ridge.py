
import numpy as np


class MyRidge:
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        if not isinstance(thetas, np.ndarray):
            raise TypeError("Thetas should be a ndarray")
        if not np.size(thetas):
            raise ValueError("Empty value in thetas")
        self.thetas = thetas
        if not isinstance(lambda_, float):
            raise TypeError("Lambda should be a float")
        self.lambda_ = lambda_

    def get_params_(self) -> dict:
        return self.__dict__

    @staticmethod
    def l2(thetas):
        """Computes the L2 regularization of a non-empty numpy.ndarray,
        without any for-loop.
        Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(thetas, np.ndarray) or \
                thetas.size == 0:
            return None
        if len(thetas.shape) == 2 and thetas.shape[1] != 1:
            return None
        p_theta = np.copy(thetas)
        p_theta[0] = 0
        return np.sum(np.dot(p_theta.T, p_theta), dtype=float)

    @staticmethod
    def check_param_x(x, thetas):
        if not isinstance(x, np.ndarray):
            return False
        if not np.size(x):
            return False
        if len(x.shape) == 2 and x.shape[1] != thetas.shape[0] - 1:
            return False
        return True

    @staticmethod
    def check_param_x_y(x, y, thetas):
        if any(not isinstance(p, np.ndarray) for p in (x, y)):
            return False
        if any(not np.size(p) for p in (x, y)):
            return False
        if x.shape[0] != y.shape[0]:
            return False
        if len(y.shape) == 2 and y.shape[1] != 1:
            return False
        if len(x.shape) == 2 and x.shape[1] != thetas.shape[0] - 1:
            return False
        return True

    def set_params_(self, params: dict) -> None:
        key_to_type = {'thetas': np.ndarray, 'alpha': float,
                       'lambda_': float, 'max_iter': int}
        if not isinstance(params, dict):
            raise TypeError("Param key should be a string")
        if any(k not in self.__dict__ for k in params.keys()):
            raise KeyError("A param to set does not exist")
        if any(not isinstance(v, key_to_type[k]) for k, v in params.items()):
            raise TypeError("A param to set as an unsupported type")
        if any(k == 'thetas' and v.shape != self.thetas.shape
                for k, v in params.items()):
            raise ValueError(
                f"Thetas dimension does not fit {self.thetas.shape}")
        if any(k in ['alpha', 'lambda_', 'max_iter'] and v < 0
                for k, v in params.items()):
            raise ValueError('Value cannot be negative')
        for k, v in params.items():
            self.__dict__[k] = v

    @staticmethod
    def loss_(self, y_hat, y) -> float:
        squared_error = MyRidge.loss_elem_(y_hat, y)
        m = len(y)
        weighted = float(1 / (2 * m))
        return weighted * (squared_error)

    @staticmethod
    def loss_elem_(self, y_hat, y):
        if any(not isinstance(p, np.ndarray) for p in (y_hat, y)):
            return None
        if any(not np.size(p) for p in (y_hat, y)):
            return None
        if y_hat.shape != y.shape:
            return None
        diff = y_hat - y
        squared_error = np.sum(diff.T @ diff, dtype=float)
        reg_term = MyRidge.l2(thetas)
        return squared_error + self.lambda_ * reg_term

    @staticmethod
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
        if not self.check_param(x, self.thetas):
            return None
        y_hat = MyRidge.add_ones(x).dot(self.thetas)
        return y_hat

    def gradient_(self, x, y, tmp_thetas):
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
        m = np.size(y)
        x_p = MyRidge.add_ones(x)
        x_t = np.transpose(x_p)
        h_0 = x_p.dot(tmp_thetas)
        diff_outputs = h_0 - y
        vectorized_gradient = x_t.dot(diff_outputs)
        reg_term = np.copy(tmp_thetas) * self.lambda_
        reg_term[0] = 0
        return (1 / m) * (vectorized_gradient + reg_term)

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
        if self.check_param_x_y(x, y, self.thetas):
            return None
        for i in range(0, self.max_iter):
            tmp_thetas = np.copy(self.thetas)
            self.thetas = tmp_thetas - self.alpha * \
                self.gradient_(x, y, tmp_thetas)
            if tmp_thetas - self.thetas < 1e-9 \
                    and tmp_thetas - self.thetas > -1e-9:
                print(f"Loop stopped on {i} iteration")
                break

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
        mse = MyRidge.loss_(y, y_hat) * 2
        return mse


if __name__ == '__main__':
    thetas = np.array([1.0, 2.0, 3.0])
    model_1 = MyRidge(thetas=thetas)
    print(model_1.get_params_())
    print(model_1.set_params_(
        {'thetas': np.array([42.0, 2.0, 3.0]), 'max_iter': 42}))
    print(model_1.get_params_())
