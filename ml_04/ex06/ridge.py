
import numpy as np
from ml_04.ex04.reg_linear_grad import add_ones


class MyRidge:
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_

    def get_params_(self) -> dict:
        return self.__dict__

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

    def loss_(self, y_hat, y) -> float:
        diff = y_hat - y
        squared_error = np.sum(diff.T @ diff, dtype=float)
        m = len(y)
        weighted = float(1 / (2 * m))
        return weighted * squared_error

    def loss_elem_(self, y_hat, y):
        diff = y_hat - y
        squared_error = np.sum(diff.T @ diff, dtype=float)
        return squared_error

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
        y_hat = add_ones(x).dot(self.thetas)
        return y_hat

    def gradient_(self, x, y, tmp_thetas):
        m = np.size(y)
        x_p = add_ones(x)
        x_t = np.transpose(x_p)
        h_0 = x_p.dot(tmp_thetas)
        diff_outputs = h_0 - y
        vectorized_gradient = x_t.dot(diff_outputs)
        reg_term = np.copy(tmp_thetas) * self.lambda_
        reg_term[0] = 0
        return (1 / m) * (vectorized_gradient + reg_term)

    def fit_(self):
        pass


if __name__ == '__main__':
    thetas = np.array([1.0, 2.0, 3.0])
    model_1 = MyRidge(thetas=thetas)
    print(model_1.get_params_())
    print(model_1.set_params_(
        {'thetas': np.array([42.0, 2.0, 3.0]), 'max_iter': 42}))
    print(model_1.get_params_())
