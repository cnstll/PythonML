from venv import create
import pytest
import numpy as np
from mylinearregression import MyLinearRegression as MyLR


class TestLinearRegressionclass:
    @pytest.fixture
    def create_mylr(self):
        return MyLR([[1.], [1.], [1.], [1.], [1]])

    def test_predict(self, create_mylr):
        mylr = create_mylr
        x = np.array([[1., 1., 2., 3.],
                      [5., 8., 13., 21.], [34., 55., 89., 144.]])
        e = np.array([[8.], [48.], [323.]])
        np.testing.assert_array_almost_equal(mylr.predict_(x), e)

    def test_loss_elem(self, create_mylr):
        mylr = create_mylr
        x = np.array([[1., 1., 2., 3.],
                      [5., 8., 13., 21.], [34., 55., 89., 144.]])
        y = np.array([[23.], [48.], [218.]])
        e = np.array([[225.], [0.], [11025.]])
        lr = mylr.predict_(x)
        np.testing.assert_array_almost_equal(mylr.loss_elem_(lr, y), e)

    def test_loss(self, create_mylr):
        mylr = create_mylr
        x = np.array([[1., 1., 2., 3.],
                      [5., 8., 13., 21.], [34., 55., 89., 144.]])
        y = np.array([[23.], [48.], [218.]])
        e = 1875.0
        lr = mylr.predict_(x)
        np.testing.assert_array_almost_equal(mylr.loss_(lr, y), e)

    def test_fit(self, create_mylr):
        mylr = create_mylr
        x = np.array([[1., 1., 2., 3.],
                      [5., 8., 13., 21.], [34., 55., 89., 144.]])
        y = np.array([[23.], [48.], [218.]])
        mylr.alpha = 1.6e-4
        mylr.max_iter = 200000
        e_th = np.array([[18.1883807], [2.76697280],
                         [-0.37477892], [1.39219391], [0.0174150184]])
        e_y_hat = np.array([[23.417207], [47.48925], [218.065638]])
        e_loss_elem = np.array([[0.174062], [0.260866], [0.004308]])
        e_loss = 0.073206
        mylr.fit_(x, y)
        y_hat = mylr.predict_(x)
        loss_e = mylr.loss_elem_(y, y_hat)
        loss_ = mylr.loss_(y, y_hat)
        np.testing.assert_array_almost_equal(mylr.thetas, e_th)
        np.testing.assert_array_almost_equal(y_hat, e_y_hat)
        np.testing.assert_array_almost_equal(loss_e, e_loss_elem)
        np.testing.assert_array_almost_equal(loss_, e_loss)
