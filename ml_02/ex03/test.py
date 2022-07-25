import pytest
import numpy as np
from gradient import gradient

x_ = [
    np.array([
             [-6, -7, -9],
             [13, -2, 14],
             [-7, 14, -1],
             [-8, -4, 6],
             [-5, -9, 6],
             [1, -5, 11],
             [9, -11, 8]]),
    np.array([
             [-6, -7, -9],
             [13, -2, 14],
             [-7, 14, -1],
             [-8, -4, 6],
             [-5, -9, 6],
             [1, -5, 11],
             [9, -11, 8]]),
]

y_ = [
    np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1)),
    np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1)),
]

theta_ = [
    np.array([0, 3, 0.5, -6]).reshape((-1, 1)),
    np.array([0, 0, 0, 0]).reshape((-1, 1)),
]

invalid_inputs = [
    [],
    np.array([2, 14, -13, 5, 12, 4]).reshape((-1, 1)),
    np.array([]),
    np.array([3, 0.5, -6]).reshape((-1, 1)),
    42,
    np.array([1, 2, 3, 4, 5]),
    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
    np.array([
             [-6, -7, -9],
             [-7, 14, -1],
             [-8, -4, 6],
             [-5, -9, 6],
             [1, -5, 11],
             [9, -11, 8]]),
]

e = [
    np.array([[-33.71428571], [-37.35714286], [183.14285714], [-393.]]),
    np.array([[-0.71428571], [0.85714286], [23.28571429], [-26.42857143]]),
]

valid_inputs = list(zip(x_, y_, theta_, e))


@pytest.mark.parametrize("x, y, theta, e", valid_inputs)
def test_gradient_(x, y, theta, e):
    np.testing.assert_array_almost_equal(gradient(x, y, theta), e)


# invalid inputs
@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("x, y, theta, e", valid_inputs)
def test_gradient_error_first_param(invalid, x, y, theta, e):
    assert gradient(x, invalid, theta) is None


@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("x, y, theta, e", valid_inputs)
def test_gradient_error_secnd_param(invalid, x, y, theta, e):
    assert gradient(invalid, y, theta) is None


@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("x, y, theta, e", valid_inputs)
def test_gradient_error_third_param(invalid, x, y, theta, e):
    assert gradient(x, y, invalid_inputs) is None
