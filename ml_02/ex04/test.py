import pytest
import numpy as np
from fit import fit_

x_ = [
    np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]]),
]

y_ = [
    np.array([[19.6], [-2.8], [-25.2], [-47.6]]),
]

theta_ = [
    np.array([[42.], [1.], [1.], [1.]]),
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
    np.array(([[41.998888], [0.977923], [0.779232], [-1.207684]])),
]

valid_inputs = list(zip(x_, y_, theta_, e))


@pytest.mark.parametrize("x, y, theta, e", valid_inputs)
def test_fit(x, y, theta, e):
    np.testing.assert_array_almost_equal(fit_(x, y, theta, 0.0005, 42000), e)


# invalid inputs
# @pytest.mark.parametrize("invalid", invalid_inputs)
# @pytest.mark.parametrize("x, y, theta, e", valid_inputs)
# def test_fiterror_first_param(invalid, x, y, theta, e):
    # assert fit_(x, invalid, theta) is None
# 
# 
# @pytest.mark.parametrize("invalid", invalid_inputs)
# @pytest.mark.parametrize("x, y, theta, e", valid_inputs)
# def test_fiterror_secnd_param(invalid, x, y, theta, e):
    # assert fit_(invalid, y, theta) is None
# 
# 
# @pytest.mark.parametrize("invalid", invalid_inputs)
# @pytest.mark.parametrize("x, y, theta, e", valid_inputs)
# def test_fiterror_third_param(invalid, x, y, theta, e):
    # assert fit_(x, y, invalid_inputs) is None
# 