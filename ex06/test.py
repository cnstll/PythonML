import pytest
import numpy as np
from loss import loss_, loss_elem_, predict_

theta_inputs = [
    np.array([[5], [0]]),
    np.array([[0], [1]]),
    np.array([[5], [3]]),
    np.array([[-3], [1]]),
]

x_ = [
    np.array([[0.], [1.], [2.], [3.], [4.]]),
    np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]]),
    np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]]),
]

y_ = [
    np.array([[2.], [7.], [12.], [17.], [22.]]),
    np.array([[19.], [42.], [67.], [93.]]),
    np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]]),
]

y_hat = [predict_(y, x) for y, x in zip(y_, x_)]

invalid_inputs = [
    [],
    np.array([]),
    42,
    np.array([1, 2, 3, 4, 5]),
    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
]

e_loss_elem = [
    np.array([[0.], [1.], [4.], [9.], [16.]]),
    np.array([[10.5625], [6.0025], [0.1225], [17.2225]]),
    np.array([[4.], [1.], [16.], [4.], [0.], [1.], [4.]]),
]

e_loss_ = [
    3.0,
    4.238750000000004,
    2.142857142857143,
]

valid_loss_elem = list(zip(y_, y_hat, e_loss_elem))
# input tests


@pytest.mark.parametrize("y_, y_hat, e", valid_loss_elem)
def test_loss_elem(y_, y_hat, e):
    print(y_, y_hat, e)
    np.testing.assert_array_equal(loss_elem_(y_, y_hat), e)


# invalid inputs
# @pytest.mark.parametrize("invalid", invalid_inputs)
# @pytest.mark.parametrize("thetas", theta_inputs)
# def test_predict_error(invalid, thetas):
#     assert predict_(invalid, thetas) is None


# # invalid thetas
# @pytest.mark.parametrize("x", x_inputs)
# @pytest.mark.parametrize("invalid", invalid_thetas)
# def test_mean_theta_error(x, invalid):
#     assert predict_(x, invalid) is None
