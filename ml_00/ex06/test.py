import pytest
import numpy as np
from loss import loss_, loss_elem_, predict_

theta_inputs = [
    np.array([[2.], [4.]]),
    np.array([[0.05], [1.], [1.], [1.]]),
    np.array([[0.], [1.]]),
]

x_ = [
    np.array([[0.], [1.], [2.], [3.], [4.]]),
    np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]]),
    np.array([[0], [15], [-9], [7], [12], [3], [-21]]),
]

y_ = [
    np.array([[2.], [7.], [12.], [17.], [22.]]),
    np.array([[19.], [42.], [67.], [93.]]),
    np.array([[2.], [14.], [-13.], [5.], [12.], [4.], [-19.]]),
    np.array([[2.], [14.], [-13.], [5.], [12.], [4.], [-19.]]),
]

y_hat = [predict_(x, th) for x, th in zip(x_, theta_inputs)]
y_hat.append(np.array([[2.], [14.], [-13.], [5.], [12.], [4.], [-19.]]))

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
    np.array([[0.], [0.], [0.], [0.], [0.], [0.], [0.]]),
]

e_loss_ = [
    3.0,
    4.238750000000004,
    2.142857142857143,
    0.0
]

valid_loss_elem = list(zip(y_, y_hat, e_loss_elem))
valid_loss_ = list(zip(y_, y_hat, e_loss_))


@pytest.mark.parametrize("y, y_hat, e", valid_loss_elem)
def test_loss_elem(y, y_hat, e):
    np.testing.assert_array_almost_equal(loss_elem_(y, y_hat), e)


@pytest.mark.parametrize("y, y_hat, e", valid_loss_)
def test_loss_(y, y_hat, e):
    np.testing.assert_array_almost_equal(loss_(y, y_hat), e)


# invalid inputs
@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("y", y_)
def test_loss_error_first_param(invalid, y):
    assert loss_(y, invalid) is None


@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("y", y_)
def test_loss_error_secnd_param(invalid, y):
    assert loss_(invalid, y) is None


@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("y", y_)
def test_loss_elem_error_first_param(invalid, y):
    assert loss_elem_(y, invalid) is None


@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("y", y_)
def test_loss_elem_error_first_param(invalid, y):
    assert loss_elem_(invalid, y) is None
