import pytest
import numpy as np
from loss import loss_

y_ = [
    np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1)),
    np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1)),
]

y_hat = [
    np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1)),
    np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1)),

]

invalid_inputs = [
    [],
    np.array([]),
    42,
    np.array([1, 2, 3, 4, 5]),
    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
]

e_loss_ = [
    2.142857142857143,
    0.0,
]

valid_loss_ = list(zip(y_, y_hat, e_loss_))


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
