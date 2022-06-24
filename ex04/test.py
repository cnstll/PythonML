import pytest
import numpy as np
from prediction import predict_

x = np.arange(1, 6).reshape(-1, 1)

theta_inputs = [
    np.array([[5], [0]]),
    np.array([[0], [1]]),
    np.array([[5], [3]]),
    np.array([[-3], [1]]),
]

x_inputs = [x for i in range(0, len(theta_inputs))]

invalid_inputs = [
    [],
    np.array([]),
    42,
    np.array([1, 2, 3, 4, 5]),
    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
]

invalid_thetas = [
    np.array([1, 2, 3, 4, 5]),
    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
    42,
    [[1], [2]],
    np.array([[], []]),
]

expected = [
    np.array([[5.], [5.], [5.], [5.], [5.]]),
    np.array([[1.], [2.], [3.], [4.], [5.]]),
    np.array([[8.], [11.], [14.], [17.], [20.]]),
    np.array([[-2.], [-1.], [0.], [1.], [2.]]),
]

tested_valid = list(zip(x_inputs, theta_inputs, expected))
# input tests


@pytest.mark.parametrize("x, thetas, e", tested_valid)
def test_predict(x, thetas, e):
    print(x, thetas, e)
    np.testing.assert_array_equal(predict_(x, thetas), e)


# invalid inputs
@pytest.mark.parametrize("invalid", invalid_inputs)
@pytest.mark.parametrize("thetas", theta_inputs)
def test_predict_error(invalid, thetas):
    assert predict_(invalid, thetas) is None


# invalid thetas
@pytest.mark.parametrize("x", x_inputs)
@pytest.mark.parametrize("invalid", invalid_thetas)
def test_mean_theta_error(x, invalid):
    assert predict_(x, invalid) is None
