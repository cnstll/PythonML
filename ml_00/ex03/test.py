import pytest
import numpy as np
from tools import add_intercept

x_inputs = [
    np.arange(1, 6).reshape((5, 1)),
    np.arange(1, 10).reshape((3, 3)),
]

invalid_inputs = [
    [],
    np.array([]),
    42,
    np.array([[]]),
]

expected = [
    np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.]]),
    np.array([[1., 1., 2., 3.], [1., 4., 5., 6.], [1., 7., 8., 9.]]),
]


# input tests
@pytest.mark.parametrize("x, e", list(zip(x_inputs, expected)))
def test_add_intercept(x, e):
    np.testing.assert_array_equal(add_intercept(x), e)


# invalid inputs
@pytest.mark.parametrize("invalid", invalid_inputs)
def test_add_intercept_errors(invalid):
    assert add_intercept(invalid) is None
