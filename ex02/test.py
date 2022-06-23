import
import pytest
import numpy as np


x_inputs = [42, 10, -24, 52, 32, 142092, -54, 0, -1,]
theta_inputs = [

]
invalid_inputs = [
    [],
    np.array([]),
    42,
]

# Tests for mean
@pytest.mark.parametrize('lst', valid_inputs)
def test_mean(lst):
    pass

# @pytest.mark.parametrize('invalid_el', invalid_inputs)
# def test_mean_input_type_error(invalid_el):
#     pass