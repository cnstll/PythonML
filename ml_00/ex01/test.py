from statistics import mean, median, stdev, variance
from TinyStatistician import TinyStatistician as ts
import pytest
import numpy as np


valid_inputs = [
    [1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 7.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    np.array([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 7.0]),
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
]

invalid_inputs = [
    [],
    np.array([]),
    42,
]

percentiles = [1, 25, 50, 75, 42, 99, 100, ]

oob_percentiles = [0, -1, 101, 142, ]


# Tests for mean
@pytest.mark.parametrize('lst', valid_inputs)
def test_mean(lst):
    res = ts.mean(lst)
    assert res == mean(lst)


@pytest.mark.parametrize('invalid_el', invalid_inputs)
def test_mean_input_type_error(invalid_el):
    res = ts.mean(invalid_el)
    assert res is None


# Tests for median
@pytest.mark.parametrize('lst', valid_inputs)
def test_median(lst):
    res = ts.median(lst)
    assert res == median(lst)


@pytest.mark.parametrize('invalid_el', invalid_inputs)
def test_median_input_type_error(invalid_el):
    res = ts.median(invalid_el)
    assert res is None


# Tests for quartiles
@pytest.mark.parametrize('lst', valid_inputs)
def test_quartile(lst):
    res = ts.quartile(lst)
    assert res == (np.percentile(lst, 25), np.percentile(lst, 75))


@pytest.mark.parametrize('invalid_el', invalid_inputs)
def test_quartile_input_type_error(invalid_el):
    res = ts.quartile(invalid_el)
    assert res is None


# Tests for percentile
@pytest.mark.parametrize('lst', valid_inputs)
@pytest.mark.parametrize('p', percentiles)
def test_percentile(lst, p):
    res = ts.percentile(lst, p)
    assert res == np.percentile(lst, p)


@pytest.mark.parametrize('invalid_el', invalid_inputs)
@pytest.mark.parametrize('p', percentiles)
def test_percentile_input_value_error(invalid_el, p):
    res = ts.percentile(invalid_el, p)
    assert res is None


@pytest.mark.parametrize('lst', valid_inputs)
@pytest.mark.parametrize('p', oob_percentiles)
def test_percentile_input_percentile_error(lst, p):
    res = ts.percentile(lst, p)
    assert res is None


# Tests for var
@pytest.mark.parametrize('lst', valid_inputs)
def test_var(lst):
    res = ts.var(lst)
    assert res == variance(lst, mean(lst))


@pytest.mark.parametrize('invalid_el', invalid_inputs)
def test_var_input_type_error(invalid_el):
    res = ts.var(invalid_el)
    assert res is None


# Tests for std
@pytest.mark.parametrize('lst', valid_inputs)
def test_std(lst):
    res = ts.std(lst)
    assert res == stdev(lst, mean(lst))


@pytest.mark.parametrize('invalid_el', invalid_inputs)
def test_std_input_type_error(invalid_el):
    res = ts.std(invalid_el)
    assert res is None
