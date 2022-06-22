from statistics import mean
from TinyStatistician import TinyStatistician as ts
import pytest
import numpy as np


class Test_valid_inputs:

    def test_mean_with_list(self):
        x = [1, 2, 3]
        res = ts.mean(x)
        assert res == mean(x)


    def test_mean_with_ndarray(self):
        x = np.array([1, 2, 3])
        res = ts.mean(x)
        assert res == mean(x)


class Test_errors:

    def test_mean_input_type_error(self):
        x = 42
        res = ts.mean(x)
        assert res is None

    def test_mean_with_empty_list(self):
        x = []
        msg = 'mean requires at least one data point'
        with pytest.raises(RuntimeError, match=msg) as e:
            res = ts.mean(x)
        assert e.type is RuntimeError
        assert str(e.value) == msg

    def test_mean_with_empty_ndarray(self):
        x = np.array([])
        msg = 'mean requires at least one data point'
        with pytest.raises(RuntimeError, match=msg) as e:
            res = ts.mean(x)
        assert e.type is RuntimeError
        assert str(e.value) == msg

