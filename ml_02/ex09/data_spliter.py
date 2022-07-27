from math import ceil

import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)
    into a training and a test set,
    while respecting the given proportion of examples to
    be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset
    that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    rng = np.random.default_rng(12345)
    p = rng.permutation(x.shape[0])
    nb_training_rows = ceil(x.shape[0] * proportion)
    x = x[p]
    y = y[p]
    print(x, y)
    x_train = x[0:nb_training_rows, :]
    x_test = x[nb_training_rows:, :]
    y_train = y[0:nb_training_rows, :]
    y_test = y[nb_training_rows:, :]
    return (x_train, x_test, y_train, y_test)


x = np.array([[100, 110, 120], [200, 210, 220], [300, 310, 320]])
y = np.array([1, 2, 3]).reshape(-1, 1)

x2 = np.array([[100, 110, 120], [200, 210, 220],
               [300, 310, 320], [400, 410, 420]])
y2 = np.array([1, 2, 3, 4]).reshape(-1, 1)

print(data_spliter(x, y, 0.25))
print(data_spliter(x2, y2, 0.5))
