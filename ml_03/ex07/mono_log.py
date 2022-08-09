import os
import sys
from math import ceil

import numpy as np
import pandas as pd

# Importing logistic prediction function from ex01
dir_executed_file = os.path.dirname(__file__)
file_path = os.path.join(dir_executed_file, '..', 'ex06')
sys.path.insert(0, file_path)
from my_logistic_regression import MyLogisticRegression as MLR  # Noqa


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
    x_train = x[0:nb_training_rows, :]
    x_test = x[nb_training_rows:, :]
    y_train = y[0:nb_training_rows, :]
    y_test = y[nb_training_rows:, :]
    return (x_train, x_test, y_train, y_test)


def parse_arguments():
    """Parse and extract zipcode from program arguments

    Returns:
        zip: float in valid zipcode values, else None
    """
    zipcode_param = '–zipcode='
    valid_zip = ['0', '1', '2', '3']
    for arg in sys.argv:
        if arg.find(zipcode_param) != -1:
            zip = arg[len(zipcode_param):]
            if zip not in valid_zip:
                return None
            else:
                return float(zip)
    return None


def extract_data():
    """Extract data from both csv available

    Returns:
        DataFrame: panda dataframe with left join data
        from both csv files available
    """
    abs_path = os.path.dirname(__file__)
    features_csv = 'solar_system_census.csv'
    zip_csv = 'solar_system_census_planets.csv'
    features_df = pd.read_csv(os.path.join(abs_path, features_csv),
                              index_col=0)
    zip_df = pd.read_csv(os.path.join(abs_path, zip_csv), index_col=0)
    return features_df.join(zip_df)


def main():
    zip = parse_arguments()
    if zip is None:
        print("Usage: python mono_log.py –zipcode=[valid_zip]")
        sys.exit()
    data = extract_data()
    # Split extracted data into x and y with y the Origin (zipcode)
    # of the citizens
    y = data.iloc[:, 3:4].to_numpy()
    x = data.iloc[:, 0:3].to_numpy()
    # Replace non zip values by -1
    y[(y != zip)] = -1
    y[(y == zip)] = 1
    y[(y == -1)] = 0
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
    thetas = np.ones((x.shape[1] + 1, 1))
    alpha = 1.e-3
    max_iter = int(1e5)
    my_mlr = MLR(thetas, alpha, max_iter)
    my_mlr.fit_(x_train, y_train)
    y_hat = my_mlr.predict_(x_test)
    print(my_mlr.loss_(y_test, y_hat))


if __name__ == '__main__':
    main()
