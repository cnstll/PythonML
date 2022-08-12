import os
import sys
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def extract_data():
    """Extract data from both csv available and join

    Returns:
        DataFrame: panda dataframe with left join data
        from both csv files available
    """
    abs_path = os.path.dirname(__file__)
    features_csv = 'solar_system_census.csv'
    zip_csv = 'solar_system_census_planets.csv'
    try:
        features_df = pd.read_csv(os.path.join(abs_path, features_csv),
                                  index_col=0)
        zip_df = pd.read_csv(os.path.join(abs_path, zip_csv), index_col=0)
        return features_df.join(zip_df)
    except Exception:
        return None


def preprocess_data(data):
    # Split extracted data into x and y with y the Origin (zipcode)
    # of the citizens
    y = data.iloc[:, 3:4].to_numpy()
    x = data.iloc[:, 0:3].to_numpy()
    return data_spliter(x, y, 0.8)


def binarize_matrix(dataset_outputs, zip):
    # Replace non zip values by -1
    for y in dataset_outputs:
        y[(y != zip)] = -1
        y[(y == zip)] = 1
        y[(y == -1)] = 0
    return dataset_outputs


def prep_and_train_model(x_train, y_train, x_test, y_test, zip):
    y_train, y_test = binarize_matrix([y_train, y_test], zip)
    thetas = np.ones((x_train.shape[1] + 1, 1))
    alpha = 1.25e-3
    max_iter = int(4e5)
    my_mlr = MLR(thetas, alpha, max_iter)
    my_mlr.fit_(x_train, y_train)
    y_hat = my_mlr.predict_(x_test)
    show_model_perf(my_mlr, y_test, y_hat, f"OvR for zip {zip}")
    return y_hat


def compute_accuracy(y, y_hat):
    accuracy = 0.0
    total = y.shape[0]
    diff = np.power(y - y_hat, 2)
    error = sum(diff.flatten())
    accuracy = 1 - error / total
    return accuracy


def show_model_perf(mlr, y_test, y_hat, label):
    print(f"{label:-^30}")
    print('loss: ', mlr.loss_(y_test, y_hat))
    print('acc: ', compute_accuracy(y_test, y_hat))


def plot_log_reg(x_test, y_test, y_hat):
    df = pd.DataFrame(x_test, columns=['weight', 'height', 'bone_density'])
    df = df.join(pd.DataFrame(y_hat, columns=['predicted_zip']))
    df = df.join(pd.DataFrame(y_test, columns=['real_zip']))
    sns.pairplot(x_vars=['weight', 'height', 'bone_density'],
                 y_vars=['real_zip', 'predicted_zip'], data=df)
    plt.show()


def main():
    data = extract_data()
    x_train, x_test, y_train, y_test = preprocess_data(data)
    # copying to have a deep copy for each model
    y_hat_0 = prep_and_train_model(x_train, y_train.copy(),
                                   x_test, y_test.copy(), 0)
    y_hat_1 = prep_and_train_model(x_train, y_train.copy(),
                                   x_test, y_test.copy(), 1)
    y_hat_2 = prep_and_train_model(x_train, y_train.copy(),
                                   x_test, y_test.copy(), 2)
    y_hat_3 = prep_and_train_model(x_train, y_train.copy(),
                                   x_test, y_test.copy(), 3)

    # Stacking all one-vs-rest predictions to find the max probability
    # for each training sample
    clustered_y_hat = np.array([y_hat_0, y_hat_1, y_hat_2, y_hat_3])
    y_hat_max = np.argmax(clustered_y_hat, axis=0)

    plot_log_reg(x_test, y_test, y_hat_max)


if __name__ == '__main__':
    main()
