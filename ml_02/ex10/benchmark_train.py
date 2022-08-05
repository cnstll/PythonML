import csv
import os

import numpy as np
import pandas as pd
from data_spliter import data_spliter
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features as add_pf


# Save model parameters
def save_parameters(param):
    filename = 'models.csv'
    with open(filename, 'a') as f:
        w = csv.DictWriter(f, param.keys())
        if os.path.getsize(filename) < 1:
            w.writeheader()
        w.writerow(param)


# Training and testing models functions
def set_model(index, theta, alpha=None, max_iter=None, desc=None):
    initial_param = {'index': index, 'model_description': desc,
                     'init_theta': theta,
                     'alpha': alpha, 'max_iter': max_iter,
                     }
    mlr = MLR(theta, alpha, int(max_iter))
    return initial_param, mlr


def train_model(x_train, y_train, mlr):
    grad = mlr.fit_(x_train, y_train)
    return mlr.thetas, grad


def test_model(x_test, y_test, mlr):
    y_hat = mlr.predict_(x_test)
    mse = mlr.mse_(y_test, y_hat)
    return y_hat, mse


# Extracting data and specifying that the first column is an index col
data = pd.read_csv('space_avocado.csv', index_col=0)
# print(data)

# Normalization of data around 0 (feature scaling)
data = (data - data.mean()) / data.std()
# print(data)
x = data.iloc[:, 0:3].to_numpy()
y = data.iloc[:, 3:4].to_numpy()
# Splitting data into training dataset and testing dataset
# Distribution ratio is 0.8 for training, the rest for testing
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Models trained and tested
# Univariate model weight to target
x1_train = x_train[:, 0].reshape(-1, 1)
x1_test = x_test[:, 0].reshape(-1, 1)
thetas = np.array([1.0, 1.0]).reshape(-1, 1)
param1, mlr1 = set_model(1, thetas, 1.5e-5, 3e5, 'Univariate model weight')
theta_trained, grad = train_model(x1_train, y_train, mlr1)
# plot_gradient(grad)
y_hat1, mse1 = test_model(x1_test, y_test, mlr1)
# plot_regression(x1_test, y_test, y_hat1, settings)
param1['theta_trained'] = theta_trained
param1['mse'] = mse1
save_parameters(param1)


# Univariate model prod_distance to target
x2_train = x_train[:, 1].reshape(-1, 1)
x2_test = x_test[:, 1].reshape(-1, 1)
thetas = np.array([1.0, 1.0]).reshape(-1, 1)
param2, mlr2 = set_model(2, thetas, 1.5e-5, 3e5,
                         'Univariate model prod_distance')
theta_trained, grad = train_model(x2_train, y_train, mlr2)
# plot_gradient(grad)
y_hat2, mse2 = test_model(x2_test, y_test, mlr2)
# plot_regression(x2_test, y_test, y_hat2, settings)
param2['theta_trained'] = theta_trained
param2['mse'] = mse2
save_parameters(param2)


# Univariate model time_delivery to target
x3_train = x_train[:, 2].reshape(-1, 1)
x3_test = x_test[:, 2].reshape(-1, 1)
thetas = np.array([1.0, 1.0]).reshape(-1, 1)
param3, mlr3 = set_model(3, thetas, 1.5e-5, 3e5,
                         'Univariate model time_delivery')
theta_trained, grad = train_model(x3_train, y_train, mlr3)
# plot_gradient(grad)
y_hat3, mse3 = test_model(x3_test, y_test, mlr3)
# plot_regression(x3_test, y_test, y_hat3, settings)
param3['theta_trained'] = theta_trained
param3['mse'] = mse3
save_parameters(param3)

# Polynomial(2) model weight to target
x4_train = x_train[:, 0].reshape(-1, 1)
x4_test = x_test[:, 0].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0]).reshape(-1, 1)
param4, mlr4 = set_model(4, thetas, 1.5e-5, 3e5, 'Polynomial(2) model weight')
theta_trained, grad = train_model(add_pf(x4_train, 2), y_train, mlr4)
# plot_gradient(grad)
y_hat4, mse4 = test_model(add_pf(x4_test, 2), y_test, mlr4)
# plot_regression(x4_test, y_test, y_hat4, settings)
param4['theta_trained'] = theta_trained
param4['mse'] = mse4
save_parameters(param4)

# Polynomial(2) model prod_dist to target
x5_train = x_train[:, 1].reshape(-1, 1)
x5_test = x_test[:, 1].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0]).reshape(-1, 1)
param5, mlr5 = set_model(5, thetas, 1.5e-5, 3e5,
                         'Polynomial(2) model prod_dist')
theta_trained, grad = train_model(add_pf(x5_train, 2), y_train, mlr5)
# plot_gradient(grad)
y_hat5, mse5 = test_model(add_pf(x5_test, 2), y_test, mlr5)
# plot_regression(x5_test, y_test, y_hat5, settings)
param5['theta_trained'] = theta_trained
param5['mse'] = mse5
save_parameters(param5)

# Polynomial(2) model time_delivery to target
x6_train = x_train[:, 2].reshape(-1, 1)
x6_test = x_test[:, 2].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0]).reshape(-1, 1)
param6, mlr6 = set_model(6, thetas, 1.5e-5, 3e5,
                         'Polynomial(2) model time_delivery')
theta_trained, grad = train_model(add_pf(x6_train, 2), y_train, mlr6)
# plot_gradient(grad)
y_hat6, mse6 = test_model(add_pf(x6_test, 2), y_test, mlr6)
# plot_regression(x6_test, y_test, y_hat6, settings)
param6['theta_trained'] = theta_trained
param6['mse'] = mse6
save_parameters(param6)

# Polynomial(3) model weight to target
x7_train = x_train[:, 0].reshape(-1, 1)
x7_test = x_test[:, 0].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
param7, mlr7 = set_model(7, thetas, 1.5e-5, 3e5,
                         'Polynomial(3) model weight')
theta_trained, grad = train_model(add_pf(x7_train, 3), y_train, mlr7)
# plot_gradient(grad)
y_hat7, mse7 = test_model(add_pf(x7_test, 3), y_test, mlr7)
# plot_regression(x7_test, y_test, y_hat7, settings)
param7['theta_trained'] = theta_trained
param7['mse'] = mse7
save_parameters(param7)

# Polynomial(3) model prod_distance to target
x8_train = x_train[:, 1].reshape(-1, 1)
x8_test = x_test[:, 1].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
param8, mlr8 = set_model(8, thetas, 1.5e-5, 3e5,
                         'Polynomial(3) model prod_dist')
theta_trained, grad = train_model(add_pf(x8_train, 3), y_train, mlr8)
# plot_gradient(grad)
y_hat8, mse8 = test_model(add_pf(x8_test, 3), y_test, mlr8)
# plot_regression(x8_test, y_test, y_hat8, settings)
param8['theta_trained'] = theta_trained
param8['mse'] = mse8
save_parameters(param8)

# Polynomial(3) model prod_distance to target
x9_train = x_train[:, 2].reshape(-1, 1)
x9_test = x_test[:, 2].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
param9, mlr9 = set_model(9, thetas, 1.5e-5, 3e5,
                         'Polynomial(3) model time_delivery')
theta_trained, grad = train_model(add_pf(x9_train, 3), y_train, mlr9)
# plot_gradient(grad)
y_hat9, mse9 = test_model(add_pf(x9_test, 3), y_test, mlr9)
# plot_regression(x9_test, y_test, y_hat9, settings)
param9['theta_trained'] = theta_trained
param9['mse'] = mse9
save_parameters(param9)

# Polynomial(4) model weight to target
x10_train = x_train[:, 0].reshape(-1, 1)
x10_test = x_test[:, 0].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
param10, mlr10 = set_model(10, thetas, 1.5e-5, 3e5,
                           'Polynomial(4) model weight')
theta_trained, grad = train_model(add_pf(x10_train, 4), y_train, mlr10)
# plot_gradient(grad)
y_hat10, mse10 = test_model(add_pf(x10_test, 4), y_test, mlr10)
# plot_regression(x10_test, y_test, y_hat10, settings)
param10['theta_trained'] = theta_trained
param10['mse'] = mse10
save_parameters(param10)

# Polynomial(4) model prod_distance to target
x11_train = x_train[:, 1].reshape(-1, 1)
x11_test = x_test[:, 1].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
param11, mlr11 = set_model(11, thetas, 1.5e-5, 3e5,
                           'Polynomial(4) model prod_dist')
theta_trained, grad = train_model(add_pf(x11_train, 4), y_train, mlr11)
# plot_gradient(grad)
y_hat11, mse11 = test_model(add_pf(x11_test, 4), y_test, mlr11)
# plot_regression(x11_test, y_test, y_hat11, settings)
param11['theta_trained'] = theta_trained
param11['mse'] = mse11
save_parameters(param11)

# Polynomial(4) model prod_distance to target
x12_train = x_train[:, 2].reshape(-1, 1)
x12_test = x_test[:, 2].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
param12, mlr12 = set_model(12, thetas, 1.5e-5, 3e5,
                           'Polynomial(4) model time_delivery')
theta_trained, grad = train_model(add_pf(x12_train, 4), y_train, mlr12)
# plot_gradient(grad)
y_hat12, mse12 = test_model(add_pf(x12_test, 4), y_test, mlr12)
# plot_regression(x12_test, y_test, y_hat12, settings)
param12['theta_trained'] = theta_trained
param12['mse'] = mse12
save_parameters(param12)
