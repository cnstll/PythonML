import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_spliter import data_spliter
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features as add_pf


# Exctract model parameters
def extract_from_parameters(label):
    filename = 'models.csv'
    p = pd.read_csv(filename)
    return p[label]


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


# Plotting functions
def plot_initial_data(data):
    fig, axs = plt.subplots(1, 3)
    fig.set_title("Gradients for each parameter of the model")
    axs[0].scatter(data['weight'], data['target'])
    axs[1].scatter(data['prod_distance'], data['target'])
    axs[2].scatter(data['time_delivery'], data['target'])
    plt.show()


def plot_regression(x, y, y_hat, settings):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', color=settings['clr_real'], label='Target')
    ax.plot(x, y_hat, 'o', color=settings['clr_pred'],
            label='Predicted target', markersize=2)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.legend(bbox_to_anchor=settings['anchor'],
              loc=settings['loc'], frameon=True, ncol=1)
    plt.show()


def plot_gradient(gradient):
    n_cols = 2
    n_rows = int(gradient.shape[1] / n_cols)
    win_title = 'Evolution of gradients for each parameter'
    fig, axs = plt.subplots(n_rows, n_cols, label=win_title)
    x = np.arange(0, len(gradient), 1).reshape(-1, 1)
    for i in range(0, gradient.shape[1]):
        axs[i].plot(x, gradient[:, i], 'o', label=f"theta_{i}", ms=1)
        axs[i].grid(color='lightgray', linestyle='-', linewidth=1)
        axs[i].legend()
    plt.show()


def plot_mse(mse_values):
    fig, ax = plt.subplots(label='MSE for each model')
    ax.bar(np.arange(1, len(mse_values) + 1, 1), mse_values,
           color='#3A89FF', edgecolor='gray')
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=3)
    ax.xaxis.set_ticks(np.arange(1, len(mse_values) + 1))
    for p in ax.patches:
        w = p.get_width()
        h = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f"{h:.2f}", ((x + w / 2), y + h * 1.02), ha='center',
                    fontweight='demi')
    plt.show()


# Settings for plotting
settings = {'clr_real': '#0310BF', 'clr_pred': '#3A89FF',
            'anchor': (0, 1),
            'loc': 'upper left',
            }


# Extracting data and specifying that the first column is an index col
data = pd.read_csv('space_avocado.csv', index_col=0)
# print(data)

# Normalization of data around 0 (feature scaling)
data = (data - data.mean()) / data.std()

# Splitting data into training dataset and testing dataset
# Distribution ratio is 0.8 for training, the rest for testing
x = data.iloc[:, 0:3].to_numpy()
y = data.iloc[:, 3:4].to_numpy()
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

# Plotting Benchmark
plot_mse(extract_from_parameters('mse'))

# Best model trained and tested
# Polynomial(2) model weight to target
x4_train = x_train[:, 0].reshape(-1, 1)
x4_test = x_test[:, 0].reshape(-1, 1)
thetas = np.array([1.0, 1.0, 1.0]).reshape(-1, 1)
param4, mlr4 = set_model(4, thetas, 1.5e-5, 3e5, 'Polynomial(2) model weight')
theta_trained, grad = train_model(add_pf(x4_train, 2), y_train, mlr4)
# plot_gradient(grad)
y_hat4, mse4 = test_model(add_pf(x4_test, 2), y_test, mlr4)
plot_regression(x4_test, y_test, y_hat4, settings)
param4['theta_trained'] = theta_trained
param4['mse'] = mse4
