import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mylinearregression import MyLinearRegression as MyLR


def plot_points(x, y):
    """Plot the data from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    plt.show()


def plot_loss(x, y, mlr):
    """Plot the different loss function J in function of theta1
        for different theta0.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    mlr: has to be an instance of MyLinearRegression.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    theta_one_range = np.arange(-14.0, -4.0, 0.01)
    theta_zero_range = np.arange(80.0, 100.0, 3.0)
    fig, ax = plt.subplots()
    greys = cm.get_cmap('Greys', len(theta_zero_range))
    ax.set_ylim(10, 120)
    ax.set_xlabel('$θ_1$')
    ax.set_ylabel('Loss function J($θ_1$,$θ_2$)')
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    for tone, th_0 in enumerate(theta_zero_range):
        mlr.thetas[0] = th_0
        j_list = []
        color = greys(len(theta_zero_range) - tone)
        for th_1 in theta_one_range:
            mlr.thetas[1] = th_1
            y_hat = mlr.predict_(x)
            j_ = mlr.loss_(y, y_hat)
            j_list.append(j_)
            label = f'J($θ_0$=$c_{tone}$,$θ_1$)'
        ax.plot(theta_one_range, j_list, '-', color=color, label=label)
    ax.legend(loc='lower right', fontsize='small')
    plt.show()


def plot_prediction(x, y, y_hat, settings):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    mlr: has to be an instance of MyLinearRegression.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', color=settings['clr_real'], label='Sell price')
    ax.plot(x, y_hat, 'o', color=settings['clr_pred'],
            label='Predicted sell price', markersize=2)
    ax.set_xlabel(settings['x_label'])
    ax.set_ylabel(settings['y_label'])
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.legend(bbox_to_anchor=settings['anchor'],
              loc=settings['loc'], frameon=True, ncol=1)
    plt.show()


def print_param_and_error(myLR, field, y_hat, title=None):
    if title is not None:
        print(f"{title:-^30}")
    print(f"theta_{field.lower()}: {myLR.thetas}")
    print(f"mse_{field.lower()}: {myLR.mse_(y_hat, y_sell_price)}")
    print()


def training_univariate_model(myLR, data):
    x = np.array(data).reshape(-1, 1)
    myLR.fit_(x, y_sell_price)
    return myLR.predict_(x)


def plotting_regression(data, y_hat, settings):
    x = np.array(data).reshape(-1, 1)
    plot_prediction(x, y_sell_price, y_hat, settings)
    print()


def training_multivariate_model(myLR, data):
    myLR.fit_(data, y_sell_price)
    return myLR.predict_(data)


data = pd.read_csv("spacecraft_data.csv")
y_sell_price = np.array(data['Sell_price']).reshape(-1, 1)

# Plotting univariate regression for Age and Sell price
thetas = np.array([[620.0], [-10.0]])
alpha = 2.5e-5
max_iter = 100000
myLR_age = MyLR(thetas, alpha, max_iter)
field = 'Age'
settings_age = {'clr_real': '#0310BF', 'clr_pred': '#3A89FF',
                'x_label': '$x_{1}$: age (in years)',
                'y_label': 'y: sell price (in keuros)',
                'anchor': (0, 0),
                'loc': 'lower left',
                }
y_hat = training_univariate_model(myLR_age, data[field])
print_param_and_error(myLR_age, field, y_hat, 'Univariate(Age)')
plotting_regression(data[field], y_hat, settings_age)

# Plotting univariate regression for Thrust and Sell price
thetas = np.array([[-10.0], [10.0]])
alpha = 1.0e-4
max_iter = 200000
myLR_thrust = MyLR(thetas, alpha, max_iter)
field = 'Thrust_power'
settings_thrust = {'clr_real': '#348200', 'clr_pred': '#96FF0E',
                   'x_label': '$x_{2}$: thrust power (in 10 Km/s)',
                   'y_label': 'y: sell price (in keuros)',
                   'anchor': (0, 1),
                   'loc': 'upper left',
                   }
y_hat = training_univariate_model(myLR_thrust, data[field])
print_param_and_error(myLR_thrust, field, y_hat, 'Univariate(Thrust)')
plotting_regression(data[field], y_hat, settings_thrust)

# Plotting univariate regression for Thrust and Sell price
thetas = np.array([[800.0], [-8.0]])
myLR_distance = MyLR(thetas, alpha, max_iter)
field = 'Terameters'
settings_dist = {'clr_real': '#9D00FF', 'clr_pred': '#E7C0FF',
                 'x_label':
                 '$x_{3}$: distance totalizer value of spacecraft(in Tmeters)',
                 'y_label': 'y: sell price (in keuros)',
                 'anchor': (1, 1),
                 'loc': 'upper right',
                 }
y_hat = training_univariate_model(myLR_distance, data[field])
print_param_and_error(myLR_distance, field, y_hat, 'Univariate(Terameters)')
plotting_regression(data[field], y_hat, settings_dist)

# Plotting multivariate regression
thetas = np.array([[1.0], [1.0], [1.0], [1.0]])
alpha = 2.5e-5
my_lreg = MyLR(thetas, alpha)
modified_data = data.drop(labels='Sell_price', axis=1).to_numpy()
y_hat = training_multivariate_model(my_lreg, modified_data)
print_param_and_error(my_lreg, 'multivariate', y_hat, 'Multivariate')
plotting_regression(data['Age'], y_hat, settings_age)
plotting_regression(data['Thrust_power'], y_hat, settings_thrust)
plotting_regression(data['Terameters'], y_hat, settings_dist)
