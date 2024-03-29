from math import floor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from my_linear_regression import MyLinearRegression as MyLR


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


def plot_prediction(x, y, mlr):
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
    y_hat = mlr.predict_(x)
    fig, ax = plt.subplots()
    ax.plot(x, y_hat, 'X', color='springgreen', label='$S_{predict}(pills)$')
    ax.plot(x, y_hat, '--', color='springgreen')
    ax.plot(x, y, 'o', color='deepskyblue', label='$S_{true}(pills)$')
    ax.set_xlabel("Quantity of blue pill(in micrograms)")
    ax.set_ylabel("Space driving score")
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.legend(bbox_to_anchor=(0, 1.1), loc='upper left', frameon=False, ncol=2)
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


data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1, 1)
Yscore = np.array(data['Score']).reshape(-1, 1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
Y_model1 = linear_model1.predict_(Xpill)
print(linear_model1.mse_(Yscore, Y_model1))
plot_prediction(Xpill, Yscore, linear_model1)
plot_loss(Xpill, Yscore, linear_model1)

linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model2 = linear_model2.predict_(Xpill)
print(linear_model2.mse_(Yscore, Y_model2))
plot_prediction(Xpill, Yscore, linear_model2)
plot_loss(Xpill, Yscore, linear_model2)
