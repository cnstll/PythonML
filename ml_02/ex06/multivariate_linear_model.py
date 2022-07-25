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


def plot_prediction(x, y, mlr, settings):
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
    ax.plot(x, y, 'o', color=settings['clr_real'], label='Sell price')
    ax.plot(x, y_hat, 'o', color=settings['clr_pred'], label='Predicted sell price', markersize=2)
    ax.set_xlabel(settings['x_label'])
    ax.set_ylabel(settings['y_label'])
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.legend(bbox_to_anchor=settings['anchor'], loc=settings['loc'], frameon=True, ncol=1)
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


def univariate_plotting(field, thetas, settings):
    x = np.array(data[field]).reshape(-1, 1)
    myLR = MyLR(thetas)
    myLR.fit_(x, y_sell_price)
    print(myLR.thetas)
    y_hat = myLR.predict_(x)
    plot_prediction(x, y_sell_price, myLR, settings)
    print(f"mse_{field.lower()}: {myLR.mse_(y_hat, y_sell_price)}")


data = pd.read_csv("spacecraft_data.csv")
y_sell_price = np.array(data['Sell_price']).reshape(-1, 1)

# Plotting univariate regression for Age and Sell price
thetas = np.array([[620.0], [-8.0]])
settings = {'clr_real': '#FFE599', 'clr_pred': '#E4610F',
            'x_label': '$x_{1}$: age (in years)',
            'y_label': 'y: sell price (in keuros)',
            'anchor' : (0, 0),
            'loc' : 'lower left',           
           }
univariate_plotting('Age', thetas, settings)

# Plotting univariate regression for Thrust and Sell price
thetas = np.array([[0.0], [8.0]])
settings = {'clr_real': '#E7C0FF', 'clr_pred': '#9D00FF',
            'x_label': '$x_{2}$: thrust power (in 10 Km/s)',
            'y_label': 'y: sell price (in keuros)',
            'anchor' : (0, 1),
            'loc' : 'upper left',
           }
univariate_plotting('Thrust_power', thetas, settings)

# Plotting univariate regression for Thrust and Sell price
thetas = np.array([[400.0], [-8.0]])
settings = {'clr_real': '#96FF0E', 'clr_pred': '#348200',
            'x_label': '$x_{2}$: distance totalizer value of spacecraft (in Tmeters)',
            'y_label': 'y: sell price (in keuros)',
            'anchor' : (1, 1),
            'loc' : 'upper right',
           }
univariate_plotting('Terameters', thetas, settings)

# 
# linear_model2 = MyLR(np.array([[89.0], [-6]]))
# Y_model2 = linear_model2.predict_(Xpill)
# print(linear_model2.mse_(Yscore, Y_model2))
# plot_prediction(Xpill, Yscore, linear_model2)
# plot_loss(Xpill, Yscore, linear_model2)
