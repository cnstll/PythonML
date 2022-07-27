import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features as add_pf


def display_mse_bar(mse_values):
    fig, ax = plt.subplots(label='MSE according to regression degree')
    ax.bar(np.arange(1, 7), mse_values, color='#3A89FF', edgecolor='gray')
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=3)
    for p in ax.patches:
        w = p.get_width()
        h = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f"{round(h)}", ((x + w / 2), y + h * 1.02), ha='center',
                    fontweight='demi')


def plot_regression(ax, x, y, mlr, power):
    continuous_x = np.arange(min(x), max(x) + 0.01, 0.01).reshape(-1, 1)
    _x = add_pf(continuous_x, power)
    continuous_y = mlr.predict_(_x)
    ax.scatter(x, y)
    ax.plot(continuous_x, continuous_y, color='orange')
    ax.set_title(f'Regression Degree {power}', size='x-small',
                 va='center', fontweight='demi')
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=3)


# Extracting the data from csv to dataframe
data = pd.read_csv('are_blue_pills_magics.csv')
x = data['Micrograms'].to_numpy().reshape(-1, 1)
y = data['Score'].to_numpy().reshape(-1, 1)

# Model trainings and predictions
# First Degree model
alpha = 2.5e-4
max_iter = 100000
theta1 = np.array([[330], [-50]])
mlr1 = MLR(theta1, alpha, max_iter)
mlr1.fit_(x, y)
y_hat1 = mlr1.predict_(x)
mse1 = mlr1.mse_(y, y_hat1)

# Second Degree model
alpha = 2.5e-4
max_iter = 100000
theta2 = np.array([[150], [-40], [10]])
mlr2 = MLR(theta2, alpha, max_iter)
x2 = add_pf(x, 2)
mlr2.fit_(x2, y)
y_hat2 = mlr2.predict_(x2)
mse2 = mlr2.mse_(y, y_hat2)


# Third Degree model
alpha = 5.0e-5
max_iter = 500000
theta3 = np.array([[350], [-300], [100], [-5]])
mlr3 = MLR(theta3, alpha, max_iter)
x3 = add_pf(x, 3)
mlr3.fit_(x3, y)
y_hat3 = mlr3.predict_(x3)
mse3 = mlr3.mse_(y, y_hat3)

# Fourth Degree model
alpha = 1.0e-6
max_iter = 100000
theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
mlr4 = MLR(theta4, alpha, max_iter)
x4 = add_pf(x, 4)
mlr4.fit_(x4, y)
y_hat4 = mlr4.predict_(x4)
mse4 = mlr4.mse_(y, y_hat4)

# Fifth Degree model
alpha = 1.0e-8
max_iter = 100000
theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
mlr5 = MLR(theta5, alpha, max_iter)
x5 = add_pf(x, 5)
mlr5.fit_(x5, y)
y_hat5 = mlr5.predict_(x5)
mse5 = mlr5.mse_(y, y_hat5)

# Sixth Degree model
alpha = 1.0e-9
max_iter = 100000
theta6 = np.array([[9110], [-18015], [13400],
                   [-4935], [966], [-96.4], [3.86]]).reshape(-1, 1)
mlr6 = MLR(theta6, alpha, max_iter)
x6 = add_pf(x, 6)
mlr6.fit_(x6, y)
y_hat6 = mlr6.predict_(x6)
mse6 = mlr6.mse_(y, y_hat6)

# Printing and plotting mse value for each model
print(mse1, mse2, mse3, mse4, mse5, mse6)
mse_values = np.array([mse1, mse2, mse3, mse4, mse5, mse6])
display_mse_bar(mse_values)

# Plotting each model
all_mlr = [mlr1, mlr2, mlr3, mlr4, mlr5, mlr6]
half_len = len(all_mlr) / 2
title = 'Models of the relation between qtt of pills taken and test scores'
fig, axs = plt.subplots(2, 3, label=title)
for i, lr in enumerate(all_mlr):
    if i < half_len:
        plot_regression(axs[0, i], x, y, lr, i + 1)
    else:
        plot_regression(axs[1, int(i % half_len)], x, y, lr, i + 1)
plt.show()
