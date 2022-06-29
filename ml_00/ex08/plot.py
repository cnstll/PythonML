import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../ex06')
from loss import predict_, loss_


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    j_value = loss_(y, y_hat) * 2
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.plot(x, y_hat, '-')
    for px, py, py_hat in zip(x, y, y_hat):
        ax.plot([px, px], [py_hat, py], ':', color='red')
    ax.set_title(f"Cost: {j_value:.6f}")
    plt.show()


x = np.arange(1, 6).reshape(-1, 1)
y = np.array([[11.52434424],
             [10.62589482], [13.14755699], [18.60682298], [14.14329568]])

theta1 = np.array([[18], [-1]])
plot_with_loss(x, y, theta1)

theta2 = np.array([[14], [0]])
plot_with_loss(x, y, theta2)

theta3 = np.array([[12], [0.8]])
plot_with_loss(x, y, theta3)
