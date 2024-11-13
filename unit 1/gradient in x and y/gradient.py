# Libraries Used:
# - numpy: Used for mathematical operations and handling arrays.
# - matplotlib.pyplot: Used for creating 2D graphs and visualizations.
# - mpl_toolkits.mplot3d: Used for creating 3D surface plots.

# Functions:
# - gradient_descent: Implements the gradient descent algorithm to find the minimum of a function.
# - func: Defines the function to minimize (f(x, y) = x^2 + y^2).
# - grad_func: Defines the gradient (partial derivatives) of the function, used in the gradient descent.

# Parameters:
# - start_value: The starting point for gradient descent (a list of x and y values).
# - learning_rate: The step size used for each iteration of gradient descent.
# - max_iterations: The maximum number of iterations for the gradient descent to run.
# - tolerance: The minimum change in value for the gradient descent to stop.

# Visualization:
# - run_and_visualize: A helper function that runs the gradient descent and generates both 2D contour
#   and 3D surface plots to visualize the function and the gradient descent path.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gradient Descent function
def gradient_descent(
    gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06,
    dtype="float64"
):
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")

    dtype_ = np.dtype(dtype)
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")

    vector = np.array(start, dtype=dtype_)
    learn_rate = np.array(learn_rate, dtype=dtype_)
    
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    for _ in range(n_iter):
        diff = -learn_rate * np.array(gradient(vector), dtype_)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff

    return vector if vector.shape else vector.item()

# Define function f(x, y) = x^2 + y^2
def func(x, y):
    return x**2 + y**2

# Define the gradient of the function f(x, y) = x^2 + y^2
def grad_func(point):
    x0, y0 = point
    return np.array([2 * x0, 2 * y0])

# Parameters for Gradient Descent
start_value = [3.0, 3.0]  # Starting point for gradient descent
learning_rate = 0.1
max_iterations = 100
tolerance = 1e-6

# Define a helper function to run gradient descent and visualize results
def run_and_visualize(func, grad_func, start_value):
    # Create a grid of x and y values for visualization
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # Run the gradient descent
    final_point = gradient_descent(
        gradient=grad_func,
        x=x_range,
        y=y_range,
        start=start_value,
        learn_rate=learning_rate,
        n_iter=max_iterations,
        tolerance=tolerance
    )

    # Print the final point obtained from gradient descent
    print(f"Final point from gradient descent:", final_point)

    # Prepare function values for visualization
    Z_final = func(final_point[0], final_point[1])

    # 2D Visualization
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)  # Contour plot of the function
    plt.colorbar(label='f(x, y) = x^2 + y^2')

    # Plot the final point
    plt.plot(final_point[0], final_point[1], 'ro', label='Final Point', markersize=10)

    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Gradient Descent Visualization for f(x, y) = x^2 + y^2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3D Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Plot the final point in 3D
    ax.scatter(final_point[0], final_point[1], Z_final, color='red', s=100, label='Final Point')

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Function Value')
    ax.set_title('3D Visualization of Gradient Descent for f(x, y) = x^2 + y^2')

    plt.legend()
    plt.show()

# Run and visualize gradient descent for the function f(x, y) = x^2 + y^2
run_and_visualize(func, grad_func, start_value)
