# This program demonstrates gradient descent on the function f(v) = v^2.
# It includes the following components:
#
# 1. Import Statements:
#    - Import required libraries: numpy for numerical operations and matplotlib for plotting.
#
# 2. Gradient Descent Functions:
#    - `gradient_descent`: Finds the final minimum position by iteratively adjusting the vector 
#      based on the gradient, learning rate, and initial position.
#    - `gradient_descent_collect`: Collects each vector position at each step, which is useful 
#      for visualizing the gradient descent path.
#    - `gradient_function`: Defines the gradient of the function f(v) = v^2, which is used 
#      in the gradient descent calculations.
#
# 3. Parameters:
#    - Starting position (start_value), learning rate (learning_rate), maximum iterations 
#      (max_iterations), and tolerance (tolerance) are defined to control the gradient descent process.
#
# 4. Gradient Descent Execution:
#    - `gradient_descent_collect` is called with the specified parameters to perform gradient 
#      descent and collect each step's values for visualization.
#
# 5. Visualization:
#    - A 2D plot is created to show the function f(v) = v^2 and the gradient descent path.
#    - The plot includes:
#      - The function f(v) = v^2 represented as a smooth curve.
#      - Points indicating each step of the gradient descent process.
#      - Labels, title, legend, and grid for readability.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent(gradient, start, learn_rate, n_iter):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
    return vector

def gradient_descent_collect(gradient, start, learn_rate, n_iter, tolerance):
    vector = start
    history = [vector]
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        history.append(vector)
    return history

def gradient_function(v):
    return 2 * v

# Parameters
start_value = 10.0
learning_rate = 0.2
max_iterations = 50
tolerance = 1e-06

# Perform gradient descent and collect the path
values = gradient_descent_collect(
    gradient=gradient_function,
    start=start_value,
    learn_rate=learning_rate,
    n_iter=max_iterations,
    tolerance=tolerance
)

# 2D Plot
plt.figure(figsize=(10, 6))
v_range = np.linspace(-10, 10, 400)
plt.plot(v_range, v_range**2, label='f(v) = v^2', color='blue')

# Plot gradient descent path
vectors = np.array(values)
function_values = vectors**2
plt.scatter(vectors, function_values, color='red', zorder=5, label='Gradient Descent Path', s=50)

# Configure 2D plot
plt.xlabel('Vector Value (v)')
plt.ylabel('Function Value f(v) = v^2')
plt.title('2D Visualization of Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for 3D plot
iterations = np.arange(len(values))  # x-axis: iteration number
vectors = np.array(values)  # y-axis: vector values
function_values = vectors**2  # z-axis: f(v) = v^2

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface (parabola f(v) = v^2)
v_range = np.linspace(-10, 10, 100)
V, I = np.meshgrid(v_range, iterations)
Z = V**2
ax.plot_surface(I, V, Z, cmap='viridis', alpha=0.6)

# Plot the gradient descent steps
ax.scatter(iterations, vectors, function_values, color='r', s=50, label='Gradient Descent Path')

# Configure 3D plot
ax.set_xlabel('Iterations')
ax.set_ylabel('Vector Value (v)')
ax.set_zlabel('Function Value f(v) = v^2')
ax.set_title('3D Visualization of Gradient Descent')

plt.legend()
plt.show()
