import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rosenbrock function
def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

# Gradient of the Rosenbrock function
def gradient(xy):
    x, y = xy
    grad_x = -400 * x * (y - x**2) - 2 * (1 - x)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])

# Armijo Rule for step size
def armijo_rule(x, d, grad, func, beta=0.5, sigma=1e-4):
    alpha = 1.0
    while func(x[0] + alpha * d[0], x[1] + alpha * d[1]) > func(x[0], x[1]) + sigma * alpha * np.dot(grad, d):
        alpha *= beta  # Reduce alpha if the Armijo condition is not satisfied
    return alpha

# Gradient Descent with Armijo Rule
def gradient_descent_armijo(x_init, func, max_iters=1000, tol=1e-6):
    path = [x_init]
    x = x_init
    for i in range(max_iters):
        grad = gradient(x)  # Compute the gradient at the current point
        d = -grad  # Descent direction (negative of gradient)
        alpha = armijo_rule(x, d, grad, func)  # Compute step size using Armijo rule
        x_new = x + alpha * d  # Update the point
        path.append(x_new)

        # Print values at each iteration
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {func(x_new[0], x_new[1])}")

        # Convergence check: If the change in x is small enough, stop the algorithm
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(path)

# Define mesh grid for contour plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Initial guess for optimization
x_init = np.array([-1.2, 1.0])

# Run gradient descent with Armijo rule and get the path
path = gradient_descent_armijo(x_init, rosenbrock)

# Create a figure with two subplots (side by side)
fig = plt.figure(figsize=(16, 8))

# 2D Contour Plot
ax1 = fig.add_subplot(121)
ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
ax1.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (2D)')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')

# 3D Surface Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='jet', alpha=0.7)
ax2.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (3D)')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$f(x_1, x_2)$')

# Loop over the path to update the plots at each iteration
for i in range(len(path)):
    # Update the 2D contour plot
    ax1.clear()
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    ax1.plot(path[:i+1, 0], path[:i+1, 1], 'ro-', markersize=4, label='Gradient Descent Path')
    ax1.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (2D)')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.legend()

    # Update the 3D surface plot
    ax2.clear()
    ax2.plot_surface(X, Y, Z, cmap='jet', alpha=0.7)
    ax2.plot(path[:i+1, 0], path[:i+1, 1], [rosenbrock(x[0], x[1]) for x in path[:i+1]], 'ro-', markersize=4, label='Gradient Descent Path')
    ax2.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (3D)')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$f(x_1, x_2)$')
    ax2.legend()

    # Pause to update the plot
    plt.pause(0.1)

# Display the final result
plt.tight_layout()
plt.show()
