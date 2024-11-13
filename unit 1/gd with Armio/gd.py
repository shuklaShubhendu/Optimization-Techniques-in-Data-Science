# gd with Armio rule exaple

# use Rosenbrock function
# (x)=100(x 2​−x 1 2​) 2 +(1−x 1​) 2

# Armijo Rule for Step Size
# Libraries Used:
# - numpy: For numerical operations and handling arrays.
# - matplotlib.pyplot: For plotting 2D graphs.

# Functions:
# - rosenbrogd with Armio rule exaple

# use Rosenbrock function
# (x)=100(x 2​−x 1 2​) 2 +(1−x 1​) 2

# Armijo Rule for Step Sizeck: The Rosenbrock function.
# - gradient: Computes the gradient of the Rosenbrock function.
# - armijo_rule: Selects the step size using the Armijo rule.
# - gradient_descent_armijo: Performs gradient descent using the Armijo rule.

# Parameters:
# - x_init: Initial guess for optimization.
# - max_iters: Maximum iterations for the descent.
# - tol: Convergence tolerance.

import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def gradient(xy):
    x, y = xy
    grad_x = -400 * x * (y - x**2) - 2 * (1 - x)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])

def armijo_rule(x, d, grad, func, beta=0.5, sigma=1e-4):
    alpha = 1.0
    while func(x[0] + alpha * d[0], x[1] + alpha * d[1]) > func(x[0], x[1]) + sigma * alpha * np.dot(grad, d):
        alpha *= beta
    return alpha

def gradient_descent_armijo(x_init, func, max_iters=1000, tol=1e-6):
    path = [x_init]
    x = x_init
    for i in range(max_iters):
        grad = gradient(x)
        d = -grad
        alpha = armijo_rule(x, d, grad, func)
        x_new = x + alpha * d
        path.append(x_new)

        # Print values at each iteration
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {func(x_new[0], x_new[1])}")

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

# Run gradient descent with Armijo rule
path = gradient_descent_armijo(x_init, rosenbrock)

# Plot the contour of the Rosenbrock function
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
plt.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, label='Gradient Descent Path')
plt.title('Gradient Descent with Armijo Rule on Rosenbrock Function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
