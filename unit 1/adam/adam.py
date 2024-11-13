from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function (Rosenbrock)
def objective(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Derivative of the Rosenbrock function (gradient)
def derivative(x, y):
    dfdx = -2 * (1 - x) - 400 * x * (y - x**2)
    dfdy = 200 * (y - x**2)
    return asarray([dfdx, dfdy])

# Adam optimization algorithm
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    solutions = list()
    # Generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    
    # Initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    
    # Run the gradient descent updates
    for t in range(n_iter):
        # Calculate gradient g(t)
        g = derivative(x[0], x[1])
        
        # Update the solution one variable at a time
        for i in range(bounds.shape[0]):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
            # mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1**(t+1))
            # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
        
        # Evaluate candidate point
        score = objective(x[0], x[1])
        
        # Keep track of solutions
        solutions.append(x.copy())
        
        # Report progress
        print(f'> Iteration {t} f({x}) = {score:.5f}')
    
    return solutions

# Seed the pseudo-random number generator
seed(1)

# Define the range for input
bounds = asarray([[-2.0, 2.0], [-1.0, 3.0]])

# Define the total iterations
n_iter = 60

# Step size (learning rate)
alpha = 0.02

# Factor for average gradient (beta1)
beta1 = 0.8

# Factor for average squared gradient (beta2)
beta2 = 0.999

# Perform the gradient descent search with Adam
solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)

# Convert solutions to a numpy array for easy indexing
solutions = asarray(solutions)

# Create a simple plot of the optimization path in 2D
plt.figure(figsize=(8, 6))

# Plot the objective function values at each iteration
plt.plot(range(n_iter), [objective(sol[0], sol[1]) for sol in solutions], label="Objective Function Value", marker='o')

plt.title('Objective Function Value vs Iteration')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Create a 2D plot of the optimization path
plt.figure(figsize=(8, 6))
plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='r', label="Optimization Path")
plt.title('Adam Optimization Path on Rosenbrock Function (2D)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Create a 3D plot of the Rosenbrock function surface and optimization path
x_vals = arange(-2.0, 2.0, 0.1)
y_vals = arange(-1.0, 3.0, 0.1)
X, Y = meshgrid(x_vals, y_vals)
Z = objective(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot the optimization path in 3D
ax.plot(solutions[:, 0], solutions[:, 1], [objective(x, y) for x, y in solutions], 
        color='r', marker='o', label="Optimization Path", markersize=5)

# Labels and title
ax.set_title('Adam Optimization Path on Rosenbrock Function (3D)')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('Objective Function Value')
ax.legend()

# Show the plot
plt.show()
