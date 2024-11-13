# Values Used:
# - X: Feature matrix (including intercept term). Example: X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
# - y: Target labels or values. Example: y = [0, 0, 1, 1, 1]
# - theta: Parameters initialized to zero. Example: theta = [0, 0]
# - learning_rate: Step size for gradient descent. Example: learning_rate = 0.1
# - iterations: Number of times the algorithm will run. Example: iterations = 1000
# - epsilon: Small constant to prevent division by zero. Example: epsilon = 1e-8

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost function for Logistic Regression
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Adagrad Gradient Descent for Logistic Regression
def adagrad(X, y, theta, learning_rate, iterations, epsilon=1e-8):
    m = len(y)
    grad_squared = np.zeros_like(theta)
    cost_history = []
    
    for i in range(iterations):
        # Compute the gradient
        gradient = (1 / m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        
        # Update the squared gradients
        grad_squared += gradient ** 2
        
        # Update theta with Adagrad
        theta -= (learning_rate / (np.sqrt(grad_squared) + epsilon)) * gradient
        
        # Track the cost
        cost_history.append(compute_cost(X, y, theta))
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]}, Theta = {theta}")
    
    return theta, cost_history

# Logistic Regression with Adagrad
def logistic_regression_adagrad(X, y, learning_rate=0.1, iterations=1000):
    theta = np.zeros(X.shape[1])
    theta, cost_history = adagrad(X, y, theta, learning_rate, iterations)
    return theta, cost_history

# Sample data
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Add intercept term
y = np.array([0, 0, 1, 1, 1])

# Run Adagrad Logistic Regression
theta, cost_history = logistic_regression_adagrad(X, y, learning_rate=0.1, iterations=1000)
print(f"Optimized theta: {theta}")

# Plot cost function history in 2D
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function during Adagrad')
plt.grid(True)
plt.show()
